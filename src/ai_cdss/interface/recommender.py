import datetime
import logging
import uuid
from typing import Dict, List

import pandas as pd
from ai_cdss.cdss import CDSS
from ai_cdss.constants import BY_PP, PATIENT_ID, PPF_PARQUET_FILEPATH
from ai_cdss.loaders import DataLoader
from ai_cdss.models import DataUnitSet
from ai_cdss.processing import (
    ClinicalSubscales,
    DataProcessor,
    ProtocolToClinicalMapper,
)
from ai_cdss.processing.features import compute_ppf
from rgs_interface.data.schemas import PrescriptionStagingRow, RecsysMetricsRow

logger = logging.getLogger(__name__)


class CDSSInterface:

    def __init__(self, loader: DataLoader, processor: DataProcessor):
        self.loader = loader
        self.processor = processor

    def recommend_for_study(
        self, study_id: List[int], n: int, days: int, protocols_per_day: int
    ) -> Dict[str, str]:
        """
        Generate recommendations for all patients in a study.

        Args:
            study_id: Identifier for the study cohort.
            n: Number of protocols to recommend.
            days: Number of days to plan for.
            protocols_per_day: Number of protocols per day.

        Returns:
            A dictionary with a success message.
        """
        # Loading
        logger.info(f"Starting recommendation generation for study: {study_id}")

        patient_data = self.loader.interface.fetch_patients_by_study(study_ids=study_id)

        if patient_data is None or patient_data.empty:  # No patients for given study
            raise ValueError(f"No patients found for study ID: {study_id}")
        patient_list = patient_data["PATIENT_ID"].tolist()
        logger.debug(f"Fetched {len(patient_list)} patients.")

        # PPF is a requirement in this pipline
        ppf = self.loader.load_ppf_data(patient_list)
        missing = ppf.metadata.get("missing_patients", [])
        if missing:
            logger.info(f"Running PPF computation for patients: {missing}")
            self.compute_patient_fit(missing)

        session = self.loader.load_session_data(patient_list)
        protocol_similarity = self.loader.load_protocol_similarity()

        rgs_data = DataUnitSet([session, ppf])

        # Processing
        scores = self.processor.process_data(rgs_data)
        cdss = CDSS(scoring=scores, n=n, days=days, protocols_per_day=protocols_per_day)

        unique_id = uuid.uuid4()
        datetime_now = datetime.datetime.now()

        # Recommendations
        for patient in patient_list:

            recommendations = cdss.recommend(patient, protocol_similarity)

            # Transform dataframes
            prescription_df = recommendations.explode("DAYS").rename(
                columns={"DAYS": "WEEKDAY"}
            )
            metrics_df = pd.melt(
                recommendations,
                id_vars=BY_PP,
                value_vars=["DELTA_DM", "ADHERENCE_RECENT", "PPF"],
                var_name="KEY",
                value_name="VALUE",
            )

            # Save prescriptions
            for _, row in prescription_df.iterrows():
                self.loader.interface.add_prescription_staging_entry(
                    PrescriptionStagingRow.from_row(
                        row, recommendation_id=unique_id, start=datetime_now
                    )
                )

            # Save metrics
            for _, row in metrics_df.iterrows():
                self.loader.interface.add_recsys_metric_entry(
                    RecsysMetricsRow.from_row(
                        row, recommendation_id=unique_id, metric_date=datetime_now
                    )
                )

        logger.info(f"Successfully generated recommendations for study {study_id}")
        return {"message": f"Recommendations generated for study {study_id}"}

    def compute_patient_fit(self, patient_id: List[int]) -> dict:
        """
        Compute and persist the Patient-Protocol Fit (PPF) matrix for a single patient.
        """
        patient = self.loader.load_patient_subscales(patient_id)

        try:
            patient = patient.loc[
                patient_id
            ]  #### Remove after load_patient_subscales db implementation
        except KeyError:
            raise ValueError(f"Patient clinical data not found for ID: {patient_id}")

        if patient.empty:
            raise ValueError(f"Patient data not found for ID: {patient_id}")

        protocol = self.loader.load_protocol_attributes()
        if protocol.empty:
            raise ValueError("Protocol data could not be loaded.")

        patient_def = ClinicalSubscales().compute_deficit_matrix(patient)
        protocol_map = ProtocolToClinicalMapper().map_protocol_features(protocol)

        missing_subscales = protocol_map.columns.difference(patient.columns)
        if not missing_subscales.empty:
            raise ValueError(
                f"Patient data is missing required subscales: {', '.join(missing_subscales)}"
            )
        patient_def = patient_def[protocol_map.columns]

        ppf, contrib = compute_ppf(patient_def, protocol_map)
        ppf_contrib = pd.merge(ppf, contrib, on=BY_PP, how="left")
        ppf_contrib.attrs = {"SUBSCALES": list(protocol_map.columns)}

        if not ppf_contrib.empty:
            try:
                if not PPF_PARQUET_FILEPATH.exists():
                    ppf_contrib.to_parquet(PPF_PARQUET_FILEPATH, index=False)
                else:
                    existing = pd.read_parquet(PPF_PARQUET_FILEPATH)
                    keys = ppf_contrib[BY_PP]
                    merged = existing.merge(keys, on=BY_PP, how="left", indicator=True)
                    filtered = existing[merged["_merge"] == "left_only"]
                    updated = pd.concat([filtered, ppf_contrib], ignore_index=True)
                    updated.attrs = {"SUBSCALES": list(protocol_map.columns)}
                    updated.to_parquet(PPF_PARQUET_FILEPATH)

                return {
                    "message": f"Computation successful for patient {patient_id}",
                    "patient_id": patient_id,
                    "subscales_used": list(protocol_map.columns),
                    "saved_to": str(PPF_PARQUET_FILEPATH.absolute()),
                }

            except Exception as e:
                raise RuntimeError(f"Failed to save results to Parquet: {e}")

        else:
            raise ValueError(f"No PPF data to save for patient {patient_id}")
