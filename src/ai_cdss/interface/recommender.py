import datetime
from datetime import timedelta
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd
from ai_cdss.cdss import CDSS
from ai_cdss.constants import (
    BY_PP,
    CLINICAL_START,
    DAYS,
    DELTA_DM,
    PATIENT_ID,
    PPF,
    RECENT_ADHERENCE,
    SCORE,
    SESSION_INDEX,
    USAGE,
    USAGE_WEEK,
    WEEKS_SINCE_START,
    DEFAULT_DEBUG_DIR
)
from ai_cdss.models import DataUnitName
from ai_cdss.loaders import DataLoader
from ai_cdss.processing import DataProcessor
from ai_cdss.services.data_preparation import RecommendationDataService
from ai_cdss.services.ppf_service import PPFService
from ai_cdss.services.protocol_similarity import ProtocolSimilarityService
from ai_cdss.interface.debug import DebugReport
from rgs_interface.data.schemas import PrescriptionStagingRow, RecsysMetricsRow

logger = logging.getLogger(__name__)


class CDSSInterface:
    """
    Main orchestrator for generating clinical decision support recommendations.
    Coordinates data preparation, processing, and persistence for study cohorts.
    """

    def __init__(
        self,
        loader: DataLoader,
        processor: DataProcessor,
        data_service: Optional[RecommendationDataService] = None,
        ppf_service: Optional[PPFService] = None,
        debug: bool = False,
    ):
        self.loader = loader
        self.processor = processor
        self.ppf_service = ppf_service or PPFService(loader)
        self.data_service = data_service or RecommendationDataService(loader)
        self.protocol_similarity_service = ProtocolSimilarityService(loader)
        self.debug = debug
        if self.debug:
            self.debug_service = DebugReport(DEFAULT_DEBUG_DIR)

    def recommend_for_patients(
        self,
        patient_ids: List[int],
        n: int,
        days: int,
        protocols_per_day: int,
        scoring_date: Optional[pd.Timestamp] = None,
    ) -> Dict[str, Any]:
        """
        Run recommendations for one **or many** patients.
        Returns the same structure as recommend_for_study, with 'per_patient' detailing each patient's result.
        """
        return self._recommend_for_patients_core(
            patient_ids,
            n=n,
            days=days,
            protocols_per_day=protocols_per_day,
            scoring_date=scoring_date,
            context={
                "patient_id": patient_ids, 
                "message": f"Recommendations generated for patients {patient_ids}"
            },
        )

    def recommend_for_study(
        self,
        study_id: List[int],
        n: int,
        days: int,
        protocols_per_day: int,
        scoring_date: Optional[pd.Timestamp] = None,
    ) -> Dict[str, Any]:
        """
        Cohort/study run.
        """
        # Keep validation where it belongs
        patient_ids = self.loader.fetch_and_validate_patients(study_ids=study_id)
        return self._recommend_for_patients_core(
            patient_ids,
            n=n,
            days=days,
            protocols_per_day=protocols_per_day,
            scoring_date=scoring_date,
            context={"study_id": study_id, "message": f"Recommendations generated for study {study_id}"},
        )

    def _recommend_for_patients_core(
        self,
        patient_ids: List[int],
        *,
        n: int,
        days: int,
        protocols_per_day: int,
        scoring_date: Optional[pd.Timestamp],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Internal core that runs the full pipeline for a given set of patient_ids.
        `context` can include study_id or other metadata to echo back in the response.
        """
        logger.info("Starting recommendation generation for patients: %s", patient_ids)
        start_time = time.time()
        unique_id = uuid.uuid4()
        datetime_now = datetime.datetime.now()

        try:
            if not patient_ids:      # catches None or empty list
                elapsed = time.time() - start_time
                payload = {
                    "status": "warning",
                    "run_id": str(unique_id),
                    "patients_processed": 0,
                    "total_recommendations": 0,
                    "per_patient": [],
                    "start_time": datetime_now.isoformat(),
                    "elapsed_seconds": elapsed,
                    **context,
                    "message": (
                        context.get("message")
                        or f"No patients provided or resolved for context={context}"
                    ),
                }
                logger.info("No patients to process. Context: %s | Result: %s", context, payload)
                return payload

            rgs_data, protocol_similarity = self.data_service.prepare(patient_list=patient_ids)
            scores = self.processor.process_data(rgs_data, scoring_date or pd.Timestamp.today())
            cdss = CDSS(scoring=scores, n=n, days=days, protocols_per_day=protocols_per_day)
            # Patient start date dict
            patient_data = rgs_data.get(DataUnitName.PATIENT).data
            # Dictionary with patient_id as key and clinical_start as value
            patient_dict = dict(zip(patient_data[PATIENT_ID], patient_data[CLINICAL_START]))

            patient_results = [
                self._process_patient(p, cdss, protocol_similarity, scores, unique_id, patient_dict[p]) # start date instead of datetime_now
                for p in patient_ids
            ]
            total_recommendations = sum(r.get("num_recommendations", 0) for r in patient_results)
            elapsed = time.time() - start_time

            payload = {
                "status": "success",
                "run_id": str(unique_id),
                "patients_processed": len(patient_ids),
                "total_recommendations": total_recommendations,
                "per_patient": patient_results,
                "start_time": datetime_now.isoformat(),
                "elapsed_seconds": elapsed,
                **context,
            }

            if self.debug:
                payload["debug"] = {
                    "scores": {
                        "file": self.debug_service.dump_df(scores, unique_id, "scores.csv"),
                        "preview": self.debug_service.preview_df(scores),
                    }
                }

            logger.info("Successfully generated recommendations. Context: %s", context)
            return payload

        except Exception as e:
            logger.error(
                "Failed to generate recommendations. Context=%s Error=%s (%s)",
                context, e, type(e).__name__, exc_info=True
            )
            return {
                "status": "failure",
                "error": f"{type(e).__name__}: {e}",
                **context,
                "message": f"Failed to generate recommendations",
            }

    def _process_patient(
        self,
        patient,
        cdss,
        protocol_similarity,
        scores,
        unique_id,
        datetime_start,
    ):
        """
        Process recommendations and metrics for a single patient.

        This helper handles recommendation generation, prescription and metrics transformation,
        persistence, and error handling for one patient in a batch run.

        Args:
            patient: The patient ID to process.
            cdss: The CDSS instance for generating recommendations.
            protocol_similarity: Protocol similarity data for recommendations.
            scores: DataFrame of all scored protocols.
            unique_id: UUID for this batch run.
            datetime_start: Timestamp for trial start.

        Returns:
            dict: A result dictionary with patient_id, num_recommendations, status, and (on failure) error.
        """
        try:
            datetime_now = datetime.datetime.now()
            recommendations = cdss.recommend(patient, protocol_similarity)
            prescription_df = self._transform_recommendations(recommendations)


            patient_scores = scores[scores[PATIENT_ID] == patient]
            all_metrics_df = self._transform_metrics(patient_scores)

            if not self.debug:
                self._save_prescriptions(prescription_df, unique_id, datetime_start)
                self._save_metrics(all_metrics_df, unique_id, datetime_now)

            result = {
                "patient_id": patient,
                "num_recommendations": len(recommendations),
                "status": "success",
            }

            if self.debug:
                logger.info("Debug mode enabled - skipping persistence to db for patient %s", patient)
                artifacts = self.debug_service.make_artifacts(
                    run_id=unique_id,
                    scores=scores[scores[PATIENT_ID] == patient],
                    recs=recommendations,
                    presc=prescription_df,
                    metrics=all_metrics_df,
                    subdir=f"patient_{patient}",   # creates <base>/<run_id>/patient_<id>/
                    format="csv",
                    preview=False
                )
                result['debug'] = artifacts

            return result
        
        except Exception as e:
            logger.error(
                "Failed to process patient %s: %s (%s)", patient, e, type(e).__name__
            )
            return {
                "patient_id": patient,
                "num_recommendations": 0,
                "status": "failure",
                "error": f"{type(e).__name__}: {e}",
            }

    def _transform_recommendations(self, recommendations: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the recommendations DataFrame into prescription DataFramee.
        Only for recommended protocols.
        """
        prescription_df = recommendations.explode(DAYS).rename(
            columns={DAYS: "WEEKDAY"}
        )
        return prescription_df

    def _transform_metrics(self, scores: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the full scored protocols DataFrame into a metrics DataFrame for all protocols.
        """
        metrics_df = pd.melt(
            scores,
            id_vars=BY_PP,
            value_vars=[
                PPF,
                DELTA_DM,
                RECENT_ADHERENCE,
                SCORE,
                USAGE,
                USAGE_WEEK,
                SESSION_INDEX,
            ],
            var_name="KEY",
            value_name="VALUE",
        )
        return metrics_df

    def _save_prescriptions(
        self,
        prescription_df: pd.DataFrame,
        unique_id: uuid.UUID,
        datetime_start: datetime.datetime,
    ) -> None:
        """
        Persist prescription data.
        """
        for _, row in prescription_df.iterrows():
            weeks = row[WEEKS_SINCE_START]
            weeks = 0 if pd.isna(weeks) else float(weeks)
            start = (datetime_start + timedelta(weeks=weeks)).date()
            self.loader.interface.add_prescription_staging_entry(
                PrescriptionStagingRow.from_row(
                    row, recommendation_id=unique_id, start=start
                )
            )

    def _save_metrics(
        self,
        metrics_df: pd.DataFrame,
        unique_id: uuid.UUID,
        datetime_now: datetime.datetime,
    ) -> None:
        """
        Persist metrics data.
        """
        for _, row in metrics_df.iterrows():
            self.loader.interface.add_recsys_metric_entry(
                RecsysMetricsRow.from_row(
                    row, recommendation_id=unique_id, metric_date=datetime_now
                )
            )

    def compute_patient_fit(self, patient_id: List[int]) -> dict:
        """
        Compute and persist the Patient-Protocol Fit (PPF) matrix for a single patient.
        Delegates to PPFService.
        """
        ppf_contrib = self.ppf_service.compute_patient_fit(patient_id)
        file_path = self.ppf_service.persist_ppf(ppf_contrib)
        return {
            "message": f"Computation and persistence successful for patient {patient_id}",
            "patient_id": patient_id,
            "subscales_used": list(ppf_contrib.attrs.get("SUBSCALES", [])),
            "saved_to": file_path,
        }

    def compute_protocol_similarity(self) -> dict:
        """
        Compute and persist the protocol similarity matrix using the ProtocolSimilarityService.
        Returns a dict with the file path and a message.
        """
        file_path = (
            self.protocol_similarity_service.compute_and_persist_protocol_similarity()
        )
        return {
            "message": "Protocol similarity computation and persistence successful.",
            "saved_to": file_path,
        }
