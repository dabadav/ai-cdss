import datetime
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd
from ai_cdss.cdss import CDSS
from ai_cdss.constants import BY_PP, DAYS, DELTA_DM, PPF, RECENT_ADHERENCE
from ai_cdss.loaders import DataLoader
from ai_cdss.processing import DataProcessor
from ai_cdss.services.data_preparation import RecommendationDataService
from ai_cdss.services.ppf_service import PPFService
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
    ):
        self.loader = loader
        self.processor = processor
        self.ppf_service = ppf_service or PPFService(loader)
        self.data_service = data_service or RecommendationDataService(loader)

    def recommend_for_study(
        self,
        study_id: List[int],
        n: int,
        days: int,
        protocols_per_day: int,
        scoring_date: Optional[pd.Timestamp] = None,
    ) -> Dict[str, Any]:
        """
        Generate recommendations for all patients in a study.
        Returns a detailed result dictionary.
        """
        logger.info("Starting recommendation generation for study: %s", study_id)
        start_time = time.time()
        try:
            patient_list, rgs_data, protocol_similarity = self.data_service.prepare(
                study_id
            )
            scores = self.processor.process_data(
                rgs_data, scoring_date or pd.Timestamp.today()
            )
            cdss = CDSS(
                scoring=scores, n=n, days=days, protocols_per_day=protocols_per_day
            )
            unique_id = uuid.uuid4()
            datetime_now = datetime.datetime.now()

            total_recommendations = 0
            patient_results = []

            for patient in patient_list:
                recommendations = cdss.recommend(patient, protocol_similarity)
                total_recommendations += len(recommendations)
                prescription_df, metrics_df = self._transform_recommendation_dataframes(
                    recommendations
                )
                self._save_prescriptions(prescription_df, unique_id, datetime_now)
                self._save_metrics(metrics_df, unique_id, datetime_now)
                patient_results.append(
                    {"patient_id": patient, "num_recommendations": len(recommendations)}
                )

            elapsed = time.time() - start_time
            logger.info("Successfully generated recommendations for study %s", study_id)
            return {
                "status": "success",
                "study_id": study_id,
                "run_id": str(unique_id),
                "patients_processed": len(patient_list),
                "total_recommendations": total_recommendations,
                "per_patient": patient_results,
                "start_time": datetime_now.isoformat(),
                "elapsed_seconds": elapsed,
                "message": f"Recommendations generated for study {study_id}",
            }
        except Exception as e:
            logger.error(
                "Failed to generate recommendations for study %s: %s", study_id, e
            )
            return {
                "status": "failure",
                "study_id": study_id,
                "error": str(e),
                "message": f"Failed to generate recommendations for study {study_id}",
            }

    def _transform_recommendation_dataframes(
        self, recommendations: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transform the recommendations DataFrame into prescription and metrics DataFrames.
        """
        prescription_df = recommendations.explode(DAYS).rename(
            columns={DAYS: "WEEKDAY"}
        )
        metrics_df = pd.melt(
            recommendations,
            id_vars=BY_PP,
            value_vars=[DELTA_DM, RECENT_ADHERENCE, PPF],
            var_name="KEY",
            value_name="VALUE",
        )
        return prescription_df, metrics_df

    def _save_prescriptions(
        self,
        prescription_df: pd.DataFrame,
        unique_id: uuid.UUID,
        datetime_now: datetime.datetime,
    ) -> None:
        """
        Persist prescription data.
        """
        for _, row in prescription_df.iterrows():
            self.loader.interface.add_prescription_staging_entry(
                PrescriptionStagingRow.from_row(
                    row, recommendation_id=unique_id, start=datetime_now
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
