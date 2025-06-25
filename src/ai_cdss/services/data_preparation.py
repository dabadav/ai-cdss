import logging
from typing import Callable, List, Tuple

from ai_cdss.loaders import DataLoader
from ai_cdss.models import DataUnitSet

logger = logging.getLogger(__name__)


class RecommendationDataService:
    """
    Service to prepare all necessary data for the recommendation pipeline.
    Handles patient validation, PPF loading and computation, session and protocol similarity loading.
    """

    def __init__(self, loader: DataLoader):
        """
        Args:
            loader (DataLoader): DataLoader instance for data access.
            compute_patient_fit (Callable): Function to compute PPF for missing patients.
        """
        self.loader = loader

    def prepare(self, study_ids: List[int]) -> Tuple[List[int], DataUnitSet, object]:
        """
        Prepare and return all data required for recommendations for a study.
        Handles missing PPF computation as needed.

        Args:
            study_ids (List[int]): List of study cohort identifiers.

        Returns:
            Tuple containing:
                - patient_list (List[int])
                - rgs_data (DataUnitSet): Contains session and ppf DataUnits
                - protocol_similarity (object)
        """
        patient_list = self.loader.fetch_and_validate_patients(study_ids=study_ids)
        ppf = self.loader.load_ppf_data(patient_list)
        missing = ppf.metadata.get("missing_patients", [])
        if missing:
            raise RuntimeError(
                f"PPF data missing for patients: {missing}. Please compute PPF before proceeding."
            )
        session = self.loader.load_session_data(patient_list)
        protocol_similarity = self.loader.load_protocol_similarity()
        rgs_data = DataUnitSet([session, ppf])
        return patient_list, rgs_data, protocol_similarity
