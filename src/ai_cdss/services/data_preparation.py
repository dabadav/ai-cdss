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
        """
        self.loader = loader

    def prepare(self, patient_list: List[int]) -> Tuple[List[int], DataUnitSet, object]:
        """
        Prepare and return all data required for recommendations for a study.
        Handles missing PPF computation as needed.

        Args:
            patient_list (List[int]): List of patient identifiers.

        Returns:
            Tuple containing:
                - patient_list (List[int])
                - rgs_data (DataUnitSet): Contains session and ppf DataUnits
                - protocol_similarity (object)
        """
        ppf = self.loader.load_ppf_data(patient_list)
        missing = ppf.metadata.get("missing_patients", [])
        if missing:
            raise RuntimeError(
                f"PPF data missing for patients: {missing}. Please compute PPF before proceeding."
            )
        session = self.loader.load_session_data(patient_list)
        patient_data = self.loader.load_patient_data(patient_list)
        protocol_similarity = self.loader.load_protocol_similarity()
        rgs_data = DataUnitSet([session, patient_data, ppf])
        return rgs_data, protocol_similarity
