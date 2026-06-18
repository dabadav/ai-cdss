# %%
import logging
from typing import Callable, List, Tuple
import pandas as pd

from ai_cdss.loaders import DataLoader
from ai_cdss.models import DataUnitSet
from ai_cdss.services.whitelist_service import ProtocolWhitelistService
from ai_cdss.constants import PROTOCOL_WHITELIST_YAML, PROTOCOL_ID, PROTOCOL_A, PROTOCOL_B, PATIENT_ID

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
        self.protocol_pool = ProtocolWhitelistService().load_whitelist()

    def prepare(self, patient_list: List[int]) -> Tuple[List[int], DataUnitSet, object]:
        """
        Prepare and return all data required for recommendations for a study.
        Handles missing PPF computation as needed.

        Patients without PPF are excluded (with a warning) rather than aborting
        the whole cohort: one missing PPF must not starve recommendations for
        every other patient in the run. Only raises if *no* patient has PPF.

        Args:
            patient_list (List[int]): List of patient identifiers.

        Returns:
            Tuple containing:
                - valid_patients (List[int]): patients that had PPF and will be processed
                - rgs_data (DataUnitSet): Contains session and ppf DataUnits
                - protocol_similarity (object)
        """
        ppf = self.loader.load_ppf_data(patient_list)
        missing = set(ppf.metadata.get("missing_patients", []))
        valid_patients = [p for p in patient_list if p not in missing]
        if missing:
            logger.warning(
                "Skipping %d patient(s) with missing PPF: %s. Proceeding for %d patient(s): %s",
                len(missing), sorted(missing), len(valid_patients), valid_patients,
            )
        if not valid_patients:
            raise RuntimeError(
                f"PPF data missing for all requested patients: {sorted(missing)}. "
                "Please compute PPF before proceeding."
            )
        # Drop placeholder PPF rows for the excluded patients before scoring.
        ppf.data = ppf.data[ppf.data[PATIENT_ID].isin(valid_patients)]
        session = self.loader.load_session_data(valid_patients)
        patient_data = self.loader.load_patient_data(valid_patients)
        protocol_similarity = self.loader.load_protocol_similarity()

        logger.info("Loaded data for patients: %s", valid_patients)
        logger.info("Session data shape: %s", session.data.shape)
        logger.info("PPF data shape: %s", ppf.data.shape)

        # --- filter by whitelist here ---
        if self.protocol_pool:
            allowed = set(self.protocol_pool)

            # Filter session & ppf DataUnits by protocol_id
            if PROTOCOL_ID in session.data.columns:
                session.data = session.data[session.data[PROTOCOL_ID].isin(allowed)]

            if PROTOCOL_ID in ppf.data.columns:
                ppf.data = ppf.data[ppf.data[PROTOCOL_ID].isin(allowed)]

            # Filter protocol similarity DataFrame on both sides of the pair.
            # protocol_similarity is long-format: PROTOCOL_A, PROTOCOL_B, SIMILARITY
            # with a default RangeIndex, so filter by the columns, not the index.
            protocol_similarity = protocol_similarity[
                protocol_similarity[PROTOCOL_A].isin(allowed)
            ]
            protocol_similarity = protocol_similarity[
                protocol_similarity[PROTOCOL_B].isin(allowed)
            ]

        rgs_data = DataUnitSet([session, patient_data, ppf])
        return valid_patients, rgs_data, protocol_similarity
    