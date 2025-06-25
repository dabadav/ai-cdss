import logging
from typing import Any, Dict, List

import pandas as pd
from ai_cdss.constants import BY_PP, PPF_PARQUET_FILEPATH
from ai_cdss.processing import ClinicalSubscales, ProtocolToClinicalMapper
from ai_cdss.processing.features import compute_ppf

logger = logging.getLogger(__name__)


class PPFService:
    """
    Service for computing and persisting Patient-Protocol Fit (PPF) matrices.
    """

    def __init__(self, loader):
        self.loader = loader

    def compute_patient_fit(self, patient_id: List[int]) -> pd.DataFrame:
        """
        Compute the Patient-Protocol Fit (PPF) matrix for a single patient.
        Returns the PPF DataFrame (not persisted).
        """
        patient = self.loader.load_patient_subscales(patient_id)
        if patient.empty:
            raise ValueError(f"Patient data not found for ID: {patient_id}")

        protocol = self.loader.load_protocol_attributes()
        if protocol.empty:
            raise ValueError("Protocol data could not be loaded.")

        patient_def = ClinicalSubscales().compute_deficit_matrix(patient)
        protocol_map = ProtocolToClinicalMapper().map_protocol_features(protocol)

        missing_subscales = protocol_map.columns.difference(patient_def.columns)
        if not missing_subscales.empty:
            raise ValueError(
                f"Patient data is missing required subscales: {', '.join(missing_subscales)}"
            )

        patient_def = patient_def[protocol_map.columns]
        ppf, contrib = compute_ppf(patient_def, protocol_map)
        ppf_contrib = pd.merge(ppf, contrib, on=BY_PP, how="left")
        ppf_contrib.attrs = {"SUBSCALES": list(protocol_map.columns)}
        if ppf_contrib.empty:
            raise ValueError(f"No PPF data to save for patient {patient_id}")
        return ppf_contrib

    def persist_ppf(self, ppf_contrib: pd.DataFrame) -> str:
        """
        Persist the PPF DataFrame to Parquet.
        Returns the file path.
        """
        try:
            if not PPF_PARQUET_FILEPATH.exists():
                ppf_contrib.to_parquet(PPF_PARQUET_FILEPATH, index=False)
            else:
                existing = pd.read_parquet(PPF_PARQUET_FILEPATH)
                keys = ppf_contrib[BY_PP]
                merged = existing.merge(keys, on=BY_PP, how="left", indicator=True)
                filtered = existing[merged["_merge"] == "left_only"]
                updated = pd.concat([filtered, ppf_contrib], ignore_index=True)
                updated.attrs = ppf_contrib.attrs
                updated.to_parquet(PPF_PARQUET_FILEPATH)
            logger.info("PPF persisted successfully.")
            return str(PPF_PARQUET_FILEPATH.absolute())
        except Exception as e:
            logger.error("Failed to save results to Parquet: %s", e)
            raise RuntimeError(f"Failed to save results to Parquet: {e}") from e

    def compute_and_persist_patient_fit(self, patient_id: List[int]) -> Dict[str, Any]:
        """
        Convenience method to compute and persist PPF in one call.
        """
        ppf_contrib = self.compute_patient_fit(patient_id)
        file_path = self.persist_ppf(ppf_contrib)
        return {
            "message": f"Computation and persistence successful for patient {patient_id}",
            "patient_id": patient_id,
            "subscales_used": list(ppf_contrib.attrs.get("SUBSCALES", [])),
            "saved_to": file_path,
        }
