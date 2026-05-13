"""Services — derived computations layered on top of loaders.

A "service" sits between the raw loaders (see `loader.py`) and the
recommendation engine (see `recommend.py`). Services orchestrate
multi-step data preparation that the recommender shouldn't have to know
about: assembling the `rgs_data` bundle, computing PPF from clinical
subscales, computing protocol similarity from attribute embeddings,
loading the protocol whitelist from YAML.

Sections:

    SECTION 1  load_yaml — trivial YAML helper used by the whitelist
               service. Module-level so it's reusable.
    SECTION 2  ProtocolWhitelistService — loads the allowed-protocols
               list from the embedded `config/protocol_whitelist.yaml`.
    SECTION 3  RecommendationDataService — orchestrator. Pulls the
               loader's four core frames (sessions / patient /
               PPF / similarity), applies the whitelist filter, returns
               the `(rgs_data, similarity)` bundle the pipeline
               consumes.
    SECTION 4  PPFService — compute PPF from patient subscales + protocol
               attributes; persist to Parquet.
    SECTION 5  ProtocolSimilarityService — compute pairwise Gower
               similarity from protocol attributes; persist to CSV.

Behavior preserved exactly from v0.3.1 (`services/whitelist_service.py`,
`services/data_preparation.py`, `services/ppf_service.py`,
`services/protocol_similarity.py`).
"""
from __future__ import annotations

import importlib.resources
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

from ai_cdss import config
from ai_cdss.clinical import ClinicalSubscales, ProtocolToClinicalMapper
from ai_cdss.constants import (
    BY_PP,
    DEFAULT_OUTPUT_DIR,
    PPF_PARQUET_FILEPATH,
    PROTOCOL_A,
    PROTOCOL_B,
    PROTOCOL_ID,
    PROTOCOL_SIMILARITY_CSV,
    PROTOCOL_WHITELIST_YAML,
)
from ai_cdss.feature import compute_ppf, compute_protocol_similarity
from ai_cdss.loader import DataLoader
from ai_cdss.pipeline import RawInputs

logger = logging.getLogger(__name__)


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1 — YAML helper                                             ║
# ╚═════════════════════════════════════════════════════════════════════╝

def load_yaml(path: str | Path) -> dict:
    """Load a YAML file into a plain dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2 — ProtocolWhitelistService                                ║
# ║                                                                      ║
# ║  Loads the AISN-trial-approved protocol set from the embedded        ║
# ║  config YAML. The list is applied as a filter on session / ppf /     ║
# ║  similarity by RecommendationDataService below.                      ║
# ╚═════════════════════════════════════════════════════════════════════╝

class ProtocolWhitelistService:
    """Reads the allowed-protocols list from a YAML config."""

    def __init__(self, whitelist_yml_path: Optional[str] = None) -> None:
        if whitelist_yml_path:
            self.scales_path = Path(whitelist_yml_path)
        else:
            self.scales_path = importlib.resources.files(config) / Path(PROTOCOL_WHITELIST_YAML)
        if not self.scales_path.exists():
            raise FileNotFoundError(f"Whitelist YAML file not found at {self.scales_path}")

    def load_whitelist(self) -> List[int]:
        """Return the list of allowed protocol IDs."""
        return load_yaml(self.scales_path)["recommendations"]["allowed_protocols"]


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3 — RecommendationDataService                               ║
# ║                                                                      ║
# ║  The "prepare" step the pipeline runs once per (study, recommendation║
# ║  request). Loads the four frames the pipeline needs and applies the  ║
# ║  whitelist filter.                                                   ║
# ╚═════════════════════════════════════════════════════════════════════╝

class RecommendationDataService:
    """Loader-orchestration service. Returns the `(raw_inputs, similarity)`
    pair the DataPipeline consumes."""

    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader
        self.protocol_pool: List[int] = ProtocolWhitelistService().load_whitelist()

    def prepare(self, patient_list: List[int]) -> Tuple["RawInputs", pd.DataFrame]:
        """Load + filter the four input frames for `patient_list`.

        Raises if any patient is missing PPF — callers should
        precompute PPF before recommending. The whitelist filter is
        applied to session, PPF, and similarity (both sides).
        """
        ppf = self.loader.load_ppf_data(patient_list)
        missing = ppf.attrs.get("missing_patients", [])
        if missing:
            raise RuntimeError(
                f"PPF data missing for patients: {missing}. "
                "Please compute PPF before proceeding."
            )

        session = self.loader.load_session_data(patient_list)
        patient = self.loader.load_patient_data(patient_list)
        protocol_similarity = self.loader.load_protocol_similarity()

        logger.info("Loaded data for patients: %s", patient_list)
        logger.info("Session data shape: %s", session.shape)
        logger.info("PPF data shape: %s", ppf.shape)

        if self.protocol_pool:
            allowed = set(self.protocol_pool)
            if PROTOCOL_ID in session.columns:
                session = session[session[PROTOCOL_ID].isin(allowed)]
            if PROTOCOL_ID in ppf.columns:
                ppf = ppf[ppf[PROTOCOL_ID].isin(allowed)]
            # similarity is long-form (PROTOCOL_A, PROTOCOL_B, SIMILARITY)
            # — filter on both sides of each pair.
            protocol_similarity = protocol_similarity[
                protocol_similarity[PROTOCOL_A].isin(allowed)
            ]
            protocol_similarity = protocol_similarity[
                protocol_similarity[PROTOCOL_B].isin(allowed)
            ]

        return RawInputs(patient=patient, session=session, ppf=ppf), protocol_similarity


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4 — PPFService                                              ║
# ║                                                                      ║
# ║  PPF = cosine(patient_deficit_vector, protocol_attribute_vector).    ║
# ║  Computed offline (per-patient on enrollment) and persisted to       ║
# ║  Parquet. The DataLoader reads the Parquet at recommendation time.   ║
# ╚═════════════════════════════════════════════════════════════════════╝

class PPFService:
    """Compute + persist PPF for one or more patients."""

    def __init__(
        self,
        loader: Any,
        mapping_yaml_path: Optional[str] = None,
        scale_yaml_path: Optional[str] = None,
    ) -> None:
        self.loader = loader
        self.mapping_yaml_path = mapping_yaml_path
        self.scale_yaml_path = scale_yaml_path

    def compute_patient_fit(self, patient_id: List[int]) -> pd.DataFrame:
        """Compute PPF for one or more patients. Returns the
        joined (PPF, CONTRIB) DataFrame; does NOT persist."""
        patient = self.loader.load_patient_subscales(patient_id)
        if patient.empty:
            raise ValueError(f"Patient data not found for ID: {patient_id}")

        protocol = self.loader.load_protocol_attributes()
        if protocol.empty:
            raise ValueError("Protocol data could not be loaded.")

        patient_def = ClinicalSubscales(
            scale_yaml_path=self.scale_yaml_path,
        ).compute_deficit_matrix(patient)
        protocol_map = ProtocolToClinicalMapper(
            mapping_yaml_path=self.mapping_yaml_path,
        ).map_protocol_features(protocol)

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
        """Persist the PPF DataFrame to Parquet.

        First call creates the file; subsequent calls upsert by `BY_PP`
        keys (existing rows for those (patient, protocol) pairs are
        replaced).
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
        """Convenience: compute + persist in one call."""
        ppf_contrib = self.compute_patient_fit(patient_id)
        file_path = self.persist_ppf(ppf_contrib)
        return {
            "message": f"Computation and persistence successful for patient {patient_id}",
            "patient_id": patient_id,
            "subscales_used": list(ppf_contrib.attrs.get("SUBSCALES", [])),
            "saved_to": file_path,
        }


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5 — ProtocolSimilarityService                               ║
# ║                                                                      ║
# ║  Gower similarity over protocol attributes (one row per protocol,    ║
# ║  columns = motor/cognitive subscale weights). Computed once when a   ║
# ║  new protocol is added; persisted to CSV.                            ║
# ╚═════════════════════════════════════════════════════════════════════╝

class ProtocolSimilarityService:
    """Compute + persist protocol pairwise similarity."""

    def __init__(self, loader: Any) -> None:
        self.loader = loader

    def compute_protocol_similarity(self) -> pd.DataFrame:
        protocol = self.loader.load_protocol_attributes()
        if protocol is None or protocol.empty:
            raise ValueError("Protocol data could not be loaded.")
        protocol_map = ProtocolToClinicalMapper().map_protocol_features(protocol)
        if protocol_map is None or protocol_map.empty:
            raise ValueError("Mapped protocol features are empty.")
        similarity_df = compute_protocol_similarity(protocol_map)
        if similarity_df is None or similarity_df.empty:
            raise ValueError("Protocol similarity computation returned no data.")
        return similarity_df

    def persist_protocol_similarity(self, similarity_df: pd.DataFrame) -> str:
        try:
            output_path = DEFAULT_OUTPUT_DIR / PROTOCOL_SIMILARITY_CSV
            output_path.parent.mkdir(parents=True, exist_ok=True)
            similarity_df.to_csv(output_path, index=False)
            logger.info("Protocol similarity persisted successfully to %s.", output_path)
            return str(output_path)
        except Exception as e:
            logger.error("Failed to save protocol similarity to CSV: %s", e)
            raise RuntimeError(f"Failed to save protocol similarity to CSV: {e}") from e

    def compute_and_persist_protocol_similarity(self) -> str:
        return self.persist_protocol_similarity(self.compute_protocol_similarity())
