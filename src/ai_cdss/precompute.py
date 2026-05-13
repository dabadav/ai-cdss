"""Offline computations — PPF and protocol similarity.

Pure functions. No classes. No state. These run during the offline
patient-registration + protocol-addition workflows, NOT during a
recommendation call (the recommender reads precomputed PPF / similarity
from disk via `RGSCohortRepository`).

Sections:

    SECTION 1  PPF — compute + persist patient-protocol fit.
    SECTION 2  Protocol similarity — compute + persist pairwise
               Gower-distance similarity over protocol attributes.

Replaces `PPFService` and `ProtocolSimilarityService` from v0.3.1
service.py — those classes were ~80 lines of stateful wrapper around
the four functions below.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ai_cdss.constants import (
    BY_PP,
    DEFAULT_OUTPUT_DIR,
    PPF_PARQUET_FILEPATH,
    PROTOCOL_SIMILARITY_CSV,
)
from ai_cdss.data import ClinicalSubscales, ProtocolToClinicalMapper
from ai_cdss.metrics import compute_ppf, compute_protocol_similarity

logger = logging.getLogger(__name__)


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1 — PPF                                                     ║
# ║                                                                      ║
# ║  PPF = cosine(patient_deficit_vector, protocol_attribute_vector).    ║
# ║  Computed offline (per-patient on enrollment) and persisted to       ║
# ║  Parquet; the production loader reads the Parquet at recommend time. ║
# ╚═════════════════════════════════════════════════════════════════════╝

def compute_ppf_for_patients(
    patient_subscales: pd.DataFrame,
    protocol_attributes: pd.DataFrame,
    scales_yaml: Optional[Path | str] = None,
    mapping_yaml: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Compute PPF for the patients in `patient_subscales`.

    Pure: takes raw frames in, returns the joined `(PPF, CONTRIB)`
    long-form DataFrame. Does NOT persist — call `persist_ppf` for
    that. The `SUBSCALES` list used in the computation is attached
    via `result.attrs["SUBSCALES"]`.

    Raises if the patient frame is missing any subscale required by
    the protocol mapping.
    """
    if patient_subscales.empty:
        raise ValueError("patient_subscales is empty — nothing to compute.")
    if protocol_attributes.empty:
        raise ValueError("protocol_attributes is empty — nothing to compute.")

    patient_def = ClinicalSubscales(
        scale_yaml_path=str(scales_yaml) if scales_yaml else None,
    ).compute_deficit_matrix(patient_subscales)
    protocol_map = ProtocolToClinicalMapper(
        mapping_yaml_path=str(mapping_yaml) if mapping_yaml else None,
    ).map_protocol_features(protocol_attributes)

    missing = protocol_map.columns.difference(patient_def.columns)
    if not missing.empty:
        raise ValueError(
            f"Patient data is missing required subscales: {', '.join(missing)}"
        )

    patient_def = patient_def[protocol_map.columns]
    ppf, contrib = compute_ppf(patient_def, protocol_map)
    ppf_contrib = pd.merge(ppf, contrib, on=BY_PP, how="left")
    ppf_contrib.attrs = {"SUBSCALES": list(protocol_map.columns)}
    if ppf_contrib.empty:
        raise ValueError("No PPF data produced.")
    return ppf_contrib


def persist_ppf(
    df: pd.DataFrame, path: Optional[Path | str] = None,
) -> Path:
    """Persist a PPF DataFrame to Parquet.

    First call creates the file; subsequent calls upsert by `BY_PP`
    keys (existing rows for those `(patient, protocol)` pairs are
    replaced).
    """
    target = Path(path) if path is not None else PPF_PARQUET_FILEPATH
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not target.exists():
            df.to_parquet(target, index=False)
        else:
            existing = pd.read_parquet(target)
            keys = df[BY_PP]
            merged = existing.merge(keys, on=BY_PP, how="left", indicator=True)
            filtered = existing[merged["_merge"] == "left_only"]
            updated = pd.concat([filtered, df], ignore_index=True)
            updated.attrs = df.attrs
            updated.to_parquet(target)
        logger.info("PPF persisted to %s", target)
        return target.absolute()
    except Exception as e:
        logger.error("Failed to save PPF Parquet: %s", e)
        raise RuntimeError(f"Failed to save PPF Parquet: {e}") from e


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2 — Protocol similarity                                     ║
# ║                                                                      ║
# ║  Gower similarity over protocol attributes (one row per protocol,    ║
# ║  columns = motor/cognitive subscale weights). Computed once per      ║
# ║  protocol-set change; persisted to CSV.                              ║
# ╚═════════════════════════════════════════════════════════════════════╝

def compute_protocol_similarity_matrix(
    protocol_attributes: pd.DataFrame,
    mapping_yaml: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Compute pairwise protocol similarity.

    Pure: protocol_attributes → long-form `(PROTOCOL_A, PROTOCOL_B,
    SIMILARITY)` DataFrame. Does NOT persist — call `persist_similarity`.
    """
    if protocol_attributes is None or protocol_attributes.empty:
        raise ValueError("protocol_attributes is empty — nothing to compute.")
    protocol_map = ProtocolToClinicalMapper(
        mapping_yaml_path=str(mapping_yaml) if mapping_yaml else None,
    ).map_protocol_features(protocol_attributes)
    if protocol_map is None or protocol_map.empty:
        raise ValueError("Mapped protocol features are empty.")
    similarity = compute_protocol_similarity(protocol_map)
    if similarity is None or similarity.empty:
        raise ValueError("Protocol similarity computation returned no data.")
    return similarity


def persist_similarity(
    df: pd.DataFrame, path: Optional[Path | str] = None,
) -> Path:
    """Persist protocol similarity to CSV."""
    target = Path(path) if path is not None else DEFAULT_OUTPUT_DIR / PROTOCOL_SIMILARITY_CSV
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(target, index=False)
        logger.info("Protocol similarity persisted to %s", target)
        return target.absolute()
    except Exception as e:
        logger.error("Failed to save similarity CSV: %s", e)
        raise RuntimeError(f"Failed to save similarity CSV: {e}") from e
