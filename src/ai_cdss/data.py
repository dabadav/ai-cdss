"""Data layer — Repository pattern.

Single entry point for everything the recommendation engine needs to
read. Mirrors the `EngineState` Protocol pattern from `engine.py`
one layer up:

    CohortRepository  Protocol      (abstract source of cohorts)
        ↑ ↑
        │ └── SyntheticCohortRepository   (future — see SYNTHETIC_DATA_PLAN.md)
        └──── RGSCohortRepository       (production)
                ↓
              Cohort  dataclass            (typed bundle of frames)
                ↓
              DataPipeline.process(cohort) (downstream)

A `Cohort` is the complete set of data the pipeline needs for one
recommendation call. The `CohortRepository.find(patient_ids)` call is
the one-shot fetch. The pipeline + engine downstream don't care which
repository implementation served the call.

This file is sectioned:

    SECTION 1  File-IO primitives — read_yaml/csv/parquet helpers,
               subscale decoder, whitelist loader. Pure functions.
    SECTION 2  Cohort dataclass — the typed bundle the engine consumes.
    SECTION 3  CohortRepository Protocol — abstract source contract.
    SECTION 4  RGSCohortRepository — production implementation
               (DB via rgs_interface + local Parquet/CSV).
    SECTION 5  Clinical mappers — ClinicalSubscales +
               ProtocolToClinicalMapper. Used by the loader's
               specialized accessors for PPF/similarity computation.
    SECTION 6  Write side — PrescriptionStore Protocol +
               RGSPrescriptionStore. Symmetric to CohortRepository:
               the engine reads a Cohort, the store writes the
               recommendation output back + answers idempotency.
"""
from __future__ import annotations

import importlib.resources
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, List, Optional, Protocol, runtime_checkable

import numpy as np
import pandas as pd
import yaml
from rgs_interface.data.interface import DatabaseInterface
from rgs_interface.data.schemas import PrescriptionStagingRow, RecsysMetricsRow

from ai_cdss import config
from ai_cdss.constants import (
    CLINICAL_SCORES,
    CLINICAL_SCORES_CSV,
    CONTRIB,
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    MAPPING_YAML,
    PATIENT_ID,
    PPF,
    PPF_PARQUET_FILEPATH,
    PROTOCOL_A,
    PROTOCOL_ATTRIBUTES_CSV,
    PROTOCOL_B,
    PROTOCOL_ID,
    PROTOCOL_SIMILARITY_CSV,
    PROTOCOL_WHITELIST_YAML,
    SCALES_YAML,
)
from ai_cdss.utils import MultiKeyDict

logger = logging.getLogger(__name__)


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1 — File-IO primitives                                      ║
# ║                                                                      ║
# ║  Pure functions for reading YAML, CSV, Parquet from the default      ║
# ║  data directory. Used by the loader + the compute module.            ║
# ╚═════════════════════════════════════════════════════════════════════╝

def read_yaml(path: str | Path) -> dict:
    """Load a YAML file into a plain dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def read_csv(
    file_path: Optional[Path | str] = None,
    default_filename: Optional[str] = None,
) -> pd.DataFrame:
    """Read a CSV from `file_path` (if given) or from `DEFAULT_DATA_DIR /
    default_filename`. Copies the file to the default directory if it
    came from elsewhere — convenient for caching."""
    if file_path is not None:
        file_path = Path(file_path)
    else:
        if default_filename is None:
            raise ValueError("Either file_path or default_filename must be provided.")
        file_path = DEFAULT_DATA_DIR / default_filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}. Ensure the correct path is specified."
        )

    try:
        df = pd.read_csv(file_path, index_col=0)
        default_file_path = DEFAULT_DATA_DIR / file_path.name
        if file_path.parent != DEFAULT_DATA_DIR:
            DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
            if default_file_path.exists():
                logger.warning(
                    "Overwriting existing file in default directory: %s", default_file_path,
                )
            shutil.copy(file_path, default_file_path)
            logger.info("File copied to default directory: %s", default_file_path)
        return df
    except Exception as e:
        raise ValueError(f"Error reading {file_path}: {e}") from e


def decode_subscales(
    row: pd.Series,
    subscales_column: str = CLINICAL_SCORES,
    id_column: str = PATIENT_ID,
) -> pd.Series:
    """Decode the latest clinical-subscale evaluation from the JSON-
    encoded `CLINICAL_SCORES` column into a flat Series.

    The DB stores patient subscales as a JSON array of evaluations.
    Take the most recent entry (`[-1]`), keep only nested-dict entries
    (subscale groups, dropping metadata like `evaluation_date`), flatten
    via `pd.json_normalize`.
    """
    data = json.loads(row[subscales_column])[-1]
    subscales = {k: v for k, v in data.items() if isinstance(v, dict)}
    flat = pd.json_normalize(subscales).iloc[0]
    flat[id_column] = row[id_column]
    return flat


def load_whitelist(path: Optional[str | Path] = None) -> List[int]:
    """Load the AISN-trial-approved protocol set from a YAML config.

    Replaces the v0.3.1 `ProtocolWhitelistService` class — it was 14
    lines of class for one YAML read.
    """
    if path is None:
        path = importlib.resources.files(config) / Path(PROTOCOL_WHITELIST_YAML)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Whitelist YAML not found at {path}")
    return read_yaml(path)["recommendations"]["allowed_protocols"]


def _load_protocol_attributes(file_path: Optional[Path | str] = None) -> pd.DataFrame:
    """Read protocol attributes from disk; fall back to the embedded
    package data if the disk file is missing. On embedded-fallback the
    CSV is also written to `DEFAULT_DATA_DIR` so future calls find it
    on disk.
    """
    from ai_cdss import resources as resources_pkg

    target = Path(file_path) if file_path is not None else DEFAULT_DATA_DIR / PROTOCOL_ATTRIBUTES_CSV
    if target.exists():
        return read_csv(target, PROTOCOL_ATTRIBUTES_CSV)

    try:
        embedded = importlib.resources.files(resources_pkg).joinpath(PROTOCOL_ATTRIBUTES_CSV)
        df = read_csv(embedded)
        save_path = DEFAULT_DATA_DIR / PROTOCOL_ATTRIBUTES_CSV
        DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=True)
        logger.info("Protocol attributes loaded from embedded package and saved to %s", save_path)
        return df
    except Exception as e:
        raise FileNotFoundError(
            f"Protocol attributes file not found at {target} or in embedded package data: {e}"
        ) from e


def _load_protocol_similarity(file_path: Optional[Path | str] = None) -> pd.DataFrame:
    """Read the protocol similarity CSV from `DEFAULT_OUTPUT_DIR`."""
    target = Path(file_path) if file_path is not None else DEFAULT_OUTPUT_DIR / PROTOCOL_SIMILARITY_CSV
    if not target.exists():
        raise FileNotFoundError(
            "No protocol similarity file found in ~/.ai_cdss/output. "
            "Expected protocol_similarity.csv."
        )
    similarity = pd.read_csv(target)
    logger.debug("Protocol similarity data loaded successfully.")
    return similarity


def _load_ppf_data(patient_list: List[int]) -> pd.DataFrame:
    """Read patient-protocol fit (PPF) from Parquet. Adds placeholder
    rows for patients missing from the file so downstream merges don't
    drop them silently — placeholder presence is tagged on
    `.attrs['missing_patients']`."""
    if not patient_list:
        logger.error("PPF load called with empty patient_list")
        raise ValueError("No patients provided. Call this function with at least one patient_id.")
    if not PPF_PARQUET_FILEPATH.exists():
        msg = (
            f"No PPF file found at '{PPF_PARQUET_FILEPATH}'. "
            "Generate the PPF parquet file first, then retry (~/.ai_cdss/output)."
        )
        logger.error(msg)
        raise FileNotFoundError(msg)

    ppf_data = pd.read_parquet(path=PPF_PARQUET_FILEPATH)
    ppf_data = ppf_data[ppf_data[PATIENT_ID].isin(patient_list)]

    missing = set(patient_list) - set(ppf_data[PATIENT_ID].unique())
    if missing:
        logger.warning(
            "PPF missing for %d patients: %s. Creating placeholder rows with "
            "PPF=None, CONTRIB=None for missing patients.",
            len(missing), sorted(missing),
        )
        protocols = set(ppf_data[PROTOCOL_ID].unique())
        if not protocols:
            raise ValueError(
                f"PPF data is missing for all requested patients: {sorted(missing)} "
                "Generate the PPF data and try again."
            )
        placeholders = pd.DataFrame([
            {PATIENT_ID: pid, PROTOCOL_ID: protocol_id, PPF: None, CONTRIB: None}
            for pid in missing
            for protocol_id in protocols
        ])
        ppf_data = pd.concat([ppf_data, placeholders], ignore_index=True)
        ppf_data.attrs["missing_patients"] = list(missing)
    return ppf_data


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2 — Cohort dataclass                                        ║
# ║                                                                      ║
# ║  The complete data bundle for one recommendation call. Modeled on    ║
# ║  sklearn.Bunch — named-attribute access to a fixed set of typed     ║
# ║  frames. Replaces the v0.3.1 scatter of (RawInputs, similarity_df,   ║
# ║  attrs['missing_patients']) into a single typed object.              ║
# ╚═════════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True)
class Cohort:
    """One cohort's worth of data — the full bundle the pipeline +
    engine consume.

    Attributes
    ----------
    patient : DataFrame
        One row per patient in the cohort. Anchors clinical window.
    session : DataFrame
        One row per (patient, protocol, session). Raw observed sessions
        windowed by the loader's patient list, NOT yet date-clamped.
    ppf : DataFrame
        One row per (patient, protocol) — the PPF cohort. Drives the
        engine's env-wide alternative set.
    similarity : DataFrame
        Long-form (PROTOCOL_A, PROTOCOL_B, SIMILARITY). Already filtered
        to the whitelist on both sides.
    whitelist : list[int]
        The allowed-protocols list applied to filter patient/session/
        ppf/similarity. Here for audit/trace, not for re-filtering.
    missing_ppf : list[int]
        Patient IDs that had no PPF rows on disk (the loader injects
        placeholder rows with PPF=None for these). Empty for healthy
        cohorts. Non-empty means callers should refuse to recommend
        until PPF is computed.
    """
    patient:     pd.DataFrame
    session:     pd.DataFrame
    ppf:         pd.DataFrame
    similarity:  pd.DataFrame
    whitelist:   List[int]
    missing_ppf: List[int]


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3 — CohortRepository Protocol                               ║
# ║                                                                      ║
# ║  Abstract source of cohorts. Mirrors the EngineState Protocol        ║
# ║  (engine.py § 1) one layer up — substrate-agnostic INPUT to the      ║
# ║  recommendation pipeline.                                            ║
# ║                                                                      ║
# ║  Implementations:                                                    ║
# ║    RGSCohortRepository       production — DB + local files         ║
# ║    SyntheticCohortRepository   future — see SYNTHETIC_DATA_PLAN.md   ║
# ║    InMemoryCohortRepository    tests — pre-built Cohort, no I/O      ║
# ╚═════════════════════════════════════════════════════════════════════╝

@runtime_checkable
class CohortRepository(Protocol):
    """Read interface every cohort source must implement.

    The minimum contract is `find(patient_ids) -> Cohort`. Concrete
    repositories MAY offer specialized accessors (`patient_subscales`,
    `protocol_attributes`) for offline workflows (PPF + similarity
    computation), but those are NOT part of the protocol — they live
    only on the concrete classes.
    """
    def find(self, patient_ids: List[int]) -> Cohort: ...


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4 — RGSCohortRepository (production)                      ║
# ║                                                                      ║
# ║  Pulls patient + session from RGS MySQL via DatabaseInterface;       ║
# ║  reads precomputed PPF + similarity from ~/.ai_cdss/output/.         ║
# ║  Applies the whitelist filter to session / PPF / similarity (both    ║
# ║  sides of the pair table).                                           ║
# ╚═════════════════════════════════════════════════════════════════════╝

class RGSCohortRepository:
    """Production `CohortRepository`. Single concrete implementation
    today; satisfies the protocol contract."""

    def __init__(
        self,
        db: Optional[DatabaseInterface] = None,
        rgs_mode: str = "plus",
        whitelist: Optional[List[int]] = None,
    ) -> None:
        self.interface = db or DatabaseInterface()
        self.rgs_mode = rgs_mode
        self.whitelist = whitelist if whitelist is not None else load_whitelist()

    # ------------------------------------------------------------------
    # Public protocol contract.

    def find(self, patient_ids: List[int]) -> Cohort:
        """One-shot fetch + filter + assemble. Returns the full Cohort
        bundle ready for the pipeline."""
        ppf = self._fetch(_load_ppf_data, patient_ids, name="ppf")
        missing_ppf = list(ppf.attrs.get("missing_patients", []))

        session = self._fetch(
            lambda p: self.interface.fetch_rgs_data(p, rgs_mode=self.rgs_mode),
            patient_ids, name="sessions",
        )
        patient = self._fetch(
            self.interface.fetch_clinical_data, patient_ids, name="patient",
        )
        similarity = self._fetch(
            lambda _: _load_protocol_similarity(), patient_ids, name="similarity",
        )

        logger.info("Loaded data for patients: %s", patient_ids)
        logger.info("Session data shape: %s", session.shape)
        logger.info("PPF data shape: %s", ppf.shape)

        # Apply whitelist filter to session / ppf / similarity.
        if self.whitelist:
            allowed = set(self.whitelist)
            if PROTOCOL_ID in session.columns:
                session = session[session[PROTOCOL_ID].isin(allowed)]
            if PROTOCOL_ID in ppf.columns:
                ppf = ppf[ppf[PROTOCOL_ID].isin(allowed)]
            # similarity is long-form — filter both sides of the pair.
            similarity = similarity[similarity[PROTOCOL_A].isin(allowed)]
            similarity = similarity[similarity[PROTOCOL_B].isin(allowed)]

        return Cohort(
            patient=patient,
            session=session,
            ppf=ppf,
            similarity=similarity,
            whitelist=list(self.whitelist),
            missing_ppf=missing_ppf,
        )

    # ------------------------------------------------------------------
    # Specialized accessors for offline PPF / similarity computation.
    # NOT part of the CohortRepository protocol contract — these are
    # production-loader specifics. Synthetic repositories don't need them.

    def patient_subscales(self, patient_ids: List[int]) -> pd.DataFrame:
        """Patient subscales from `clinical_data.CLINICAL_SCORES`
        (JSON-encoded). Decode the latest evaluation per patient,
        return as a flat DataFrame indexed by PATIENT_ID."""
        patient = self._fetch(
            self.interface.fetch_clinical_data, patient_ids, name="patient",
        )
        decoded = patient.apply(decode_subscales, axis=1)
        return decoded.set_index(PATIENT_ID)

    def protocol_attributes(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Protocol attributes from local CSV (with embedded-package
        fallback)."""
        return _load_protocol_attributes(file_path=file_path)

    def fetch_and_validate_patients(self, study_ids: List[int]) -> List[int]:
        """Patient IDs for one or more study cohorts. Returns `[]` (with
        a warning) when no patients are found — callers expect this
        empty-list contract."""
        patient_data = self.interface.fetch_patients_by_study(study_ids=study_ids)
        if patient_data is None or patient_data.empty:
            logger.warning("No patients found for study IDs %s", study_ids)
            return []
        return patient_data[PATIENT_ID].tolist()

    # ------------------------------------------------------------------
    # Internal — fetch wrapper with consistent error logging.

    def _fetch(
        self,
        fetch_fn: Callable[[List[int]], Any],
        patient_list: List[int],
        *,
        name: str,
    ) -> pd.DataFrame:
        """Run a fetch and log success. Wraps any exception as
        `RuntimeError(f"Failed to load {name}: ...")` so the caller
        gets one consistent error type per source."""
        try:
            data = fetch_fn(patient_list)
            logger.debug("%s data loaded successfully.", name)
            return data
        except Exception as e:
            logger.error("Failed to load %s: %s", name, e)
            raise RuntimeError(f"Failed to load {name}: {e}") from e


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5 — Clinical mappers                                        ║
# ║                                                                      ║
# ║  Used by the PPF + similarity computation flows (see compute.py).    ║
# ║  Live here because they map raw-frame → normalized-frame — that's    ║
# ║  loading-adjacent, not computation.                                  ║
# ╚═════════════════════════════════════════════════════════════════════╝

class ClinicalSubscales:
    """Patient subscale-scores → deficit-matrix transformer.

    Reads max-subscale values from a YAML config (default: the embedded
    `config/scales.yaml`). The deficit matrix is `1 - (score / max)`
    so higher values mean larger deficits.
    """

    def __init__(self, scale_yaml_path: Optional[str] = None) -> None:
        if scale_yaml_path:
            self.scales_path = Path(scale_yaml_path)
        else:
            self.scales_path = importlib.resources.files(config) / Path(SCALES_YAML)
        if not self.scales_path.exists():
            raise FileNotFoundError(f"Scale YAML file not found at {self.scales_path}")
        self.scales_dict = MultiKeyDict.from_yaml(self.scales_path)

    def compute_deficit_matrix(self, patient_df: pd.DataFrame) -> pd.DataFrame:
        """Compute deficit matrix given patient clinical scores."""
        max_subscales = [self.scales_dict.get(scale, None) for scale in patient_df.columns]
        if None in max_subscales:
            missing_subscales = [
                scale for scale, max_val in zip(patient_df.columns, max_subscales)
                if max_val is None
            ]
            raise ValueError(f"Missing max values for subscales: {missing_subscales}")

        deficit_matrix = 1 - (
            patient_df / pd.Series(max_subscales, index=patient_df.columns)
        )
        deficit_matrix.rename(self.scales_dict._keys, axis=1, inplace=True)
        return deficit_matrix


class ProtocolToClinicalMapper:
    """Protocol attribute frame → clinical-scale frame.

    Reads the protocol→subscale mapping from YAML (default:
    `config/mapping.yaml`). Each clinical scale becomes a column whose
    value is `agg_func` (default mean) over the protocol attributes
    that map to it.
    """

    def __init__(self, mapping_yaml_path: Optional[str] = None) -> None:
        if mapping_yaml_path:
            self.mapping_path = Path(mapping_yaml_path)
        else:
            self.mapping_path = importlib.resources.files(config) / Path(MAPPING_YAML)
        if not self.mapping_path.exists():
            raise FileNotFoundError(f"Mapping YAML file not found at {self.mapping_path}")
        self.mapping = MultiKeyDict.from_yaml(self.mapping_path)

    def map_protocol_features(
        self, protocol_df: pd.DataFrame, agg_func=np.mean,
    ) -> pd.DataFrame:
        """Map protocol-level features into clinical scales."""
        df_clinical = pd.DataFrame(index=protocol_df.index)
        for clinical_scale, features in self.mapping.items():
            df_clinical[clinical_scale] = protocol_df[features].apply(agg_func, axis=1)
        df_clinical.index = protocol_df[PROTOCOL_ID]
        return df_clinical


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 6 — Write side (PrescriptionStore)                          ║
# ║                                                                      ║
# ║  Symmetric to the CohortRepository read side (§ 3). CohortRepository ║
# ║  answers "what does this cohort look like?"; PrescriptionStore       ║
# ║  answers "has this patient already been prescribed this week?" and   ║
# ║  persists the recommendation output (prescriptions + metrics).       ║
# ║                                                                      ║
# ║  Keeping it a Protocol means the orchestrator no longer reaches into ║
# ║  the concrete DB interface (or its private _fetch) — a synthetic /   ║
# ║  in-memory store can run the full orchestrator without a database.   ║
# ╚═════════════════════════════════════════════════════════════════════╝

@runtime_checkable
class PrescriptionStore(Protocol):
    """Write interface every prescription sink must implement.

    Implementations:
        RGSPrescriptionStore      production — DB via rgs_interface
        InMemoryPrescriptionStore tests / synthetic backtests (future)
    """
    def already_prescribed(self, patient_id: int, week_start: date) -> bool: ...
    def save_prescriptions(self, rows: List[PrescriptionStagingRow]) -> None: ...
    def save_metrics(self, rows: List[RecsysMetricsRow]) -> None: ...


class RGSPrescriptionStore:
    """Production `PrescriptionStore` backed by `DatabaseInterface`.

    Constructed with the same interface instance as `RGSCohortRepository`
    (see `CDSS.__init__`) so read + write share one DB connection.
    """

    def __init__(self, db: Optional[DatabaseInterface] = None) -> None:
        self.interface = db or DatabaseInterface()

    def already_prescribed(self, patient_id: int, week_start: date) -> bool:
        """True if `prescription_staging` already has any row (any STATUS)
        for `(patient_id, week_start)`. Swallows query errors as
        not-prescribed — a failed check must not block a fresh run."""
        engine = getattr(self.interface, "engine", None)
        if engine is None:
            return False
        sql = (
            "SELECT COUNT(*) AS n FROM prescription_staging "
            "WHERE PATIENT_ID = :pid AND DATE(STARTING_DATE) = :wk"
        )
        try:
            df = self.interface._fetch(
                query=sql,
                params={"pid": int(patient_id), "wk": week_start.isoformat()},
            )
            return bool(df is not None and not df.empty and int(df.iloc[0]["n"]) > 0)
        except Exception:
            logger.exception(
                "Duplication check failed for patient %s; treating as "
                "not-prescribed.", patient_id,
            )
            return False

    def save_prescriptions(self, rows: List[PrescriptionStagingRow]) -> None:
        for row in rows:
            self.interface.add_prescription_staging_entry(row)

    def save_metrics(self, rows: List[RecsysMetricsRow]) -> None:
        for row in rows:
            self.interface.add_recsys_metric_entry(row)
