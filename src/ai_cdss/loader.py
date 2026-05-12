"""Data loaders — one file for all I/O.

Three loader implementations share a common abstract interface:

    DataLoaderBase    — abstract: defines the contract every loader
                        must satisfy (session / timeseries / ppf /
                        similarity / subscales / attributes / patient
                        validation).
    DataLoader        — production: pulls from the RGS MySQL via
                        `rgs_interface.DatabaseInterface`, augmented
                        with local Parquet/CSV reads for PPF +
                        similarity (which live in `~/.ai_cdss/`).
    DataLoaderLocal   — file-backed: reads everything from CSVs.
                        Useful for tests and offline replays.
    DataLoaderMock    — synthetic: generates fake data via
                        `evaluation.synthetic`. Used in unit tests.

Behavior preserved exactly from v0.3.1 (`loaders/base.py`,
`loaders/db_loader.py`, `loaders/local_loader.py`,
`loaders/mock_loader.py`, `loaders/utils.py`). The file is sectioned:

    SECTION 1  File-IO helpers (CSV / Parquet readers, JSON-encoded
               subscale decoding). Pure functions used by all loaders.
    SECTION 2  DataLoaderBase (abstract)
    SECTION 3  DataLoader (DB-backed)
    SECTION 4  DataLoaderLocal (CSV-backed)
    SECTION 5  DataLoaderMock (synthetic)
"""
from __future__ import annotations

import json
import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import pandas as pd
from pandera.errors import SchemaError
from rgs_interface.data.interface import DatabaseInterface

from ai_cdss.constants import (
    CLINICAL_SCORES,
    CLINICAL_SCORES_CSV,
    CONTRIB,
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    PATIENT_ID,
    PPF,
    PPF_PARQUET_FILEPATH,
    PROTOCOL_ATTRIBUTES_CSV,
    PROTOCOL_ID,
    PROTOCOL_SIMILARITY_CSV,
)
from ai_cdss.models import (
    DataUnit,
    DataUnitName,
    Granularity,
    PPFSchema,
    SessionSchema,
    TimeseriesSchema,
)

logger = logging.getLogger(__name__)


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1 — File-IO helpers                                         ║
# ║                                                                      ║
# ║  Pure functions for reading CSVs and Parquet from the default data   ║
# ║  directory. Used by all three loader implementations below.          ║
# ╚═════════════════════════════════════════════════════════════════════╝

def _decode_subscales(
    row: pd.Series,
    subscales_column: str = CLINICAL_SCORES,
    id_column: str = PATIENT_ID,
) -> pd.Series:
    """Decode the latest clinical-subscale evaluation from a JSON-encoded
    column into a flat Series.

    The DB stores patient subscales as JSON arrays of evaluations. We
    take the most recent entry (`[-1]`), keep only nested-dict entries
    (subscale groups, dropping metadata like `evaluation_date`), and
    flatten via `pd.json_normalize`.
    """
    data = json.loads(row[subscales_column])[-1]
    subscales = {k: v for k, v in data.items() if isinstance(v, dict)}
    flat = pd.json_normalize(subscales).iloc[0]
    flat[id_column] = row[id_column]
    return flat


def _safe_load_csv(
    file_path: Optional[Union[str, Path]] = None,
    default_filename: Optional[str] = None,
) -> pd.DataFrame:
    """Load a CSV from `file_path` (if given) or from the default data
    directory under `default_filename`. Copies the file to the default
    directory if it came from elsewhere — convenient for caching.
    """
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


def _load_patient_subscales(file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load patient clinical subscale scores. Defaults to the standard
    location under `DEFAULT_DATA_DIR/CLINICAL_SCORES_CSV`."""
    return _safe_load_csv(file_path, CLINICAL_SCORES_CSV)


def _load_protocol_attributes(file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load protocol attributes from disk; fall back to the embedded
    package data if the disk file is missing.

    On embedded-data fallback, the CSV is written to `DEFAULT_DATA_DIR`
    so subsequent calls find it on disk.
    """
    import importlib.resources

    from ai_cdss import data

    target = Path(file_path) if file_path is not None else DEFAULT_DATA_DIR / PROTOCOL_ATTRIBUTES_CSV
    if target.exists():
        return _safe_load_csv(target, PROTOCOL_ATTRIBUTES_CSV)

    try:
        embedded = importlib.resources.files(data).joinpath(PROTOCOL_ATTRIBUTES_CSV)
        df = _safe_load_csv(embedded)
        save_path = DEFAULT_DATA_DIR / PROTOCOL_ATTRIBUTES_CSV
        DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=True)
        logger.info("Protocol attributes loaded from embedded package and saved to %s", save_path)
        return df
    except Exception as e:
        raise FileNotFoundError(
            f"Protocol attributes file not found at {target} or in embedded package data: {e}"
        ) from e


def _load_protocol_similarity(file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load the protocol similarity CSV from `DEFAULT_OUTPUT_DIR`."""
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
    """Load patient-protocol fit (PPF) data from Parquet for the given
    patients. Adds placeholder rows (PPF=None, CONTRIB=None) for
    patients missing from the Parquet so downstream merges don't drop
    them silently.
    """
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
# ║  SECTION 2 — DataLoaderBase (abstract interface)                     ║
# ║                                                                      ║
# ║  Every loader must implement these methods. The pipeline             ║
# ║  (RecommendationDataService) holds a DataLoaderBase reference and    ║
# ║  doesn't care which subclass is plugged in.                          ║
# ╚═════════════════════════════════════════════════════════════════════╝

class DataLoaderBase(ABC):
    @abstractmethod
    def load_session_data(self, patient_list: List[int]) -> Union[pd.DataFrame, DataUnit]: ...

    @abstractmethod
    def load_timeseries_data(self, patient_list: List[int]) -> Union[pd.DataFrame, DataUnit]: ...

    @abstractmethod
    def load_ppf_data(self, patient_list: List[int]) -> Union[pd.DataFrame, DataUnit]: ...

    @abstractmethod
    def load_protocol_similarity(self) -> Union[pd.DataFrame, DataUnit]: ...

    @abstractmethod
    def load_patient_subscales(self, patient_list: List[int]) -> Union[pd.DataFrame, DataUnit]: ...

    @abstractmethod
    def load_protocol_attributes(self, file_path: Optional[str] = None) -> pd.DataFrame: ...

    @abstractmethod
    def fetch_and_validate_patients(self, study_ids: Optional[List[int]] = None) -> List[int]: ...


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3 — DataLoader (production, DB-backed)                      ║
# ║                                                                      ║
# ║  Pulls from RGS MySQL via DatabaseInterface for sessions, patient    ║
# ║  metadata, and timeseries. PPF + similarity come from local Parquet  ║
# ║  / CSV in ~/.ai_cdss/output/ since they're precomputed offline.      ║
# ╚═════════════════════════════════════════════════════════════════════╝

class DataLoader(DataLoaderBase):
    """RGS-MySQL-backed loader. PPF + similarity from local FS."""

    def __init__(self, rgs_mode: str = "plus") -> None:
        self.interface: DatabaseInterface = DatabaseInterface()
        self.rgs_mode = rgs_mode

    def load_patient_data(self, patient_list: List[int]) -> Union[pd.DataFrame, DataUnit]:
        return self._load_data(
            fetch_fn=self.interface.fetch_clinical_data,
            patient_list=patient_list,
            name=DataUnitName.PATIENT,
            granularity=Granularity.PATIENT_ID,
            wrap_in_dataunit=True,
        )

    def load_session_data(self, patient_list: List[int]) -> Union[pd.DataFrame, DataUnit]:
        return self._load_data(
            fetch_fn=lambda p: self.interface.fetch_rgs_data(p, rgs_mode=self.rgs_mode),
            patient_list=patient_list,
            name=DataUnitName.SESSIONS,
            granularity=Granularity.BY_PPS,
            schema_cls=SessionSchema,
            wrap_in_dataunit=True,
        )

    def load_timeseries_data(self, patient_list: List[int]) -> Union[pd.DataFrame, DataUnit]:
        return self._load_data(
            fetch_fn=lambda p: self.interface.fetch_dm_data(p, rgs_mode=self.rgs_mode),
            patient_list=patient_list,
            schema_cls=TimeseriesSchema,
        )

    def load_ppf_data(self, patient_list: List[int]) -> Union[pd.DataFrame, DataUnit]:
        return self._load_data(
            fetch_fn=lambda p: _load_ppf_data(p),
            patient_list=patient_list,
            name=DataUnitName.PPF,
            granularity=Granularity.BY_PP,
            schema_cls=PPFSchema,
            wrap_in_dataunit=True,
        )

    def load_patient_subscales(self, patient_list: List[int]) -> Union[pd.DataFrame, DataUnit]:
        """Patient subscales come from `clinical_data.CLINICAL_SCORES`
        (JSON-encoded). Decode the latest evaluation per patient, return
        as a flat DataFrame indexed by PATIENT_ID."""
        patient_data = self._load_data(
            fetch_fn=self.interface.fetch_clinical_data,
            patient_list=patient_list,
            name=DataUnitName.PATIENT,
            granularity=Granularity.PATIENT_ID,
            wrap_in_dataunit=True,
        )
        decoded = patient_data.data.apply(_decode_subscales, axis=1)
        return decoded.set_index(PATIENT_ID)

    def load_protocol_attributes(self, file_path: Optional[str] = None) -> pd.DataFrame:
        return _load_protocol_attributes()

    def load_protocol_similarity(self) -> pd.DataFrame:
        try:
            data = _load_protocol_similarity()
            logger.debug("Protocol similarity data loaded successfully.")
            return data
        except Exception as e:
            logger.error("Failed to load protocol similarity data: %s", e)
            raise

    def _load_data(
        self,
        fetch_fn: Callable[[List[int]], Any],
        patient_list: List[int],
        name: Optional[DataUnitName] = None,
        granularity: Optional[Granularity] = None,
        schema_cls: Optional[Any] = None,
        wrap_in_dataunit: bool = False,
    ) -> Union[pd.DataFrame, DataUnit]:
        """Generic fetch + optional schema validation + optional DataUnit
        wrapping. Used by all the `load_*` methods above."""
        if wrap_in_dataunit:
            assert name is not None, "Name must not be None"
            assert granularity is not None, "Granularity must not be None"
        try:
            data = fetch_fn(patient_list)
            logger.debug("%s data loaded successfully.", name or fetch_fn.__name__)
            if wrap_in_dataunit:
                metadata = dict(data.attrs) if hasattr(data, "attrs") else {}
                return DataUnit(name, data, granularity, metadata, schema_cls)  # type: ignore[arg-type]
            return data
        except SchemaError as e:
            logger.error("Data validation failed: %s", e)
            if schema_cls:
                empty_df = pd.DataFrame(columns=schema_cls.to_schema().columns.keys())
                if wrap_in_dataunit:
                    return DataUnit(name, empty_df, granularity, {}, schema_cls)  # type: ignore[arg-type]
                return empty_df
            raise
        except Exception as e:
            logger.error("Failed to load %s: %s", name or fetch_fn.__name__, e)
            raise RuntimeError(f"Failed to load {name or fetch_fn.__name__}: {e}") from e

    def fetch_and_validate_patients(self, study_ids: List[int]) -> List[int]:
        """Patient IDs for one or more study cohorts.

        Returns `[]` (with a warning) when no patients are found —
        callers expect this empty-list contract.
        """
        patient_data = self.interface.fetch_patients_by_study(study_ids=study_ids)
        if patient_data is None or patient_data.empty:
            logger.warning("No patients found for study IDs %s", study_ids)
            return []
        return patient_data[PATIENT_ID].tolist()


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4 — DataLoaderLocal (CSV-backed)                            ║
# ║                                                                      ║
# ║  Reads every input from a CSV file path supplied at construction.    ║
# ║  Used by tests and offline replay scenarios where the DB is          ║
# ║  unavailable. Timeseries loading isn't implemented — local fixtures  ║
# ║  don't usually carry per-second DM data.                             ║
# ╚═════════════════════════════════════════════════════════════════════╝

class DataLoaderLocal(DataLoaderBase):
    """CSV-backed loader. All paths supplied at construction."""

    def __init__(
        self,
        session_file: str,
        ppf_file: str,
        protocol_similarity_file: str,
        patient_subscales_file: str,
        protocol_attributes_file: str,
    ) -> None:
        self.session_file = session_file
        self.ppf_file = ppf_file
        self.protocol_similarity_file = protocol_similarity_file
        self.patient_subscales_file = patient_subscales_file
        self.protocol_attributes_file = protocol_attributes_file

    def load_session_data(self, patient_list: List[int]) -> DataUnit:
        df = pd.read_csv(self.session_file)
        if patient_list:
            df = df[df["PATIENT_ID"].isin(patient_list)]
        return DataUnit(
            name=DataUnitName.SESSIONS, data=df,
            level=Granularity.BY_PPS, schema=SessionSchema,
        )

    def load_timeseries_data(self, patient_list: List[int]):
        raise NotImplementedError(
            "Timeseries data loading is not implemented for local loader."
        )

    def load_ppf_data(self, patient_list: List[int]) -> DataUnit:
        if self.ppf_file:
            df = pd.read_csv(self.ppf_file)
            if patient_list:
                df = df[df["PATIENT_ID"].isin(patient_list)]
        else:
            df = _load_ppf_data(patient_list)
        return DataUnit(
            name=DataUnitName.PPF, data=df,
            level=Granularity.BY_PP, schema=PPFSchema,
        )

    def load_protocol_similarity(self) -> pd.DataFrame:
        return pd.read_csv(self.protocol_similarity_file)

    def load_patient_subscales(self, patient_list: Optional[List[int]] = None) -> pd.DataFrame:
        df = pd.read_csv(self.patient_subscales_file)
        if "STUDY_ID" in df.columns:
            df = df.drop(columns=["STUDY_ID"])
        if patient_list:
            df = df[df["PATIENT_ID"].isin(patient_list)]
        return df.set_index("PATIENT_ID")

    def load_protocol_attributes(self, file_path: Optional[str] = None) -> pd.DataFrame:
        return _load_protocol_attributes(file_path=self.protocol_attributes_file)

    def fetch_and_validate_patients(self, study_ids: Optional[List[int]] = None) -> List[int]:
        df = pd.read_csv(self.patient_subscales_file)
        if study_ids is not None:
            df = df[df["STUDY_ID"].isin(study_ids)]
        patient_ids = df["PATIENT_ID"].unique().tolist()
        if not patient_ids:
            raise ValueError("No patients found in the local patient subscales file.")
        return patient_ids


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5 — DataLoaderMock (synthetic)                              ║
# ║                                                                      ║
# ║  Generates fake data via `evaluation.synthetic`. Used in unit tests  ║
# ║  to exercise the pipeline without DB or fixture files.               ║
# ╚═════════════════════════════════════════════════════════════════════╝

class DataLoaderMock(DataLoaderBase):
    """Synthetic-data loader. Constructs deterministic fake data."""

    def __init__(
        self, num_patients: int = 5, num_protocols: int = 3, num_sessions: int = 10,
    ) -> None:
        from ai_cdss.evaluation.synthetic import generate_synthetic_ids
        self.ids = generate_synthetic_ids(
            num_patients=num_patients,
            num_protocols=num_protocols,
            num_sessions=num_sessions,
        )
        self.num_protocols = num_protocols

    def load_timeseries_data(self, patient_list: List[int] = None) -> DataUnit:
        from ai_cdss.evaluation.synthetic import generate_synthetic_timeseries_data
        return generate_synthetic_timeseries_data(shared_ids=self.ids)

    def load_session_data(self, patient_list: List[int] = None) -> DataUnit:
        from ai_cdss.evaluation.synthetic import generate_synthetic_session_data
        df = generate_synthetic_session_data(shared_ids=self.ids)
        return DataUnit(name=DataUnitName.SESSIONS, data=df,
                        level=Granularity.BY_PPS, schema=None)

    def load_ppf_data(self, patient_list: List[int] = None) -> DataUnit:
        from ai_cdss.evaluation.synthetic import generate_synthetic_ppf_data
        df = generate_synthetic_ppf_data(shared_ids=self.ids)
        return DataUnit(name=DataUnitName.PPF, data=df,
                        level=Granularity.BY_PP, schema=None)

    def load_patient_data(self, patient_list: List[int] = None) -> DataUnit:
        from ai_cdss.evaluation.synthetic import generate_synthetic_patient_data
        df = generate_synthetic_patient_data(shared_ids=self.ids)
        return DataUnit(name=DataUnitName.PATIENT, data=df,
                        level=Granularity.PATIENT_ID, schema=None)

    def load_protocol_similarity(self) -> pd.DataFrame:
        from ai_cdss.evaluation.synthetic import generate_synthetic_protocol_similarity
        return generate_synthetic_protocol_similarity(num_protocols=self.num_protocols)

    def load_protocol_init(self) -> pd.DataFrame:
        from ai_cdss.evaluation.synthetic import generate_synthetic_protocol_metric
        return generate_synthetic_protocol_metric(num_protocols=self.num_protocols)

    def load_patient_subscales(self, patient_list: List[int] = None):
        return super().load_patient_subscales(patient_list)  # type: ignore[arg-type]

    def load_protocol_attributes(self, file_path: Optional[str] = None) -> pd.DataFrame:
        return super().load_protocol_attributes(file_path)

    def fetch_and_validate_patients(self, *args: Any, **kwargs: Any) -> List[int]:
        """Sorted unique patient IDs from the synthetic shared-ids set."""
        return sorted({pid for pid, _, _ in self.ids})
