# ai_cdss/models.py
import logging
from dataclasses import dataclass, field
from enum import Enum
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Optional, Type

import pandas as pd
import pandera as pa
from ai_cdss.constants import *
from pandera.errors import SchemaError

NullableField = partial(pa.Field, nullable=True)

# Set up logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Dataclasses


class Granularity(Enum):
    PATIENT_ID = "PATIENT_ID"
    BY_PP = "BY_PP"
    BY_PPS = "BY_PPS"
    BY_PPST = "BY_PPST"
    BY_ID = "BY_ID"

    def id_cols(self) -> List[str]:
        try:
            return {
                Granularity.BY_PP: BY_PP,
                Granularity.BY_PPS: BY_PPS,
                Granularity.BY_PPST: BY_PPST,
                Granularity.BY_ID: BY_ID,
                Granularity.PATIENT_ID: [PATIENT_ID],
            }[self]
        except KeyError:
            raise ValueError("Unsupported Granularity: %s" % self)


class DataUnitName(str, Enum):
    PATIENT = "patient"
    SESSIONS = "sessions"
    PPF = "ppf"


@dataclass
class DataUnit:
    """
    A unit of data with an associated granularity and optional metadata.
    """

    name: DataUnitName
    data: pd.DataFrame
    level: Granularity
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema: Optional[Type[pa.DataFrameModel]] = None

    @property
    def id_cols(self) -> List[str]:
        return self.level.id_cols()

    def validate(self):
        """Validate the data using the attached schema, if present."""
        if self.schema is not None:
            try:
                self.data = self.schema.validate(self.data)
            except SchemaError as e:
                logger.error(
                    "Schema validation failed for DataUnit '%s': %s", self.name, e
                )
                raise
        return self

    def __post_init__(self):
        if self.name is None:
            raise ValueError("DataUnit.name must not be None")
        if self.level is None:
            raise ValueError("DataUnit.level (granularity) must not be None")


# class DataUnitSet:
#     def __init__(self, units: List[DataUnit]):
#         self.units: Dict[str, DataUnit] = {unit.name: unit for unit in units}

#     def get(self, name: DataUnitName) -> DataUnit:
#         return self.units[name.value]


class DataUnitSet:
    def __init__(self, units: List[DataUnit]):
        self.units: Dict[DataUnitName, DataUnit] = {unit.name: unit for unit in units}

    def get(self, name: DataUnitName) -> DataUnit:
        return self.units[name]

    def __getitem__(self, name: DataUnitName) -> DataUnit:
        return self.units[name]
    
    def __repr__(self) -> str:
        unit_names = ", ".join(self.units.keys())
        return f"<DataUnitSet units=[{unit_names}]>"

# ---------------------------------------------------------------------
# RGS Data Input


class SessionSchema(pa.DataFrameModel):
    """
    Schema for RGS session-level data, including patient profile, prescription and session details.
    """

    # Patient profile
    patient_id: int = pa.Field(alias=PATIENT_ID)

    # Identifiers
    prescription_id: int = pa.Field(alias=PRESCRIPTION_ID)
    session_id: int = NullableField(alias=SESSION_ID)
    protocol_id: int = NullableField(alias=PROTOCOL_ID)

    # Prescription
    prescription_starting_date: pa.DateTime = pa.Field(alias=PRESCRIPTION_STARTING_DATE)
    prescription_ending_date: pa.DateTime = pa.Field(alias=PRESCRIPTION_ENDING_DATE)

    # Session
    session_date: pa.DateTime = NullableField(alias=SESSION_DATE)
    weekday: int = NullableField(
        alias=WEEKDAY_INDEX,
        ge=0,
        le=6,
        description="Weekday Index (0=Monday, 6=Sunday)",
    )
    status: str = NullableField(alias=STATUS, isin=[e.value for e in SessionStatus])

    # Metrics
    real_session_duration: int = NullableField(alias=REAL_SESSION_DURATION, ge=0)
    prescribed_session_duration: int = NullableField(
        alias=PRESCRIBED_SESSION_DURATION, ge=0
    )
    session_duration: int = NullableField(alias=SESSION_DURATION, ge=0)
    adherence: float = NullableField(alias=ADHERENCE, ge=0, le=1)
    dm_value: float = NullableField(alias=DM_VALUE)


class TimeseriesSchema(pa.DataFrameModel):
    """
    Schema for timeseries session data. Includes measurements per-second of difficulty modulators (DM) and performance estimates (PE).
    """

    # Identifiers
    patient_id: int = NullableField(alias=PATIENT_ID, gt=0)
    session_id: int = NullableField(alias=SESSION_ID, gt=0)
    protocol_id: int = NullableField(alias=PROTOCOL_ID, gt=0)

    # Protocol
    game_mode: str = NullableField(alias=GAME_MODE)

    # Time
    timepoint: int = NullableField(alias=SECONDS_FROM_START)

    # Metrics
    dm_key: str = NullableField(alias=DM_KEY)
    dm_value: float = NullableField(alias=DM_VALUE)
    pe_key: str = NullableField(alias=PE_KEY)
    pe_value: float = NullableField(alias=PE_VALUE)


class PPFSchema(pa.DataFrameModel):
    """
    Schema for Patient-Protocol Fit (PPF) data. Represents how well a protocol fits a patient, including a PPF score and feature contributions.
    """

    patient_id: int = pa.Field(alias=PATIENT_ID)
    protocol_id: int = pa.Field(alias=PROTOCOL_ID)

    ppf: float = pa.Field(alias=PPF)
    contrib: object = pa.Field(alias=CONTRIB)


class PCMSchema(pa.DataFrameModel):
    """
    Schema for protocol similarity matrix. Include pairwise similarity scores between protocols based on clinical domain overlap.
    """

    protocol_a: int = pa.Field(alias=PROTOCOL_A)
    protocol_b: int = pa.Field(alias=PROTOCOL_B)
    similarity: float = pa.Field(alias=SIMILARITY)


# ---------------------------------------------------------------------
# Recommender Output


class ScoringSchema(pa.DataFrameModel):
    """
    Schema for prescription scoring output. Represents the result of a recommendation.
    """

    class Config:
        coerce = True

    patient_id: int = pa.Field(
        alias=PATIENT_ID, gt=0, description="Must be a positive integer."
    )
    protocol_id: int = pa.Field(
        alias=PROTOCOL_ID, gt=0, description="Must be a positive integer."
    )
    adherence: float = pa.Field(
        alias=RECENT_ADHERENCE, ge=0, le=1, description="Must be a probability (0-1)."
    )
    dm: float = pa.Field(
        alias=DELTA_DM
    )  # , ge=-1, le=1, description="Must be between (-1, 1).")
    ppf: float = pa.Field(
        alias=PPF, ge=0, le=1, description="Must be a probability (0-1)."
    )
    contrib: List[float] = pa.Field(alias="CONTRIB", nullable=False, coerce=True)
    score: float = pa.Field(
        alias=SCORE, ge=0, description="Score must be a positive float."
    )
    usage: int = pa.Field(
        alias=USAGE, ge=0, description="Usage count must be a non-negative integer."
    )
    days: List[int] = pa.Field(
        alias=DAYS, description="Days of the week the protocol is prescribed."
    )


# ---------------------------------------------------------------------
# Validation Decorator


def safe_check_types(schema_model: Type[pa.DataFrameModel]):
    """
    Custom decorator: skips dtype checks for nullable columns with all null values.
    schema_model: A pandera DataFrameModel class.
    """
    schema = schema_model.to_schema()
    schema_name = schema_model.__name__  # Get the name of the schema model class

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            df: pd.DataFrame = func(*args, **kwargs)

            if df.empty:
                logger.warning(
                    "Returned DataFrame from `%s` is empty. Kwargs: %s",
                    func.__name__,
                    kwargs,
                )
                return df

            modified_columns = {}
            skipped_columns = []

            for col_name, col_schema in schema.columns.items():
                if col_schema.nullable and df[col_name].isna().all():
                    skipped_columns.append(
                        col_name
                    )  # Skip dtype validation for this nullable column with all nulls
                    modified_columns[col_name] = pa.Column(
                        dtype=None,
                        checks=col_schema.checks,
                        nullable=col_schema.nullable,
                        required=col_schema.required,
                        unique=col_schema.unique,
                        coerce=col_schema.coerce,
                        regex=col_schema.regex,
                        description=col_schema.description,
                        title=col_schema.title,
                    )
                else:
                    # Keep original schema if dtype validation is needed
                    modified_columns[col_name] = col_schema

            # Log all skipped columns once
            if skipped_columns:
                logger.debug(
                    "Skipped dtype check for empty columns in `%s`: %s",
                    schema_name,
                    ", ".join(skipped_columns),
                )

            # Reconstruct modified schema
            temp_schema = pa.DataFrameSchema(
                columns=modified_columns,
                checks=schema.checks,
                index=schema.index,
                dtype=schema.dtype,
                coerce=schema.coerce,
                strict=schema.strict,
            )

            # Perform validation
            validated_df = temp_schema.validate(df)
            return validated_df

        return wrapper

    return decorator
