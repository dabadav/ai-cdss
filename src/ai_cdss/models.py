# ai_cdss/models.py
import logging
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Any, Dict, List, Optional, Type

import pandas as pd
import pandera as pa
from ai_cdss.constants import *

NullableField = partial(pa.Field, nullable=True)

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


