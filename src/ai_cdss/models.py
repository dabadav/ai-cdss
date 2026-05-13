# ai_cdss/models.py
"""Pandera DataFrame schemas for the canonical input/output frames.

Only schemas live here after F4b. The DataUnit / DataUnitSet /
Granularity / DataUnitName machinery was removed — it was a 5-field
wrapper around a single DataFrame where 3 of the 5 fields had no
caller. The pipeline now takes `RawInputs` (in `pipeline.py`) and
returns `ScoringOutput` directly.
"""
import logging
from functools import partial
from typing import List

import pandera as pa
from ai_cdss.constants import *

NullableField = partial(pa.Field, nullable=True)

logger = logging.getLogger(__name__)


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


