import pandera as pa
from pandera.typing import DataFrame, Series
import pandas as pd
from dataclasses import dataclass
from datetime import date

class SessionBatchSchema(pa.DataFrameModel):
    PATIENT_ID: Series[pa.Int64] = pa.Field(nullable=False)
    HOSPITAL_ID: Series[pa.Int64] = pa.Field(nullable=False)
    PARETIC_SIDE: Series[str] = pa.Field(nullable=False)
    UPPER_EXTREMITY_TO_TRAIN: Series[str] = pa.Field(nullable=False)
    HAND_RAISING_CAPACITY: Series[str] = pa.Field(nullable=False)
    COGNITIVE_FUNCTION_LEVEL: Series[str] = pa.Field(nullable=False)
    HAS_HEMINEGLIGENCE: Series[pa.Int64] = pa.Field(nullable=False)
    GENDER: Series[str] = pa.Field(nullable=False)
    SKIN_COLOR: Series[str] = pa.Field(nullable=False)
    AGE: Series[pa.Int64] = pa.Field(nullable=False, ge=0, le=120)  # Validate reasonable age range
    VIDEOGAME_EXP: Series[pa.Int64] = pa.Field(nullable=False, ge=0)
    COMPUTER_EXP: Series[pa.Int64] = pa.Field(nullable=False, ge=0)
    COMMENTS: Series[str] = pa.Field(nullable=True)  # Allow null values
    PTN_HEIGHT_CM: Series[pa.Int64] = pa.Field(nullable=False, ge=100, le=250)  # Reasonable height range
    ARM_SIZE_CM: Series[pa.Int64] = pa.Field(nullable=False, ge=20, le=60)  # Reasonable arm size
    PRESCRIPTION_ID: Series[pa.Int64] = pa.Field(nullable=False)
    SESSION_ID: Series[pa.Int64] = pa.Field(nullable=False)
    PROTOCOL_ID: Series[pa.Int64] = pa.Field(nullable=False)
    PRESCRIPTION_STARTING_DATE: Series[date] = pa.Field(nullable=False)
    PRESCRIPTION_ENDING_DATE: Series[date] = pa.Field(nullable=False)
    SESSION_DATE: Series[date] = pa.Field(nullable=False)
    STARTING_HOUR: Series[pa.Int64] = pa.Field(nullable=False, ge=0, le=23)  # Hours range
    STARTING_TIME_CATEGORY: Series[str] = pa.Field(nullable=False, isin=["MORNING", "AFTERNOON", "EVENING", "NIGHT"])
    STATUS: Series[str] = pa.Field(nullable=False, isin=["CLOSED", "ABORTED", "ONGOING"])
    PROTOCOL_TYPE: Series[str] = pa.Field(nullable=False)
    AR_MODE: Series[str] = pa.Field(nullable=False)
    WEEKDAY: Series[str] = pa.Field(nullable=False, isin=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    REAL_SESSION_DURATION: Series[pa.Int64] = pa.Field(nullable=False, ge=0)
    PRESCRIBED_SESSION_DURATION: Series[pa.Int64] = pa.Field(nullable=False, ge=0)
    SESSION_DURATION: Series[pa.Int64] = pa.Field(nullable=False, ge=0)
    ADHERENCE: Series[float] = pa.Field(nullable=True, ge=0, le=1)  # Allow for some adherence over-prescription but within bounds
    TOTAL_SUCCESS: Series[pa.Int64] = pa.Field(nullable=False, ge=0)
    TOTAL_ERRORS: Series[pa.Int64] = pa.Field(nullable=False, ge=0)
    SCORE: Series[float] = pa.Field(nullable=True, ge=0)  # Scores cannot be negative

@dataclass
class DataBatch:
    """
    DataBatch class
    """
    session_batch: DataFrame[SessionBatchSchema]
    patient_data: pd.DataFrame
    protocol_data: pd.DataFrame
