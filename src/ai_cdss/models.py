# ai_cdss/models.py
import pandera as pa
import pandas as pd
from typing import List, Callable, Type
from functools import partial, wraps
import logging

NullableField = partial(pa.Field, nullable=True)

# Set up logging
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# RGS Data Input

class SessionSchema(pa.DataFrameModel):
    """
    Schema for RGS session-level data, including patient profile, prescription and session details.
    """

    # Patient profile
    patient_id: int = NullableField(alias='PATIENT_ID')
    hospital_id: int = NullableField(alias='HOSPITAL_ID')
    paretic_side: str = NullableField(alias='PARETIC_SIDE')
    upper_extremity_to_train: str = NullableField(alias='UPPER_EXTREMITY_TO_TRAIN')
    hand_raising_capacity: str = NullableField(alias='HAND_RAISING_CAPACITY')
    cognitive_function_level: str = NullableField(alias='COGNITIVE_FUNCTION_LEVEL')
    has_heminegligence: int = NullableField(alias='HAS_HEMINEGLIGENCE')
    gender: str = NullableField(alias='GENDER')
    skin_color: str = NullableField(alias='SKIN_COLOR')
    age: int = NullableField(alias='AGE', ge=0, le=120)
    videogame_exp: int = NullableField(alias='VIDEOGAME_EXP', ge=0)
    computer_exp: int = NullableField(alias='COMPUTER_EXP', ge=0)
    comments: str = NullableField(alias='COMMENTS')
    ptn_height_cm: int = NullableField(alias='PTN_HEIGHT_CM', ge=100, le=250)
    arm_size_cm: int = NullableField(alias='ARM_SIZE_CM', ge=20, le=60)

    # Identifiers
    prescription_id: int = NullableField(alias='PRESCRIPTION_ID')
    session_id: int = NullableField(alias='SESSION_ID')
    protocol_id: int = NullableField(alias='PROTOCOL_ID')

    # Prescription
    prescription_starting_date: pa.DateTime = NullableField(alias='PRESCRIPTION_STARTING_DATE')
    prescription_ending_date: pa.DateTime = NullableField(alias='PRESCRIPTION_ENDING_DATE')
    
    # Session
    session_date: pa.DateTime = NullableField(alias='SESSION_DATE')
    starting_hour: int = NullableField(alias='STARTING_HOUR', ge=0, le=23)
    starting_time_category: str = NullableField(alias='STARTING_TIME_CATEGORY', isin=["MORNING", "AFTERNOON", "EVENING", "NIGHT"])
    weekday: int = NullableField(alias='WEEKDAY_INDEX', ge=0, le=6, description="Weekday Index (0=Monday, 6=Sunday)")
    
    status: str = NullableField(alias='STATUS', isin=["CLOSED", "ABORTED", "ONGOING"])
    
    # Protocol
    protocol_type: str = NullableField(alias='PROTOCOL_TYPE')
    ar_mode: str = NullableField(alias='AR_MODE')
    
    # Metrics
    real_session_duration: int = NullableField(alias='REAL_SESSION_DURATION', ge=0)
    prescribed_session_duration: int = NullableField(alias='PRESCRIBED_SESSION_DURATION', ge=0)
    session_duration: int = NullableField(alias='SESSION_DURATION', ge=0)
    adherence: float = NullableField(alias='ADHERENCE', ge=0, le=1)

    total_success: int = NullableField(alias='TOTAL_SUCCESS', ge=0)
    total_errors: int = NullableField(alias='TOTAL_ERRORS', ge=0)
    score: int = NullableField(alias='SCORE', ge=0)

class TimeseriesSchema(pa.DataFrameModel):
    """
    Schema for timeseries session data. Includes measurements per-second of difficulty modulators (DM) and performance estimates (PE).
    """
    # Identifiers
    patient_id: int = pa.Field(alias="PATIENT_ID", gt=0)
    session_id: int = pa.Field(alias="SESSION_ID", gt=0)
    protocol_id: int = pa.Field(alias="PROTOCOL_ID", gt=0)
    
    # Protocol
    game_mode: str = pa.Field(alias="GAME_MODE")
    
    # Time
    timepoint: int = pa.Field(alias="SECONDS_FROM_START")
    
    # Metrics
    dm_key: str = pa.Field(alias="DM_KEY")
    dm_value: float = pa.Field(alias="DM_VALUE")
    pe_key: str = pa.Field(alias="PE_KEY")
    pe_value: float = pa.Field(alias="PE_VALUE")

class PPFSchema(pa.DataFrameModel):
    """
    Schema for Patient-Protocol Fit (PPF) data. Represents how well a protocol fits a patient, including a PPF score and feature contributions.
    """
    patient_id: int = pa.Field(alias="PATIENT_ID")
    protocol_id: int = pa.Field(alias="PROTOCOL_ID")

    ppf: float = pa.Field(alias="PPF")
    contrib: object = pa.Field(alias="CONTRIB")

class PCMSchema(pa.DataFrameModel):
    """
    Schema for protocol similarity matrix. Include pairwise similarity scores between protocols based on clinical domain overlap.
    """
    protocol_a: int = pa.Field(alias="PROTOCOL_A")
    protocol_b: int = pa.Field(alias="PROTOCOL_B")
    similarity: float = pa.Field(alias="SIMILARITY")

# ---------------------------------------------------------------------
# Recommender Output

class ScoringSchema(pa.DataFrameModel):
    """
    Schema for prescription scoring output. Represents the result of a recommendation.
    """

    class Config:
        coerce = True
    
    patient_id: int = pa.Field(alias="PATIENT_ID", gt=0, description="Must be a positive integer.")
    protocol_id: int = pa.Field(alias="PROTOCOL_ID", gt=0, description="Must be a positive integer.")
    adherence: float = pa.Field(alias="ADHERENCE_RECENT", ge=0, le=1, description="Must be a probability (0-1).")
    dm: float = pa.Field(alias="DELTA_DM") # , ge=-1, le=1, description="Must be between (-1, 1).")
    ppf: float = pa.Field(alias="PPF", ge=0, le=1, description="Must be a probability (0-1).")
    contrib: List[float] = pa.Field(alias="CONTRIB", nullable=False, coerce=True)
    score: float = pa.Field(alias="SCORE", ge=0, description="Score must be a positive float.")
    usage: int = pa.Field(alias="USAGE", ge=0, description="Usage count must be a non-negative integer.")
    days: List[int] = pa.Field(alias="DAYS", description="Days of the week the protocol is prescribed.")

# ---------------------------------------------------------------------
# Validation Decorator

def safe_check_types(schema_model: Type[pa.DataFrameModel]):
    """
    Custom decorator: skips dtype checks for nullable columns with all null values.
    schema_model: A pandera DataFrameModel class.
    """
    schema = schema_model.to_schema()

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            df: pd.DataFrame = func(*args, **kwargs)
            
            if df.empty:
                logger.warning(
                    f"Returned DataFrame from `{func.__name__}` is empty. "
                    f"Kwargs: {kwargs}"
                )
                return df
            
            modified_columns = {}
            skipped_columns = []

            for col_name, col_schema in schema.columns.items():
                if col_schema.nullable and df[col_name].isna().all():
                    skipped_columns.append(col_name)                    # Skip dtype validation for this nullable column with all nulls
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
                logger.info(
                    f"Skipping dtype check for nullable columns with all null values: {', '.join(skipped_columns)}"
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
