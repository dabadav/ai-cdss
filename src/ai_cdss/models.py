import pandera as pa
from typing import List
from pandera.typing import Series
from datetime import date, datetime
from functools import partial

NullableField = partial(pa.Field, nullable=True)

class SessionSchema(pa.DataFrameModel):
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

    prescription_id: int = NullableField(alias='PRESCRIPTION_ID')
    session_id: int = NullableField(alias='SESSION_ID')
    protocol_id: int = NullableField(alias='PROTOCOL_ID')
    prescription_starting_date: pa.DateTime = NullableField(alias='PRESCRIPTION_STARTING_DATE')
    prescription_ending_date: pa.DateTime = NullableField(alias='PRESCRIPTION_ENDING_DATE')
    session_date: pa.DateTime = NullableField(alias='SESSION_DATE')
    starting_hour: int = NullableField(alias='STARTING_HOUR', ge=0, le=23)
    starting_time_category: str = NullableField(alias='STARTING_TIME_CATEGORY', isin=["MORNING", "AFTERNOON", "EVENING", "NIGHT"])
    weekday: int = NullableField(alias='WEEKDAY_INDEX', ge=0, le=6, description="Weekday Index (0=Monday, 6=Sunday)")
    status: str = NullableField(alias='STATUS', isin=["CLOSED", "ABORTED", "ONGOING"])
    protocol_type: str = NullableField(alias='PROTOCOL_TYPE')
    ar_mode: str = NullableField(alias='AR_MODE')
    real_session_duration: int = NullableField(alias='REAL_SESSION_DURATION', ge=0)
    prescribed_session_duration: int = NullableField(alias='PRESCRIBED_SESSION_DURATION', ge=0)
    session_duration: int = NullableField(alias='SESSION_DURATION', ge=0)
    adherence: float = NullableField(alias='ADHERENCE', ge=0, le=1)

    total_success: int = NullableField(alias='TOTAL_SUCCESS', ge=0)
    total_errors: int = NullableField(alias='TOTAL_ERRORS', ge=0)
    score: int = NullableField(alias='SCORE', ge=0)

    class Config:
        coerce = False
        strict = False

class SessionProcessedSchema(SessionSchema):
    PATIENT_ID: int = pa.Field(alias='patient_id', nullable=False)
    HOSPITAL_ID: int = pa.Field(alias='hospital_id', nullable=False)
    PRESCRIPTION_ID: int = pa.Field(alias='prescription_id', nullable=False)
    SESSION_ID: int = pa.Field(alias='session_id', nullable=False)
    HOSPITAL_ID: int = pa.Field(alias='hospital_id', nullable=False)
    PRESCRIPTION_STARTING_DATE: date = pa.Field(alias='prescription_starting_date', nullable=False)
    PRESCRIPTION_ENDING_DATE: date = pa.Field(alias='prescription_ending_date', nullable=False)
    SESSION_DATE: date = pa.Field(alias='session_date', nullable=False)
    STARTING_HOUR: int = pa.Field(alias='starting_hour', nullable=False, ge=0, le=23)
    STARTING_TIME: str = pa.Field(alias='starting_hour', nullable=False)
    PARETIC_SIDE: str = pa.Field(alias='paretic_side', nullable=False)
    WEEKDAY: str = pa.Field(alias='weekday', nullable=False, isin=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    STATUS: str = pa.Field(alias='status', nullable=False, isin=["CLOSED", "ABORTED", "ONGOING"])
    PROTOCOL_TYPE: str = pa.Field(alias='protocol_type', nullable=False)
    AR_MODE: str = pa.Field(alias='ar_mode', nullable=False)
    STARTING_HOUR: int = pa.Field(alias='starting_hour', nullable=False, ge=0, le=23)
    REAL_STARTING_HOUR: int = pa.Field(alias='real_starting_hour', nullable=False, ge=0, le=23)
    PROTOCOL_TYPE: str = pa.Field(alias='protocol_type', nullable=False)
    SESSION_DURATION_MINUTES: int = pa.Field(alias='session_duration', nullable=False, ge=0)
    ADHERENCE: float = pa.Field(alias='adherence', nullable=True, ge=0, le=1)
    TOTAL_SUCCESS: int = pa.Field(alias='total_success', nullable=False, ge=0)
    TOTAL_ERRORS: int = pa.Field(alias='total_errors', nullable=False, ge=0)

    class Config:
        name = "SessionProcessedSchema"
        strict = False # allow additional columns

class PrescriptionSchema(pa.DataFrameModel):
    """
    Prescription output validation schema.
    """
    patient_id: int = pa.Field(gt=0, description="Must be a positive integer.")
    protocol_id: int = pa.Field(gt=0, description="Must be a positive integer.")
    ppf: float = pa.Field(ge=0, le=1, description="Must be a probability (0-1).")
    adherence: float = pa.Field(ge=0, le=1, description="Must be a probability (0-1).")
    dm: float = pa.Field(ge=0, description="Must be non-negative.")
    contribution: Series[list] = pa.Field(nullable=False, coerce=True)
    score: float = pa.Field(ge=0, description="Score must be a positive float.")
    days: Series[List[int]] = pa.Field(nullable=False, coerce=True)

    # Custom validation checks
    @pa.check("contribution")
    def check_contribution_sum(cls, contribution: Series[List[float]]) -> Series[bool]:
        return contribution.apply(
            lambda lst: isinstance(lst, list) and 
            all(isinstance(x, (int, float)) for x in lst) and 
            abs(sum(lst) - 1.0) < 1e-6
        )

    @pa.check("schedule_days")
    def check_no_repeated_days(cls, days: Series[List[int]]) -> Series[bool]:
        return days.apply(lambda lst: len(lst) == len(set(lst)))

class PatientSchema(pa.DataFrameModel):
    """Pandera schema for patient clinical scores validation."""
    
    patient_id: Series[int]
    barthel: Series[int]
    ash_proximal: Series[int]
    ma_distal: Series[int]
    fatigue: Series[int]  # Can be float if needed
    vas: float
    fm_a: Series[int]
    fm_b: Series[int]
    fm_c: Series[int]
    fm_d: Series[int]
    fm_total: Series[int]
    act_au: float  # Can contain float values
    act_qom: float  # Can contain float values

    class Config:
        coerce = True  # Auto-cast types to expected types

class ProtocolMatrixSchema(pa.DataFrameModel):
    protocol_name: str
    protocol_id: Series[int]
    difficulty_cognitive: float
    difficulty_motor: float
    body_part_finger: float
    body_part_wrist: float
    body_part_arm: float
    body_part_shoulder: float
    body_part_trunk: float
    reaching: float
    grasping: float
    pinching: float
    pronation_supination: float
    range_of_motion_h: float
    range_of_motion_v: float
    processing_speed: float
    attention: float
    visual_language: float
    visualspatial_processing_awareness_neglect: float
    coordination: float
    memory_wm: float
    memory_semantic: float
    math: float
    daily_living_activity: float
    symbolic_understanding: float
    semantic_processing: float
