# constants.py
from pathlib import Path

############################
# DATA
############################
DEFAULT_DATA_DIR = Path.home() / ".ai_cdss" / "data"
DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_OUTPUT_DIR = Path.home() / ".ai_cdss" / "output"
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_LOG_DIR = Path.home() / ".ai_cdss" / "logs"
DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)

PPF_PARQUET_FILEPATH = DEFAULT_OUTPUT_DIR / "ppf.parquet"

############################
# PARAMETERS
############################

# Delta DM
SAVGOL_WINDOW_SIZE = 7
SAVGOL_POLY_ORDER = 2
THEILSON_REGRESSION_WINDOW_SIZE = 7


############################
# ABBREVIATIONS
############################
PATIENT_ID = "PATIENT_ID"
PROTOCOL_ID = "PROTOCOL_ID"
PRESCRIPTION_ID = "PRESCRIPTION_ID"
SESSION_ID = "SESSION_ID"

ADHERENCE = "ADHERENCE"
RECENT_ADHERENCE = "ADHERENCE_RECENT"

DM_KEY = "DM_KEY"
PE_KEY = "PE_KEY"

DM_VALUE = "DM_VALUE"
PE_VALUE = "PE_VALUE"

DELTA_DM = "DELTA_DM"

PPF = "PPF"
CONTRIB = "CONTRIB"

USAGE = "USAGE"
USAGE_WEEK = "USAGE_WEEK"
DAYS = "DAYS"

SESSION_DATE = "SESSION_DATE"
GAME_MODE = "GAME_MODE"
SECONDS_FROM_START = "SECONDS_FROM_START"
WEEKDAY_INDEX = "WEEKDAY_INDEX"
PRESCRIPTION_ENDING_DATE = "PRESCRIPTION_ENDING_DATE"
PRESCRIPTION_ACTIVE = "2100-01-01"

SCORE = "GAME_SCORE"

# Session columns
SESSION_COLUMNS: list[str] = [
    "SESSION_ID", "TOTAL_SUCCESS", "TOTAL_ERRORS", "GAME_SCORE"
]
METRIC_COLUMNS: list[str] = [
    "DM_VALUE", "PE_VALUE"
]
TIME_COLUMNS: list[str] = [
    "SESSION_DATE", "STARTING_HOUR", "STARTING_TIME_CATEGORY", "SECONDS_FROM_START"
]

# Common sets of columns
BY_PP: list[str] = [PATIENT_ID, PROTOCOL_ID] # By Patient–Protocol
BY_PPS: list[str] = [PATIENT_ID, PROTOCOL_ID, SESSION_ID] # By Patient–Protocol–Session
BY_PPST: list[str] = [PATIENT_ID, PROTOCOL_ID, SESSION_ID, SECONDS_FROM_START] # By PPS + Time
BY_ID: list[str] = BY_PPS + [PRESCRIPTION_ID] # By Patient–Protocol–Session + Prescription

# Metrics
FINAL_METRICS = [PPF, CONTRIB, RECENT_ADHERENCE, DELTA_DM, USAGE, USAGE_WEEK, DAYS, SCORE]
