import pytest
from ai_cdss.models import SessionSchema, TimeseriesSchema
import numpy as np
import pandas as pd

@pytest.fixture
def synthetic_session_data(request):
    """
    Generates structured synthetic session data.

    - num_patients: Number of unique patients
    - num_protocols: Protocols per patient
    - num_sessions: Sessions per patient
    - columns_with_nulls: List of columns that should have null values
    """
    np.random.seed(42)

    num_patients = request.param.get("num_patients", 3)
    num_protocols = request.param.get("num_protocols", 2)
    num_sessions = request.param.get("num_sessions", 3)
    columns_with_nulls = request.param.get("columns_with_nulls", [])

    data = []
    session_ids = []

    for patient_id in range(1, num_patients + 1):
        for protocol_id in range(1, num_protocols + 1):
            for _ in range(num_sessions):
                session_id = np.random.randint(15000, 20000)
                session_ids.append((patient_id, protocol_id, session_id))

                session_data = {
                    # Patient Information
                    "PATIENT_ID": patient_id,
                    "HOSPITAL_ID": np.random.randint(1, 50),
                    "PARETIC_SIDE": np.random.choice(["LEFT", "RIGHT"]),
                    "UPPER_EXTREMITY_TO_TRAIN": np.random.choice(["LEFT", "RIGHT"]),
                    "HAND_RAISING_CAPACITY": np.random.choice(["LOW", "MEDIUM", "HIGH"]),
                    "COGNITIVE_FUNCTION_LEVEL": np.random.choice(["LOW", "MEDIUM", "HIGH"]),
                    "HAS_HEMINEGLIGENCE": np.random.choice([0, 1]),
                    "GENDER": np.random.choice(["MALE", "FEMALE"]),
                    "SKIN_COLOR": np.random.choice(["FDC3AD", "E0AC69", "8D5524"]),
                    "AGE": np.random.randint(18, 90),

                    # Handle nullable AGE correctly as `float`, convert back to `Int64`
                    "VIDEOGAME_EXP": np.random.randint(0, 5),
                    "COMPUTER_EXP": np.random.randint(0, 5),
                    "COMMENTS": "Test comment",

                    # Physical Attributes
                    "PTN_HEIGHT_CM": np.random.randint(150, 190),
                    "ARM_SIZE_CM": np.random.randint(20, 50),

                    # Identifiers
                    "PRESCRIPTION_ID": np.random.randint(70000, 80000),
                    "SESSION_ID": session_id,
                    "PROTOCOL_ID": protocol_id,

                    # Prescription Info (Convert to datetime)
                    "PRESCRIPTION_STARTING_DATE": pd.to_datetime("2024-03-28 08:55:00"),
                    "PRESCRIPTION_ENDING_DATE": pd.to_datetime("2100-01-01 00:00:00"),

                    # Session Info
                    "SESSION_DATE": pd.to_datetime("2024-03-29"),
                    "STARTING_HOUR": np.random.randint(0, 23),
                    "STARTING_TIME_CATEGORY": np.random.choice(["MORNING", "AFTERNOON", "EVENING", "NIGHT"]),
                    "WEEKDAY_INDEX": np.random.randint(0, 7),
                    "STATUS": np.random.choice(["CLOSED", "ABORTED", "ONGOING"]),

                    # Protocol Info
                    "PROTOCOL_TYPE": np.random.choice(["Hands", "AR"]),
                    "AR_MODE": np.random.choice(["NONE", "TABLE"]),

                    # Metrics
                    "REAL_SESSION_DURATION": np.random.randint(200, 600),
                    "PRESCRIBED_SESSION_DURATION": np.random.randint(200, 600),
                    "SESSION_DURATION": np.random.randint(200, 600),
                    "ADHERENCE": np.round(np.random.uniform(0.5, 1.0), 2),

                    "TOTAL_SUCCESS": np.random.randint(50, 100),
                    "TOTAL_ERRORS": np.random.randint(0, 20),
                    "SCORE": np.random.randint(50, 300),
                }

                # Introduce null values in specified columns
                for col in columns_with_nulls:
                    session_data[col] = None

                data.append(session_data)

    df = pd.DataFrame(data)
    validated_df = SessionSchema.validate(df)
    return validated_df, session_ids  # Return session_ids for timeseries consistency

@pytest.fixture
def synthetic_timeseries_data(request, synthetic_session_data):
    """
    Generates structured synthetic timeseries data.

    - num_timepoints: Number of timepoints per session
    - columns_with_nulls: List of columns that should have null values
    - test_discrepancies: If True, introduces session mismatches
    """
    np.random.seed(42)

    num_timepoints = request.param.get("num_timepoints", 10)
    columns_with_nulls = request.param.get("columns_with_nulls", [])
    test_discrepancies = request.param.get("test_discrepancies", False)

    _, session_ids = synthetic_session_data  # Ensure consistency with session fixture
    data = []

    for patient_id, protocol_id, session_id in session_ids:
        if test_discrepancies and np.random.rand() < 0.2:
            # Introduce mismatch: either change session_id or add a non-matching patient
            session_id = session_id + 10000  # Non-existent session ID
            patient_id = patient_id + 50  # Non-existent patient

        for timepoint in range(1, num_timepoints + 1):
            timeseries_data = {
                "SESSION_ID": session_id,
                "PATIENT_ID": patient_id,
                "PROTOCOL_ID": protocol_id,
                "GAME_MODE": np.random.choice(["STANDARD", "PAY", "SPELL_WORDS"]),
                "SECONDS_FROM_START": timepoint * np.random.randint(1000, 5000),
                "DM_KEY": f"param_{np.random.randint(1, 5)}",
                "DM_VALUE": np.round(np.random.uniform(0, 10), 2),
                "PE_KEY": f"metric_{np.random.randint(1, 3)}",
                "PE_VALUE": np.round(np.random.uniform(0, 100), 2)
            }

            for col in columns_with_nulls:
                timeseries_data[col] = None

            data.append(timeseries_data)

    df = pd.DataFrame(data)
    validated_df = TimeseriesSchema.validate(df)
    return validated_df


