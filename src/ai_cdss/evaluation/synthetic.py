import pandas as pd
import numpy as np
from ai_cdss.models import SessionSchema, TimeseriesSchema, PPFSchema, PCMSchema

# -- Synthetic Functions

def generate_synthetic_data(
    num_patients=3,
    num_protocols=2,
    num_sessions=3,
    timepoints=10,
    null_cols_session=None,
    null_cols_timeseries=None,
    test_discrepancies=False 
):
    shared_ids = generate_synthetic_ids(
        num_patients=num_patients,
        num_protocols=num_protocols,
        num_sessions=num_sessions
    )

    session_df = generate_synthetic_session_data(
        shared_ids, columns_with_nulls=null_cols_session
    )

    timeseries_df = generate_synthetic_timeseries_data(
        shared_ids,
        num_timepoints=timepoints,
        columns_with_nulls=null_cols_timeseries,
        test_discrepancies=test_discrepancies
    )

    ppf_df = generate_synthetic_ppf_data(shared_ids)

    return session_df, timeseries_df, ppf_df

def generate_synthetic_ids(num_patients=3, num_protocols=2, num_sessions=3):
    np.random.seed(42)
    session_ids = []
    for patient_id in range(1, num_patients + 1):
        for protocol_id in range(1, num_protocols + 1):
            for _ in range(num_sessions):
                session_id = np.random.randint(15000, 20000)
                session_ids.append((patient_id, protocol_id, session_id))
    return session_ids

# -- Synthetic Data for Session and Timeseries

def generate_synthetic_session_data(shared_ids, columns_with_nulls=[]):
    """
    Generates structured synthetic session data using default shared IDs.
    """
    np.random.seed(42)
    data = []
    columns_with_nulls = columns_with_nulls or []

    for patient_id, protocol_id, session_id in shared_ids:
        session_data = {
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
            "VIDEOGAME_EXP": np.random.randint(0, 5),
            "COMPUTER_EXP": np.random.randint(0, 5),
            "COMMENTS": "Test comment",
            "PTN_HEIGHT_CM": np.random.randint(150, 190),
            "ARM_SIZE_CM": np.random.randint(20, 50),
            "PRESCRIPTION_ID": np.random.randint(70000, 80000),
            "SESSION_ID": session_id,
            "PROTOCOL_ID": protocol_id,
            "PRESCRIPTION_STARTING_DATE": pd.to_datetime("2024-03-28 08:55:00"),
            "PRESCRIPTION_ENDING_DATE": pd.to_datetime("2100-01-01 00:00:00"),
            "SESSION_DATE": pd.to_datetime("2024-03-29"),
            "STARTING_HOUR": np.random.randint(0, 23),
            "STARTING_TIME_CATEGORY": np.random.choice(["MORNING", "AFTERNOON", "EVENING", "NIGHT"]),
            "WEEKDAY_INDEX": np.random.randint(0, 7),
            "STATUS": np.random.choice(["CLOSED", "ABORTED", "ONGOING"]),
            "PROTOCOL_TYPE": np.random.choice(["Hands", "AR"]),
            "AR_MODE": np.random.choice(["NONE", "TABLE"]),
            "REAL_SESSION_DURATION": np.random.randint(200, 600),
            "PRESCRIBED_SESSION_DURATION": np.random.randint(200, 600),
            "SESSION_DURATION": np.random.randint(200, 600),
            "ADHERENCE": np.round(np.random.uniform(0.5, 1.0), 2),
            "TOTAL_SUCCESS": np.random.randint(50, 100),
            "TOTAL_ERRORS": np.random.randint(0, 20),
            "SCORE": np.random.randint(50, 300),
        }

        for col in columns_with_nulls:
            session_data[col] = None

        data.append(session_data)

    df = pd.DataFrame(data)
    return SessionSchema.validate(df)

def generate_synthetic_timeseries_data(shared_ids, num_timepoints=10, columns_with_nulls=[], test_discrepancies=False):
    """
    Generates structured synthetic timeseries data using default shared IDs.
    """
    np.random.seed(42)
    
    data = []

    for patient_id, protocol_id, session_id in shared_ids:
        if test_discrepancies and np.random.rand() < 0.2:
            session_id += 10000
            patient_id += 50

        for timepoint in range(1, num_timepoints + 1):
            row = {
                "PATIENT_ID": patient_id,
                "SESSION_ID": session_id,
                "PROTOCOL_ID": protocol_id,
                "GAME_MODE": np.random.choice(["STANDARD", "PAY", "SPELL_WORDS"]),
                "SECONDS_FROM_START": timepoint * np.random.randint(1000, 5000),
                "DM_KEY": f"param_{np.random.randint(1, 5)}",
                "DM_VALUE": np.round(np.random.uniform(0, 10), 2),
                "PE_KEY": f"metric_{np.random.randint(1, 3)}",
                "PE_VALUE": np.round(np.random.uniform(0, 100), 2)
            }

            for col in columns_with_nulls:
                row[col] = None

            data.append(row)

    df = pd.DataFrame(data)
    return TimeseriesSchema.validate(df)

def generate_synthetic_ppf_data(shared_ids, num_features=5):
    """
    Generates synthetic PPF data based on shared session IDs.

    Each PPF value is a random float in [0.3, 1.0], and the corresponding
    CONTRIB vector (feature contributions) is a Dirichlet-distributed vector
    scaled to sum to the PPF value.

    Parameters
    ----------
    shared_ids : List[Tuple[int, int, int]]
        List of (PATIENT_ID, PROTOCOL_ID, SESSION_ID) tuples.

    num_features : int, optional
        Number of features contributing to PPF score (default = 5).

    Returns
    -------
    pd.DataFrame
        With columns: PATIENT_ID, PROTOCOL_ID, PPF, CONTRIB
    """
    np.random.seed(42)

    patient_protocol_pairs = sorted(set((p, prot) for p, prot, _ in shared_ids))
    num_pairs = len(patient_protocol_pairs)

    # Step 1: Generate one random PPF per pair
    ppf_values = np.round(np.random.uniform(0.3, 1.0, size=num_pairs), 6)

    # Step 2: Generate Dirichlet proportions
    proportions = np.random.dirichlet(np.ones(num_features), size=num_pairs)

    # Step 3: Scale contributions to sum to the corresponding PPF
    contribs = np.round(proportions * ppf_values[:, None], 6)

    # Optional: fix last value to make sure exact sum due to rounding
    for i in range(num_pairs):
        contribs[i, -1] = np.round(ppf_values[i] - contribs[i, :-1].sum(), 6)

    # Step 4: Assemble dataframe
    data = []
    for i, (patient_id, protocol_id) in enumerate(patient_protocol_pairs):
        data.append({
            "PATIENT_ID": patient_id,
            "PROTOCOL_ID": protocol_id,
            "PPF": ppf_values[i],
            "CONTRIB": contribs[i].tolist()
        })

    df = pd.DataFrame(data)
    return PPFSchema.validate(df)

# -- Synthetic Protocol Similarity

def generate_synthetic_protocol_similarity(num_protocols=5, seed=42):
    """
    Generate synthetic protocol similarity data.

    Parameters
    ----------
    num_protocols : int
        Number of unique protocols (default is 5).
    seed : int
        Random seed for reproducibility (default is 42).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: PROTOCOL_ID_1, PROTOCOL_ID_2, SIMILARITY_SCORE.
    """
    np.random.seed(seed)
    data = []

    protocol_ids = list(range(1, num_protocols + 1))

    for i in range(len(protocol_ids)):
        for j in range(i + 1, len(protocol_ids)):
            data.append({
                "PROTOCOL_A": protocol_ids[i],
                "PROTOCOL_B": protocol_ids[j],
                "SIMILARITY": round(np.random.uniform(0.5, 1.0), 3)
            })

    df = pd.DataFrame(data)
    return PCMSchema.validate(df)
