from typing import List, Optional

import numpy as np
import pandas as pd
from ai_cdss.constants import *
from pandas import Timestamp
from scipy.signal import savgol_filter
from sklearn.linear_model import TheilSenRegressor

# ---------------------------------------------------------------
# Adherence
# ---------------------------------------------------------------


def include_missing_sessions(session: pd.DataFrame):
    """
    For each prescription, generate expected sessions (based on weekday, start and end dates),
    and merge them with actual performed sessions. Sessions that were expected but not performed
    are marked as NOT_PERFORMED.

    All date comparisons are done at day-level (time is ignored).
    """

    # Normalize to date (drop time component)
    date_cols = [SESSION_DATE, PRESCRIPTION_STARTING_DATE, PRESCRIPTION_ENDING_DATE]
    for col in date_cols:
        session[col] = pd.to_datetime(session[col]).dt.normalize()

    # Get last performed session date per patient
    valid_sessions = session.dropna(subset=[SESSION_DATE])

    last_session_per_patient = (
        valid_sessions.groupby(PATIENT_ID)[SESSION_DATE].max().to_dict()
    )

    # Get exisiting prescriptions
    prescriptions = session.drop_duplicates(
        subset=[
            PRESCRIPTION_ID,
            PATIENT_ID,
            PROTOCOL_ID,
            PRESCRIPTION_STARTING_DATE,
            PRESCRIPTION_ENDING_DATE,
            WEEKDAY_INDEX,
        ]
    )

    # Generate expected session dates
    expected_session_rows = []

    for _, row in prescriptions.iterrows():
        patient_id = row[PATIENT_ID]
        start = row[PRESCRIPTION_STARTING_DATE]
        end = row[PRESCRIPTION_ENDING_DATE]
        weekday = row[WEEKDAY_INDEX]

        # Safety
        if pd.isna(start) or pd.isna(end) or pd.isna(weekday):
            continue

        # Cap at last performed session for the patient
        last_session = last_session_per_patient.get(
            patient_id, pd.Timestamp.today().normalize()
        )
        # If the prescription end date is in the future, use today as the end limit (assuming future sessions are not yet done)
        end = min(end, last_session)

        # Generate expected session dates for this prescription
        expected_dates = generate_expected_sessions(
            start, end, int(weekday)
        )  # Should return date-like list

        # Fill rows with NOT_PERFORMED status
        for session_date in expected_dates:
            row_dict = {
                **row.to_dict(),
                SESSION_DATE: pd.to_datetime(session_date).normalize(),
                STATUS: SessionStatus.NOT_PERFORMED,
                ADHERENCE: 0.0,
                SESSION_DURATION: 0,
                REAL_SESSION_DURATION: 0,
            }
            # Overwrite session metric columns with NaN
            row_dict.update({col: np.nan for col in SESSION_COLUMNS})
            expected_session_rows.append(row_dict)

    expected_df = pd.DataFrame(expected_session_rows)

    if expected_df.empty:
        return session

    # Filter out already performed sessions
    performed_index = pd.MultiIndex.from_frame(
        valid_sessions[[PRESCRIPTION_ID, SESSION_DATE]]
    )
    expected_index = pd.MultiIndex.from_frame(
        expected_df[[PRESCRIPTION_ID, SESSION_DATE]]
    )
    # Identify expected sessions that were not actually performed
    # (i.e., keep only those not present in the performed session index)
    mask = ~expected_index.isin(performed_index)
    expected_df = expected_df.loc[mask]

    # Merge performed sessions with expected sessions
    session_all = pd.concat([valid_sessions, expected_df], ignore_index=True)

    return session_all.sort_values(
        by=[PATIENT_ID, PRESCRIPTION_ID, SESSION_DATE]
    ).reset_index(drop=True)


def generate_expected_sessions(
    start: Timestamp, end: Timestamp, weekday: int
) -> List[Timestamp]:
    """
    Generate session dates between start and end for the given weekday index.
    Weekday: 0=Monday, 1=Tuesday, ..., 6=Sunday
    """
    weekday_map = {
        0: "W-MON",
        1: "W-TUE",
        2: "W-WED",
        3: "W-THU",
        4: "W-FRI",
        5: "W-SAT",
        6: "W-SUN",
    }

    freq = weekday_map.get(weekday)
    if freq is None:
        return []

    return list(pd.date_range(start=start, end=end, freq=freq))


def compute_ewma(df, value_col, group_cols, sufix="", alpha=EWMA_ALPHA):
    """
    Compute the exponentially weighted moving average (EWMA) of a value column grouped by specified columns.

    Args:
        df (DataFrame): Input DataFrame.
        value_col (str): Name of the value column to smooth.
        group_cols (list): Columns to group by.
        sufix (str): Suffix to append to the new column name.
        alpha (float): Smoothing factor for EWMA. Default is EWMA_ALPHA from constants.

    Returns:
        DataFrame: DataFrame with the new EWMA column added.
    """
    return df.assign(
        **{
            f"{value_col}{sufix}": df.groupby(by=group_cols)[value_col].transform(
                lambda x: x.ewm(alpha=alpha, adjust=True).mean()
            )
        }
    )


# ---------------------------------------------------------------
# Delta DM
# ---------------------------------------------------------------


def apply_savgol_filter_groupwise(series, window_size, polyorder):
    series_len = len(series)
    if series_len < polyorder + 1:
        return series
    window = min(window_size, series_len)
    if window <= polyorder:
        window = polyorder + 1
    if window > series_len:
        return series
    if window % 2 == 0:
        window -= 1
    if window <= polyorder:
        return series
    try:
        return savgol_filter(series, window_length=window, polyorder=polyorder)
    except ValueError:
        return series


def get_rolling_theilsen_slope(series_y, series_x, window_size):
    slopes = pd.Series([np.nan] * len(series_y), index=series_y.index)
    if len(series_y) < 2:
        return slopes
    regressor = TheilSenRegressor(random_state=42, max_subpopulation=1000)
    for i in range(len(series_y)):
        start_index = max(0, i - window_size + 1)
        window_y = series_y.iloc[start_index : i + 1]
        window_x = series_x.iloc[start_index : i + 1]
        if len(window_y) < 2:
            slopes.iloc[i] = 0.0 if len(window_y) == 1 else np.nan
            continue
        if len(window_x.unique()) == 1 and len(window_y.unique()) > 1:
            slopes.iloc[i] = np.nan
            continue
        if len(window_y.unique()) == 1:
            slopes.iloc[i] = 0.0
            continue
        X_reshaped = window_x.values.reshape(-1, 1)
        try:
            regressor.fit(X_reshaped, window_y.values)
            slopes.iloc[i] = regressor.coef_[0]
        except Exception:
            slopes.iloc[i] = np.nan
    return slopes


# ---------------------------------------------------------------
# PPF
# ---------------------------------------------------------------


def feature_contributions(df_A, df_B):
    # Convert to numpy
    A = df_A.to_numpy()  # (patients, subscales)
    B = df_B.to_numpy()  # (protocols, subscales)

    # Compute row-wise norms
    A_norms = np.linalg.norm(A, axis=1, keepdims=True)  # (patients, 1)
    B_norms = np.linalg.norm(B, axis=1, keepdims=True)  # (protocols, 1)

    # Replace zero norms with a small value to avoid NaN (division by zero)
    A_norms[A_norms == 0] = 1e-10
    B_norms[B_norms == 0] = 1e-10

    # Normalize each row to unit vectors
    A_norm = A / A_norms  # (patient, subscales)
    B_norm = B / B_norms  # (protocol, subscales)

    # Compute feature contributions
    contributions = (
        A_norm[:, np.newaxis, :] * B_norm[np.newaxis, :, :]
    )  # (patient, dim, subscales) * (dim, protocol, subscales)

    return contributions  # (patients, protocols, subscales_sim)


def compute_ppf(patient_deficiency, protocol_mapped):
    """Compute the patient-protocol feature matrix (PPF) and feature contributions."""
    contributions = feature_contributions(patient_deficiency, protocol_mapped)
    ppf = np.sum(contributions, axis=2)  # (patients, protocols, cosine)
    ppf = pd.DataFrame(
        ppf, index=patient_deficiency.index, columns=protocol_mapped.index
    )
    contributions = pd.DataFrame(
        contributions.tolist(),
        index=patient_deficiency.index,
        columns=protocol_mapped.index,
    )

    ppf_long = ppf.stack().reset_index()
    ppf_long.columns = BY_PP + [PPF]

    contrib_long = contributions.stack().reset_index()
    contrib_long.columns = BY_PP + [CONTRIB]

    return ppf_long, contrib_long


def compute_protocol_similarity(
    protocol_mapped,
    id_col: str = "PROTOCOL_ID",
    hot_encoded_prefix: Optional[str] = None,
):
    """Compute pairwise protocol similarity."""
    import gower

    protocol_attributes = protocol_mapped.copy().reset_index()
    protocol_ids = protocol_attributes[id_col]
    protocol_attributes = protocol_attributes.drop(columns=[id_col])

    # Weighting for hot-encoded columns (if present)
    if hot_encoded_prefix:
        hot_encoded_cols = protocol_attributes.columns.str.startswith(
            hot_encoded_prefix
        )
        weights = np.ones(len(protocol_attributes.columns))
        if hot_encoded_cols.any():
            weights[hot_encoded_cols] /= hot_encoded_cols.sum()
    else:
        weights = np.ones(len(protocol_attributes.columns))

    # Ensure all attributes are numeric (float)
    try:
        protocol_attributes = protocol_attributes.astype(float)
    except Exception as e:
        raise ValueError(
            "Non-numeric columns found. Please encode categorical variables before passing."
        ) from e

    # Compute Gower similarity
    gower_sim_matrix = gower.gower_matrix(protocol_attributes, weight=weights)
    # Convert to dataframe with similarity (1 - distance)
    gower_sim_df = (
        pd.DataFrame(1 - gower_sim_matrix, index=protocol_ids, columns=protocol_ids)
        .stack()
        .rename_axis([PROTOCOL_A, PROTOCOL_B])
        .reset_index()
    )
    gower_sim_df.columns = [PROTOCOL_A, PROTOCOL_B, SIMILARITY]

    return gower_sim_df
