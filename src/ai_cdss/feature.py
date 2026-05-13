"""Feature engineering — one file, organized by tensor aggregation level.

The data is a tensor with axes
`patient × protocol × prescription × session × time → values`.
Every feature in this file is a **reduction** over one or more of those
axes. The file is sectioned by which axis gets reduced — read top-to-
bottom for the full feature catalog, jump to a section to dig into one
level.

    Section 1 — Time-axis primitives
                EWMA, Savitzky-Golay, Theil-Sen — reduce over time
                within a single (patient, protocol) cell.
    Section 2 — Session-shape primitives
                include_missing_sessions, generate_expected_sessions —
                fill in skipped prescriptions before features run.
    Section 3 — Per-(patient, protocol, session, time) features
                build_delta_dm, build_recent_adherence.
                Output: one row per (PP, session_date).
    Section 4 — Per-(patient, protocol) aggregations
                build_usage, build_week_usage, build_prescription_days.
                Output: one row per (PP).
    Section 5 — Per-patient features
                build_week_since_start (patient scalar) +
                _last_completed_week_window helper.
    Section 6 — Cross-cohort features
                compute_ppf, compute_protocol_similarity. Operate on
                patient × subscale and protocol × attribute matrices.
    Section 7 — FeatureBuilder class
                Object-oriented surface bundling the per-(PP) and
                per-patient features. DataPipeline uses this.

Behavior preserved exactly from v0.3.1 (`processing/features.py` +
`processing/feature_builder.py`). The 21 existing unit tests pass
unchanged.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from pandas import Timestamp
from scipy.signal import savgol_filter
from sklearn.linear_model import TheilSenRegressor

from ai_cdss.constants import (
    ADHERENCE,
    BY_PP,
    BY_PPS,
    CLINICAL_START,
    CONTRIB,
    DAYS,
    DELTA_DM,
    DM_SMOOTH,
    DM_VALUE,
    EWMA_ALPHA,
    PATIENT_ID,
    PPF,
    PRESCRIPTION_ENDING_DATE,
    PRESCRIPTION_ID,
    PRESCRIPTION_STARTING_DATE,
    PROTOCOL_A,
    PROTOCOL_B,
    PROTOCOL_ID,
    REAL_SESSION_DURATION,
    RECENT_ADHERENCE,
    SAVGOL_POLY_ORDER,
    SAVGOL_WINDOW_SIZE,
    SESSION_COLUMNS,
    SESSION_DATE,
    SESSION_DURATION,
    SESSION_ID,
    SESSION_INDEX,
    SIMILARITY,
    STATUS,
    THEILSON_REGRESSION_WINDOW_SIZE,
    TOTAL_PRESCRIBED,
    USAGE,
    USAGE_WEEK,
    WEEKDAY_INDEX,
    WEEKS_SINCE_START,
    SessionStatus,
)


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1 — Time-axis primitives                                    ║
# ║                                                                      ║
# ║  Reduce over the time axis within a (patient, protocol) group.       ║
# ║  Used inside the per-session feature builders below.                 ║
# ╚═════════════════════════════════════════════════════════════════════╝

def compute_ewma(
    df: pd.DataFrame, value_col: str, group_cols: list[str],
    sufix: str = "", alpha: float = EWMA_ALPHA,
) -> pd.DataFrame:
    """Exponentially-weighted moving average of `value_col`, grouped.

    Returns the input DataFrame with an extra column named
    `value_col + sufix`. EWMA with `adjust=True` so the first values
    aren't biased toward zero.
    """
    return df.assign(
        **{
            f"{value_col}{sufix}": df.groupby(by=group_cols)[value_col].transform(
                lambda x: x.ewm(alpha=alpha, adjust=True).mean()
            )
        }
    )


def apply_savgol_filter_groupwise(series, window_size, polyorder):
    """Savitzky-Golay smoothing on a single group's series.

    Defensive: skips when the series is shorter than the polyorder,
    auto-adjusts window to be odd and ≥ polyorder+1, returns unsmoothed
    if savgol_filter raises.
    """
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
    """Rolling Theil-Sen regression slope over a window.

    Returns one slope per position in `series_y`. Robust to outliers
    (Theil-Sen is the median-of-pairwise-slopes estimator).
    """
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


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2 — Session-shape primitives                                ║
# ║                                                                      ║
# ║  Patient may skip a prescribed session; their session table won't    ║
# ║  carry a row for that day. `include_missing_sessions` fills the      ║
# ║  gaps with NOT_PERFORMED placeholders so downstream EWMA / adherence ║
# ║  computations see the full prescribed cadence.                       ║
# ╚═════════════════════════════════════════════════════════════════════╝

def include_missing_sessions(session: pd.DataFrame) -> pd.DataFrame:
    """For each prescription, materialize expected session rows (one
    per weekday between PRESCRIPTION_STARTING_DATE and
    PRESCRIPTION_ENDING_DATE) and merge with actual performed sessions.
    Missing rows get STATUS=NOT_PERFORMED.

    Date comparisons at day-level — time component dropped.
    """
    date_cols = [SESSION_DATE, PRESCRIPTION_STARTING_DATE, PRESCRIPTION_ENDING_DATE]
    for col in date_cols:
        session[col] = pd.to_datetime(session[col]).dt.normalize()

    valid_sessions = session.dropna(subset=[SESSION_DATE])
    last_session_per_patient = (
        valid_sessions.groupby(PATIENT_ID)[SESSION_DATE].max().to_dict()
    )

    prescriptions = session.drop_duplicates(subset=[
        PRESCRIPTION_ID, PATIENT_ID, PROTOCOL_ID,
        PRESCRIPTION_STARTING_DATE, PRESCRIPTION_ENDING_DATE, WEEKDAY_INDEX,
    ])

    expected_rows: list[dict] = []
    for _, row in prescriptions.iterrows():
        patient_id = row[PATIENT_ID]
        start = row[PRESCRIPTION_STARTING_DATE]
        end = row[PRESCRIPTION_ENDING_DATE]
        weekday = row[WEEKDAY_INDEX]
        if pd.isna(start) or pd.isna(end) or pd.isna(weekday):
            continue
        last_session = last_session_per_patient.get(
            patient_id, pd.Timestamp.today().normalize()
        )
        end_clamped = min(end, last_session)
        for session_date in generate_expected_sessions(start, end_clamped, int(weekday)):
            row_dict = {
                **row.to_dict(),
                SESSION_DATE: pd.to_datetime(session_date).normalize(),
                STATUS: SessionStatus.NOT_PERFORMED,
                ADHERENCE: 0.0,
                SESSION_DURATION: 0,
                REAL_SESSION_DURATION: 0,
            }
            row_dict.update({col: np.nan for col in SESSION_COLUMNS})
            expected_rows.append(row_dict)

    expected_df = pd.DataFrame(expected_rows)
    if expected_df.empty:
        return session

    performed_index = pd.MultiIndex.from_frame(
        valid_sessions[[PRESCRIPTION_ID, SESSION_DATE]]
    )
    expected_index = pd.MultiIndex.from_frame(
        expected_df[[PRESCRIPTION_ID, SESSION_DATE]]
    )
    expected_df = expected_df.loc[~expected_index.isin(performed_index)]

    return pd.concat([valid_sessions, expected_df], ignore_index=True).sort_values(
        by=[PATIENT_ID, PRESCRIPTION_ID, SESSION_DATE]
    ).reset_index(drop=True)


def generate_expected_sessions(
    start: Timestamp, end: Timestamp, weekday: int,
) -> list[Timestamp]:
    """Dates between `start` and `end` (inclusive) falling on `weekday`.

    Weekday: 0=Monday, 1=Tuesday, …, 6=Sunday. Returns empty list if
    `weekday` is out of range.
    """
    weekday_map = {
        0: "W-MON", 1: "W-TUE", 2: "W-WED", 3: "W-THU",
        4: "W-FRI", 5: "W-SAT", 6: "W-SUN",
    }
    freq = weekday_map.get(weekday)
    if freq is None:
        return []
    return list(pd.date_range(start=start, end=end, freq=freq))


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3 — Per-(patient, protocol, session, time) features         ║
# ║                                                                      ║
# ║  Output shape: one row per (patient, protocol, session_date).        ║
# ║  Reduces over time within a (PP) cell to produce a value per         ║
# ║  session.                                                            ║
# ╚═════════════════════════════════════════════════════════════════════╝

def build_delta_dm(session_df: pd.DataFrame) -> pd.DataFrame:
    """Trend of DM value over time, per (patient, protocol).

    Pipeline:
      1. Sort by (PP, session_date).
      2. Smooth via Savitzky-Golay (window=SAVGOL_WINDOW_SIZE, poly=SAVGOL_POLY_ORDER).
      3. Rolling Theil-Sen slope over THEILSON_REGRESSION_WINDOW_SIZE
         window → one slope per session, robust to outliers.

    Returns columns: BY_PP + [SESSION_DATE, DM_VALUE, DELTA_DM].
    """
    grouped = session_df.copy().sort_values(by=BY_PP + [SESSION_DATE])
    grouped[SESSION_INDEX] = grouped.groupby(BY_PP).cumcount() + 1
    grouped[DM_SMOOTH] = grouped.groupby(BY_PP)[DM_VALUE].transform(
        apply_savgol_filter_groupwise, SAVGOL_WINDOW_SIZE, SAVGOL_POLY_ORDER
    )
    grouped[DELTA_DM] = (
        grouped.groupby(BY_PP)[DM_SMOOTH]
        .transform(
            lambda g: get_rolling_theilsen_slope(
                g, grouped.loc[g.index, SESSION_INDEX],
                THEILSON_REGRESSION_WINDOW_SIZE,
            )
        )
        .fillna(0)
    )
    return grouped[BY_PP + [SESSION_DATE, DM_VALUE, DELTA_DM]]


def build_recent_adherence(session_df: pd.DataFrame) -> pd.DataFrame:
    """Per-session recent adherence (EWMA over ADHERENCE).

    Days where ALL sessions were NOT_PERFORMED have their ADHERENCE
    set to NaN — a fully-skipped day shouldn't anchor the EWMA at 0.

    Returns columns: BY_PPS + [SESSION_DATE, STATUS, SESSION_INDEX,
    ADHERENCE, RECENT_ADHERENCE].
    """
    df = session_df.copy()

    def day_skip_to_nan(group):
        if all(group[STATUS] == SessionStatus.NOT_PERFORMED):
            group[ADHERENCE] = np.nan
        return group

    df = df.groupby(by=[PATIENT_ID, SESSION_DATE], group_keys=False).apply(day_skip_to_nan)
    df = df.sort_values(by=BY_PP + [SESSION_DATE, WEEKDAY_INDEX])
    df[SESSION_INDEX] = (df.groupby(BY_PP).cumcount() + 1).astype("Int64")
    df = compute_ewma(df, ADHERENCE, BY_PP, sufix="_RECENT")
    return df[BY_PPS + [SESSION_DATE, STATUS, SESSION_INDEX, ADHERENCE, RECENT_ADHERENCE]]


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4 — Per-(patient, protocol) aggregations                    ║
# ║                                                                      ║
# ║  Output shape: one row per (patient, protocol).                      ║
# ║  Reduces over sessions / prescriptions to produce per-PP scalars.    ║
# ╚═════════════════════════════════════════════════════════════════════╝

def build_usage(session_df: pd.DataFrame) -> pd.DataFrame:
    """Total distinct sessions per (patient, protocol). Lifetime count."""
    df = session_df.copy()
    return (
        df.groupby([PATIENT_ID, PROTOCOL_ID], dropna=False)[SESSION_ID]
        .nunique()
        .reset_index(name=USAGE)
        .astype({USAGE: "Int64"})
    )


def build_week_usage(
    session_df: pd.DataFrame, patient_df: pd.DataFrame, scoring_date: pd.Timestamp,
) -> pd.DataFrame:
    """Distinct sessions per (patient, protocol) within the last
    *completed* patient-aligned week before scoring_date.

    Window is `[CLINICAL_START + 7*(weeks_since_start - 1),
    CLINICAL_START + 7*weeks_since_start)`. Half-open.
    """
    session_df = session_df.copy()
    session_df[SESSION_DATE] = pd.to_datetime(session_df[SESSION_DATE], errors="coerce")

    anchors = _last_completed_week_window(patient_df, scoring_date)
    df = session_df.merge(
        anchors[[PATIENT_ID, "week_start", "week_end"]],
        on=PATIENT_ID, how="inner",
    )
    in_window = (df[SESSION_DATE] >= df["week_start"]) & (df[SESSION_DATE] < df["week_end"])
    df = df.loc[in_window]

    return (
        df.groupby([PATIENT_ID, PROTOCOL_ID], dropna=False)[SESSION_ID]
        .nunique()
        .reset_index(name=USAGE_WEEK)
        .astype({USAGE_WEEK: "Int64"})
    )


def build_prescription_days(
    session_df: pd.DataFrame, patient_df: pd.DataFrame, scoring_date: Timestamp,
) -> pd.DataFrame:
    """Prescribed weekday indices per (patient, protocol) for the last
    completed week.

    The session_df is the prescription_plus LEFT JOIN session_plus
    output — WEEKDAY_INDEX reflects the **prescribed** day, not session
    performance. We filter by prescription-window OVERLAP with the last
    completed week (strict `<` on week_start, strict `>` on week_end
    boundary — touch-not-overlap excluded).
    """
    session_df = session_df.copy()
    psd_col = "PRESCRIPTION_STARTING_DATE"
    ped_col = "PRESCRIPTION_ENDING_DATE"
    session_df[psd_col] = pd.to_datetime(session_df[psd_col], errors="coerce")
    session_df[ped_col] = pd.to_datetime(session_df[ped_col], errors="coerce")

    anchors = _last_completed_week_window(patient_df, scoring_date)
    df = session_df.merge(
        anchors[[PATIENT_ID, "week_start", "week_end"]],
        on=PATIENT_ID, how="inner",
    )
    overlap = (df[psd_col] < df["week_end"]) & (df[ped_col] > df["week_start"])
    df = df.loc[overlap]

    # prescription_plus rows appear once per attached session (LEFT JOIN),
    # so dedup on the prescribed key before aggregating.
    df = df.drop_duplicates(subset=[PATIENT_ID, PROTOCOL_ID, WEEKDAY_INDEX])

    return (
        df.groupby([PATIENT_ID, PROTOCOL_ID])[WEEKDAY_INDEX]
        .agg(lambda x: sorted(x.dropna().astype(int).unique()))
        .rename(DAYS)
        .reset_index()
    )


def build_number_prescriptions(session_df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative prescription count per (PP). Currently unused by the
    pipeline but kept for backward compat — old downstream may call."""
    df = session_df.copy()
    df[TOTAL_PRESCRIBED] = df.groupby(BY_PP).cumcount() + 1
    return df


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5 — Per-patient features                                    ║
# ║                                                                      ║
# ║  Output shape: one row per patient. Reduces over everything except   ║
# ║  the patient axis.                                                   ║
# ╚═════════════════════════════════════════════════════════════════════╝

def build_week_since_start(
    patient_df: pd.DataFrame, scoring_date: pd.Timestamp,
) -> pd.DataFrame:
    """Whole weeks since each patient's CLINICAL_START as of scoring_date.

    Patient scalar (one row per patient). Reduces clinical_start +
    scoring_date to an integer week count.
    """
    df = _with_weeks_since_start(patient_df, scoring_date)
    return df[[PATIENT_ID, WEEKS_SINCE_START]]


def _with_weeks_since_start(
    patient_df: pd.DataFrame, scoring_date: pd.Timestamp,
) -> pd.DataFrame:
    """Internal: patient_df + WEEKS_SINCE_START column.

    `weeks_since_start = floor((scoring_date - clinical_start).days / 7)`.
    Normalized to midnight before subtraction so DST / time-of-day
    don't shift the count.
    """
    df = patient_df[[PATIENT_ID, CLINICAL_START]].copy()
    df[CLINICAL_START] = pd.to_datetime(df[CLINICAL_START], errors="coerce")

    scoring_day = scoring_date.normalize()
    start_day = df[CLINICAL_START].dt.normalize()
    df[WEEKS_SINCE_START] = ((scoring_day - start_day).dt.days // 7).astype("Int64")
    return df


def _last_completed_week_window(
    patient_df: pd.DataFrame, scoring_date: pd.Timestamp,
) -> pd.DataFrame:
    """Internal: for each patient with ≥1 completed week, the
    `[week_start, week_end)` of their LAST completed week.

    Returns columns: PATIENT_ID, CLINICAL_START, WEEKS_SINCE_START,
    week_start, week_end. Patients with 0 completed weeks are dropped.
    """
    df = _with_weeks_since_start(patient_df, scoring_date)
    df = df.loc[df[WEEKS_SINCE_START] > 0].copy()
    last_week_idx = (df[WEEKS_SINCE_START] - 1).astype(int)
    start_day = df[CLINICAL_START].dt.normalize()
    df["week_start"] = start_day + pd.to_timedelta(last_week_idx * 7, unit="D")
    df["week_end"] = df["week_start"] + pd.Timedelta(days=7)
    return df


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 6 — Cross-cohort features                                   ║
# ║                                                                      ║
# ║  Operate on patient × subscale (deficit) and protocol × attribute    ║
# ║  matrices. Output: (patient, protocol) pair-level features (PPF +    ║
# ║  contribution decomposition) and (protocol, protocol) similarity.    ║
# ╚═════════════════════════════════════════════════════════════════════╝

def feature_contributions(df_A: pd.DataFrame, df_B: pd.DataFrame) -> np.ndarray:
    """Element-wise contribution of each subscale to the patient×protocol
    cosine. Shape: (n_patients, n_protocols, n_subscales).

    Caller can `np.sum(..., axis=2)` to recover the cosine similarity
    or inspect per-subscale to see which dimension drives the fit.
    """
    A = df_A.to_numpy()
    B = df_B.to_numpy()
    A_norms = np.linalg.norm(A, axis=1, keepdims=True)
    B_norms = np.linalg.norm(B, axis=1, keepdims=True)
    A_norms[A_norms == 0] = 1e-10
    B_norms[B_norms == 0] = 1e-10
    A_unit = A / A_norms
    B_unit = B / B_norms
    return A_unit[:, np.newaxis, :] * B_unit[np.newaxis, :, :]


def compute_ppf(
    patient_deficiency: pd.DataFrame, protocol_mapped: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """PPF = cosine of (deficit vector, protocol-attribute vector).

    Returns `(ppf_long, contrib_long)`:
      ppf_long:     BY_PP + [PPF]      — one row per (patient, protocol)
      contrib_long: BY_PP + [CONTRIB]  — same rows, CONTRIB is a list
                                         of per-subscale contributions
    """
    contributions = feature_contributions(patient_deficiency, protocol_mapped)
    ppf = np.sum(contributions, axis=2)
    ppf_df = pd.DataFrame(
        ppf, index=patient_deficiency.index, columns=protocol_mapped.index,
    )
    contrib_df = pd.DataFrame(
        contributions.tolist(),
        index=patient_deficiency.index, columns=protocol_mapped.index,
    )

    ppf_long = ppf_df.stack().reset_index()
    ppf_long.columns = BY_PP + [PPF]

    contrib_long = contrib_df.stack().reset_index()
    contrib_long.columns = BY_PP + [CONTRIB]
    return ppf_long, contrib_long


def compute_protocol_similarity(
    protocol_mapped: pd.DataFrame,
    id_col: str = "PROTOCOL_ID",
    hot_encoded_prefix: Optional[str] = None,
) -> pd.DataFrame:
    """Pairwise protocol similarity via Gower distance (1 - distance).

    Returns long-form `[PROTOCOL_A, PROTOCOL_B, SIMILARITY]`.
    `hot_encoded_prefix` re-weights one-hot columns so they don't
    dominate the distance vs continuous dimensions.
    """
    import gower

    attributes = protocol_mapped.copy().reset_index()
    protocol_ids = attributes[id_col]
    attributes = attributes.drop(columns=[id_col])

    if hot_encoded_prefix:
        hot_mask = attributes.columns.str.startswith(hot_encoded_prefix)
        weights = np.ones(len(attributes.columns))
        if hot_mask.any():
            weights[hot_mask] /= hot_mask.sum()
    else:
        weights = np.ones(len(attributes.columns))

    try:
        attributes = attributes.astype(float)
    except Exception as e:
        raise ValueError(
            "Non-numeric columns found. Please encode categorical variables before passing."
        ) from e

    distance_matrix = gower.gower_matrix(attributes, weight=weights)
    long = (
        pd.DataFrame(1 - distance_matrix, index=protocol_ids, columns=protocol_ids)
        .stack()
        .rename_axis([PROTOCOL_A, PROTOCOL_B])
        .reset_index()
    )
    long.columns = [PROTOCOL_A, PROTOCOL_B, SIMILARITY]
    return long


# FeatureBuilder class removed. The DataPipeline now calls the
# module-level functions directly — fewer indirection layers, same
# behavior.
