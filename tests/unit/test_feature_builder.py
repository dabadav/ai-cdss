import datetime

import numpy as np
import pandas as pd
import pytest
from ai_cdss.processing.feature_builder import FeatureBuilder
from ai_cdss.processing.features import (
    apply_savgol_filter_groupwise,
    get_rolling_theilsen_slope,
)

# Mock constants needed for the test
PATIENT_ID = "PATIENT_ID"
PROTOCOL_ID = "PROTOCOL_ID"
SESSION_ID = "SESSION_ID"
STATUS = "STATUS"
SESSION_DATE = "SESSION_DATE"
DM_VALUE = "DM_VALUE"
BY_PP = [PATIENT_ID, PROTOCOL_ID]

# ---------------------------------------------------------------
# Delta DM
# ---------------------------------------------------------------


def test_build_delta_dm_basic(monkeypatch):
    # Prepare minimal input DataFrame
    df = pd.DataFrame(
        {
            PATIENT_ID: [1, 1, 1, 2, 2],
            PROTOCOL_ID: [10, 10, 10, 20, 20],
            SESSION_DATE: pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-01", "2024-01-02"]
            ),
            DM_VALUE: [0.5, 0.7, 0.8, 1.0, 0.9],
        }
    )
    print("Input DataFrame:")
    print(df)

    # Patch the feature_builder module's constants to use our test values
    monkeypatch.setattr("ai_cdss.processing.feature_builder.BY_PP", BY_PP)
    monkeypatch.setattr("ai_cdss.processing.feature_builder.SESSION_DATE", SESSION_DATE)
    monkeypatch.setattr("ai_cdss.processing.feature_builder.DM_VALUE", DM_VALUE)
    # monkeypatch.setattr("ai_cdss.processing.feature_builder.SAVGOL_WINDOW_SIZE", 3)
    # monkeypatch.setattr("ai_cdss.processing.feature_builder.SAVGOL_POLY_ORDER", 1)
    # monkeypatch.setattr(
    #     "ai_cdss.processing.feature_builder.THEILSON_REGRESSION_WINDOW_SIZE", 2
    # )
    monkeypatch.setattr("ai_cdss.processing.feature_builder.DELTA_DM", "DELTA_DM")

    fb = FeatureBuilder()
    result = fb.build_delta_dm(df)

    print("Output DataFrame:")
    print(result)

    # Check output columns
    expected_columns = BY_PP + [SESSION_DATE, DM_VALUE, "DELTA_DM"]
    assert list(result.columns) == expected_columns
    # Check output shape
    assert result.shape[0] == df.shape[0]


def test_apply_savgol_filter_groupwise_dm_like(logger=None):
    # DM-like data: a trend with some noise
    x = pd.Series([0.5, 0.7, 0.8])
    window = 7
    poly = 2
    result = apply_savgol_filter_groupwise(x, window, poly)
    log = logger.info if logger else print
    log("DM-like Input Series:")
    log(list(x.values), "\n")
    log("Savitzky-Golay Output (DM-like):")
    log(list(result), "\n")
    # The output should be smoother than the input (less noisy)
    assert len(result) == len(x)
    # Check that the mean is preserved approximately
    assert abs(result.mean() - x.mean()) < 0.1


def test_get_rolling_theilsen_slope_dm_like(logger=None):
    # DM-like data: a trend with some noise
    y = pd.Series([0.5, 0.7, 0.8])
    x_idx = pd.Series(range(1, 4))
    window = 3
    result = get_rolling_theilsen_slope(y, x_idx, window)
    log = logger.info if logger else print
    log("\nDM-like Input y:", list(y.values))
    log("DM-like Input x:", list(x_idx.values))
    log("Theil-Sen rolling slope (DM-like):", list(result))
    # The slope should be positive in the center (since the trend is increasing)
    assert len(result) == len(y)


# ---------------------------------------------------------------
# Adherence
# ---------------------------------------------------------------


@pytest.fixture
def patient_df():
    return pd.DataFrame(
        {
            "PATIENT_ID": [1, 1, 2],
            "CLINICAL_TRIAL_START_DATE": [datetime.datetime(2024, 1, 1)] * 3,
            "CLINICAL_TRIAL_END_DATE": [datetime.datetime(2024, 1, 31)] * 3,
        }
    )


# Also for scoring date
@pytest.fixture
def session_df():
    return pd.DataFrame(
        {
            "PATIENT_ID": [1, 1, 2],
            "PRESCRIPTION_ID": [101, 102, 201],
            "SESSION_ID": [1001, np.nan, 2001],
            "PROTOCOL_ID": [10, 10, 20],
            "PRESCRIPTION_STARTING_DATE": [
                datetime.datetime(2024, 1, 1),
                datetime.datetime(2024, 1, 8),
                datetime.datetime(2024, 1, 1),
            ],
            "PRESCRIPTION_ENDING_DATE": [
                datetime.datetime(2024, 1, 8),
                datetime.datetime(2024, 1, 15),
                datetime.datetime(2024, 1, 8),
            ],
            "SESSION_DATE": [
                datetime.datetime(2024, 1, 2),
                datetime.datetime(2024, 1, 3),
                datetime.datetime(2024, 1, 2),
            ],
            "WEEKDAY_INDEX": [1, 2, 1],
            "STATUS": ["CLOSED", "NOT_PERFORMED", "CLOSED"],
            "REAL_SESSION_DURATION": [30, 0, 45],
            "PRESCRIBED_SESSION_DURATION": [30, 30, 45],
            "SESSION_DURATION": [30, 0, 45],
            "ADHERENCE": [1.0, 0.0, 1.0],
        }
    )


# ---------------------------------------------------------------
# Usage
# ---------------------------------------------------------------


def test_build_usage_with_fixture(session_df, logger=None):
    print("\n--- Testing build_usage ---")
    fb = FeatureBuilder()
    log = logger.info if logger else print
    log("Input DataFrame:")
    with pd.option_context("display.max_columns", None, "display.width", 1000):
        print(session_df)
    result = fb.build_usage(session_df)
    log("Output DataFrame:")
    with pd.option_context("display.max_columns", None, "display.width", 1000):
        print(result)
    assert {"PATIENT_ID", "PROTOCOL_ID", "USAGE"}.issubset(result.columns)
    assert result.groupby(["PATIENT_ID", "PROTOCOL_ID"]).size().min() == 1


# ---------------------------------------------------------------
# Usage week
# ---------------------------------------------------------------


def test_build_week_usage_with_fixture(session_df, patient_df, logger=None):
    print("\n--- Testing build_week_usage ---")
    fb = FeatureBuilder()
    log = logger.info if logger else print
    scoring_date = pd.Timestamp("2024-01-08")
    log("Input DataFrame:")
    with pd.option_context("display.max_columns", None, "display.width", 1000):
        print(session_df)
    result = fb.build_week_usage(session_df, patient_df, scoring_date)
    log("Output DataFrame:")
    with pd.option_context("display.max_columns", None, "display.width", 1000):
        print(result)
    assert {"PATIENT_ID", "PROTOCOL_ID", "USAGE_WEEK"}.issubset(result.columns)
    assert result["USAGE_WEEK"].min() >= 0


# ---------------------------------------------------------------
# Weeks since start
# ---------------------------------------------------------------


def test_build_week_since_start_with_fixture(patient_df, logger=None):
    print("\n--- Testing build_week_since_start ---")
    fb = FeatureBuilder()
    log = logger.info if logger else print
    log("Input DataFrame:")
    with pd.option_context("display.max_columns", None, "display.width", 1000):
        print(patient_df)
    result = fb.build_week_since_start(
        patient_df, scoring_date=pd.Timestamp("2024-01-08")
    )
    log("Output DataFrame:")
    with pd.option_context("display.max_columns", None, "display.width", 1000):
        print(result)
    assert {"PATIENT_ID", "WEEKS_SINCE_START"}.issubset(result.columns)


# ---------------------------------------------------------------
# Prescription days
# ---------------------------------------------------------------


def test_build_prescription_days_with_fixture(session_df, patient_df, logger=None):
    print("\n--- Testing build_prescription_days ---")
    fb = FeatureBuilder()
    log = logger.info if logger else print
    scoring_date = pd.Timestamp("2024-01-08")
    log("Input DataFrame:")
    with pd.option_context("display.max_columns", None, "display.width", 1000):
        print(session_df)
    result = fb.build_prescription_days(session_df, patient_df, scoring_date)
    log("Output DataFrame:")
    with pd.option_context("display.max_columns", None, "display.width", 1000):
        print(result)
    assert {"PATIENT_ID", "PROTOCOL_ID", "DAYS"}.issubset(result.columns)
    assert result["DAYS"].apply(lambda x: isinstance(x, list)).all()


def test_build_prescription_days_uses_prescribed_not_performed():
    """DAYS must reflect *prescribed* weekdays in the last completed week,
    not weekdays the patient actually performed sessions on. Patient
    compliance must not collapse next week's coverage."""
    fb = FeatureBuilder()
    patient_df = pd.DataFrame({
        "PATIENT_ID": [1],
        "CLINICAL_TRIAL_START_DATE": [datetime.datetime(2024, 1, 1)],
        "CLINICAL_TRIAL_END_DATE":   [datetime.datetime(2024, 1, 31)],
    })
    # Patient prescribed for Mon (0), Tue (1), Wed (2), Thu (3), Fri (4)
    # in the [2024-01-01, 2024-01-08) window. Sessions only happened on
    # Tuesday (1). Pre-fix this would yield DAYS=[1]; post-fix it must
    # be [0, 1, 2, 3, 4] because all 5 weekdays were *prescribed*.
    session_df = pd.DataFrame({
        "PATIENT_ID":                 [1, 1, 1, 1, 1],
        "PRESCRIPTION_ID":            [10, 11, 12, 13, 14],
        "SESSION_ID":                 [np.nan, 1001, np.nan, np.nan, np.nan],
        "PROTOCOL_ID":                [200, 200, 200, 200, 200],
        "PRESCRIPTION_STARTING_DATE": [datetime.datetime(2024, 1, 1)] * 5,
        "PRESCRIPTION_ENDING_DATE":   [datetime.datetime(2024, 1, 8)] * 5,
        "SESSION_DATE":               [pd.NaT, datetime.datetime(2024, 1, 2),
                                        pd.NaT, pd.NaT, pd.NaT],
        "WEEKDAY_INDEX":              [0, 1, 2, 3, 4],
    })
    result = fb.build_prescription_days(
        session_df, patient_df, scoring_date=pd.Timestamp("2024-01-08")
    )
    assert len(result) == 1
    days = result.iloc[0]["DAYS"]
    assert days == [0, 1, 2, 3, 4], f"DAYS should follow prescribed weekdays, got {days}"


def test_build_prescription_days_filters_to_last_completed_week():
    """Prescriptions outside the last completed week window are excluded.
    A prescription whose window ends before week_start, or starts after
    week_end, must not contribute weekdays."""
    fb = FeatureBuilder()
    patient_df = pd.DataFrame({
        "PATIENT_ID": [1],
        "CLINICAL_TRIAL_START_DATE": [datetime.datetime(2024, 1, 1)],
        "CLINICAL_TRIAL_END_DATE":   [datetime.datetime(2024, 1, 31)],
    })
    # Last completed week (scoring 2024-01-08) is [2024-01-01, 2024-01-08).
    session_df = pd.DataFrame({
        "PATIENT_ID":                 [1, 1, 1],
        "PRESCRIPTION_ID":            [1, 2, 3],
        "SESSION_ID":                 [np.nan, np.nan, np.nan],
        "PROTOCOL_ID":                [200, 201, 202],
        # Prescription 1: ends before window  -> excluded
        # Prescription 2: overlaps window     -> included
        # Prescription 3: starts after window -> excluded
        "PRESCRIPTION_STARTING_DATE": [datetime.datetime(2023, 12, 18),
                                        datetime.datetime(2024, 1, 1),
                                        datetime.datetime(2024, 1, 9)],
        "PRESCRIPTION_ENDING_DATE":   [datetime.datetime(2023, 12, 25),
                                        datetime.datetime(2024, 1, 8),
                                        datetime.datetime(2024, 1, 16)],
        "SESSION_DATE":               [pd.NaT, pd.NaT, pd.NaT],
        "WEEKDAY_INDEX":              [0, 3, 5],
    })
    result = fb.build_prescription_days(
        session_df, patient_df, scoring_date=pd.Timestamp("2024-01-08")
    )
    assert set(result["PROTOCOL_ID"]) == {201}
    assert result.iloc[0]["DAYS"] == [3]


def test_build_prescription_days_excludes_prior_week_ending_at_week_start():
    """A prescription whose ENDING_DATE equals this week's week_start must be
    excluded — both intervals are half-open, so a prior-week prescription
    that ends exactly when the current week begins does not overlap. This
    is the regression for patient 4899 wk5, where the prior-prior week's
    rows leaked into DAYS via a `>=` boundary check."""
    fb = FeatureBuilder()
    patient_df = pd.DataFrame({
        "PATIENT_ID": [1],
        "CLINICAL_TRIAL_START_DATE": [datetime.datetime(2024, 1, 1)],
        "CLINICAL_TRIAL_END_DATE":   [datetime.datetime(2024, 1, 31)],
    })
    # scoring_date 2024-01-15 -> last completed week [2024-01-08, 2024-01-15).
    # Two prescriptions:
    #   - 200: [2024-01-01, 2024-01-08) — prior-prior week, ENDING_DATE
    #          equals week_start. Must be excluded.
    #   - 201: [2024-01-08, 2024-01-15) — this is the actual last completed
    #          week. Must be included.
    session_df = pd.DataFrame({
        "PATIENT_ID":                 [1, 1],
        "PRESCRIPTION_ID":            [1, 2],
        "SESSION_ID":                 [np.nan, np.nan],
        "PROTOCOL_ID":                [200, 201],
        "PRESCRIPTION_STARTING_DATE": [datetime.datetime(2024, 1, 1),
                                        datetime.datetime(2024, 1, 8)],
        "PRESCRIPTION_ENDING_DATE":   [datetime.datetime(2024, 1, 8),
                                        datetime.datetime(2024, 1, 15)],
        "SESSION_DATE":               [pd.NaT, pd.NaT],
        "WEEKDAY_INDEX":              [0, 2],
    })
    result = fb.build_prescription_days(
        session_df, patient_df, scoring_date=pd.Timestamp("2024-01-15")
    )
    assert set(result["PROTOCOL_ID"]) == {201}, (
        f"prior-prior week prescription leaked into DAYS: {result.to_dict('records')}"
    )
    assert result.iloc[0]["DAYS"] == [2]


def test_build_prescription_days_dedups_multiple_sessions_per_prescription():
    """A prescription with multiple session rows (LEFT JOIN duplicates)
    must contribute its weekday only once."""
    fb = FeatureBuilder()
    patient_df = pd.DataFrame({
        "PATIENT_ID": [1],
        "CLINICAL_TRIAL_START_DATE": [datetime.datetime(2024, 1, 1)],
        "CLINICAL_TRIAL_END_DATE":   [datetime.datetime(2024, 1, 31)],
    })
    # Same prescription appears 3 times because 3 sessions attached.
    session_df = pd.DataFrame({
        "PATIENT_ID":                 [1, 1, 1],
        "PRESCRIPTION_ID":            [10, 10, 10],
        "SESSION_ID":                 [1001, 1002, 1003],
        "PROTOCOL_ID":                [200, 200, 200],
        "PRESCRIPTION_STARTING_DATE": [datetime.datetime(2024, 1, 1)] * 3,
        "PRESCRIPTION_ENDING_DATE":   [datetime.datetime(2024, 1, 8)] * 3,
        "SESSION_DATE":               [datetime.datetime(2024, 1, 2),
                                        datetime.datetime(2024, 1, 3),
                                        datetime.datetime(2024, 1, 4)],
        "WEEKDAY_INDEX":              [1, 1, 1],
    })
    result = fb.build_prescription_days(
        session_df, patient_df, scoring_date=pd.Timestamp("2024-01-08")
    )
    assert len(result) == 1
    assert result.iloc[0]["DAYS"] == [1]


# ---------------------------------------------------------------
