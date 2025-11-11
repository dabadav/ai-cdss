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


# ---------------------------------------------------------------
