"""Tests for the typed pipeline contracts.

Each dataclass in `processing/contracts.py` declares required columns.
These tests assert:
  1. Construction with the right columns succeeds.
  2. Construction with a missing column raises `ContractError`.
  3. Construction with `validate_on_init=False` skips validation.
  4. Extra columns beyond `REQUIRED` are tolerated.
"""
from __future__ import annotations

import pandas as pd
import pytest

from ai_cdss.constants import (
    BY_PP,
    CLINICAL_END,
    CLINICAL_START,
    DAYS,
    DELTA_DM,
    PATIENT_ID,
    PPF,
    PROTOCOL_ID,
    RECENT_ADHERENCE,
    SCORE,
    SESSION_DATE,
    USAGE,
    USAGE_WEEK,
    WEEKS_SINCE_START,
)
from ai_cdss.pipeline import (
    ContractError,
    MergedFeatures,
    PreparedInputs,
    ProtocolLevelFeatures,
    ScoringInput,
    ScoringOutput,
    SessionLevelFeatures,
)


# ---------------------------------------------------------------------------
# PreparedInputs

def test_prepared_inputs_accepts_minimal_columns():
    patient = pd.DataFrame({
        PATIENT_ID: [1], CLINICAL_START: [pd.Timestamp("2025-01-01")],
        CLINICAL_END: [pd.Timestamp("2025-04-01")],
    })
    session = pd.DataFrame({
        PATIENT_ID: [1], PROTOCOL_ID: [200], SESSION_DATE: [pd.Timestamp("2025-01-05")],
    })
    ppf = pd.DataFrame({PATIENT_ID: [1], PROTOCOL_ID: [200], PPF: [0.7]})
    inputs = PreparedInputs(patient=patient, session=session, ppf=ppf)
    assert inputs.has_sessions is True


def test_prepared_inputs_empty_session_marks_not_has_sessions():
    patient = pd.DataFrame({
        PATIENT_ID: [1], CLINICAL_START: [pd.Timestamp("2025-01-01")],
        CLINICAL_END: [pd.Timestamp("2025-04-01")],
    })
    session = pd.DataFrame({PATIENT_ID: [], PROTOCOL_ID: [], SESSION_DATE: []})
    ppf = pd.DataFrame({PATIENT_ID: [1], PROTOCOL_ID: [200], PPF: [0.7]})
    inputs = PreparedInputs(patient=patient, session=session, ppf=ppf)
    assert inputs.has_sessions is False


def test_prepared_inputs_missing_patient_column_raises():
    patient = pd.DataFrame({PATIENT_ID: [1]})  # missing CLINICAL_START/_END
    session = pd.DataFrame({PATIENT_ID: [1], PROTOCOL_ID: [200], SESSION_DATE: [pd.Timestamp("2025-01-01")]})
    ppf = pd.DataFrame({PATIENT_ID: [1], PROTOCOL_ID: [200], PPF: [0.7]})
    with pytest.raises(ContractError, match=r"patient.*CLINICAL"):
        PreparedInputs(patient=patient, session=session, ppf=ppf)


def test_prepared_inputs_missing_ppf_column_raises():
    patient = pd.DataFrame({
        PATIENT_ID: [1], CLINICAL_START: [pd.Timestamp("2025-01-01")],
        CLINICAL_END: [pd.Timestamp("2025-04-01")],
    })
    session = pd.DataFrame({PATIENT_ID: [1], PROTOCOL_ID: [200], SESSION_DATE: [pd.Timestamp("2025-01-05")]})
    ppf = pd.DataFrame({PATIENT_ID: [1], PROTOCOL_ID: [200]})  # missing PPF
    with pytest.raises(ContractError, match="ppf.*PPF"):
        PreparedInputs(patient=patient, session=session, ppf=ppf)


def test_prepared_inputs_validate_on_init_false_skips_check():
    """The pipeline opts out of validation internally for speed; bare
    DataFrames go through. This must not raise."""
    bogus = pd.DataFrame({"unrelated": [1]})
    inputs = PreparedInputs(
        patient=bogus, session=bogus, ppf=bogus, validate_on_init=False,
    )
    assert inputs.has_sessions is True  # non-empty unrelated frame


# ---------------------------------------------------------------------------
# SessionLevelFeatures

def test_session_level_features_requires_recent_adherence_and_delta_dm():
    df = pd.DataFrame(columns=BY_PP + [SESSION_DATE, RECENT_ADHERENCE, DELTA_DM])
    f = SessionLevelFeatures(df=df)
    assert f.df is df


def test_session_level_features_missing_delta_dm_raises():
    df = pd.DataFrame(columns=BY_PP + [SESSION_DATE, RECENT_ADHERENCE])
    with pytest.raises(ContractError, match="SessionLevelFeatures.*DELTA_DM"):
        SessionLevelFeatures(df=df)


# ---------------------------------------------------------------------------
# ProtocolLevelFeatures

def test_protocol_level_features_requires_full_protocol_set():
    df = pd.DataFrame(columns=BY_PP + [PPF, USAGE, USAGE_WEEK, DAYS, WEEKS_SINCE_START])
    f = ProtocolLevelFeatures(df=df)
    assert list(f.df.columns) == list(df.columns)


def test_protocol_level_features_missing_days_raises():
    df = pd.DataFrame(columns=BY_PP + [PPF, USAGE, USAGE_WEEK, WEEKS_SINCE_START])
    with pytest.raises(ContractError, match="ProtocolLevelFeatures.*DAYS"):
        ProtocolLevelFeatures(df=df)


# ---------------------------------------------------------------------------
# MergedFeatures, ScoringInput, ScoringOutput

def test_merged_features_requires_union_of_levels():
    cols = BY_PP + [PPF, USAGE, USAGE_WEEK, DAYS, WEEKS_SINCE_START,
                    SESSION_DATE, RECENT_ADHERENCE, DELTA_DM]
    df = pd.DataFrame(columns=cols)
    MergedFeatures(df=df)


def test_scoring_input_requires_metrics_columns():
    cols = BY_PP + [PPF, DELTA_DM, RECENT_ADHERENCE, USAGE, USAGE_WEEK,
                    DAYS, WEEKS_SINCE_START]
    df = pd.DataFrame(columns=cols)
    ScoringInput(df=df)


def test_scoring_output_requires_score_column():
    cols = BY_PP + [PPF, DELTA_DM, RECENT_ADHERENCE, USAGE, USAGE_WEEK,
                    DAYS, WEEKS_SINCE_START, SCORE]
    df = pd.DataFrame(columns=cols)
    out = ScoringOutput(df=df)
    # attrs pass-through
    df.attrs["k"] = "v"
    assert out.attrs == {"k": "v"}


def test_scoring_output_missing_score_raises():
    cols = BY_PP + [PPF, DELTA_DM, RECENT_ADHERENCE, USAGE, USAGE_WEEK,
                    DAYS, WEEKS_SINCE_START]
    df = pd.DataFrame(columns=cols)
    with pytest.raises(ContractError, match="ScoringOutput.*SCORE"):
        ScoringOutput(df=df)


# ---------------------------------------------------------------------------
# Extras are tolerated

def test_extra_columns_are_allowed():
    """REQUIRED is a floor, not a ceiling — stages may carry extra
    columns (e.g. CONTRIB, SESSION_INDEX). The contract validates
    presence, not exclusivity."""
    cols = BY_PP + [
        PPF, DELTA_DM, RECENT_ADHERENCE, USAGE, USAGE_WEEK,
        DAYS, WEEKS_SINCE_START, SCORE,
        "EXTRA_DIAGNOSTIC_COL", "ANOTHER_ONE",
    ]
    df = pd.DataFrame(columns=cols)
    ScoringOutput(df=df)
