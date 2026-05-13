"""Tests for `RecommendationResult` (phase F1).

Verifies the PCA/sklearn-style introspectable result object exposes
every intermediate artifact as a public attribute, plus preserves
backward-compat DataFrame-like access patterns.
"""
from __future__ import annotations

import pandas as pd
import pytest

from ai_cdss.constants import DAYS, PATIENT_ID, PROTOCOL_ID, SCORE, USAGE, USAGE_WEEK, PPF
from ai_cdss.recommend import CDSS, PatientState, RecommendationResult, SubstituteResult


# ---------------------------------------------------------------------------
# Fixtures — minimal scoring frames for each branch.

def _make_scoring(patient_id: int = 1, prescriptions_have_days: bool = False) -> pd.DataFrame:
    """Build a minimal scoring frame with 12 protocols for one patient.

    `prescriptions_have_days=True` ⇒ rows have non-empty DAYS (engine
    takes update branch). Default = all empty DAYS (bootstrap).
    """
    rows = []
    for i, pid in enumerate(range(200, 212)):
        if prescriptions_have_days:
            days = [i % 7, (i + 2) % 7, (i + 4) % 7]
        else:
            days = []
        rows.append({
            PATIENT_ID:  patient_id,
            PROTOCOL_ID: pid,
            SCORE:       1.0 + 0.05 * i,
            USAGE:       i,
            USAGE_WEEK:  i % 3,
            DAYS:        days,
            PPF:         0.5 + 0.02 * i,
        })
    df = pd.DataFrame(rows)
    df.attrs = {"SUBSCALES": ["motor", "cognitive"]}
    return df


def _make_similarity(min_id: int = 200, max_id: int = 211) -> pd.DataFrame:
    """All-pairs similarity table for protocols [min_id..max_id]."""
    rows = []
    for a in range(min_id, max_id + 1):
        for b in range(min_id, max_id + 1):
            if a == b:
                continue
            rows.append({
                "PROTOCOL_A": a,
                "PROTOCOL_B": b,
                "SIMILARITY": 0.5 + 0.01 * (b - a),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bootstrap branch result.

def test_result_returned_from_recommend_is_recommendation_result():
    cdss = CDSS(scoring=_make_scoring())
    result = cdss.recommend(1, _make_similarity())
    assert isinstance(result, RecommendationResult)


def test_result_bootstrap_branch_label():
    cdss = CDSS(scoring=_make_scoring())
    result = cdss.recommend(1, _make_similarity())
    assert result.branch == "bootstrap"


def test_result_bootstrap_has_no_mvt_mean():
    cdss = CDSS(scoring=_make_scoring())
    result = cdss.recommend(1, _make_similarity())
    assert result.mvt_mean is None


def test_result_bootstrap_has_no_swap_decisions():
    cdss = CDSS(scoring=_make_scoring())
    result = cdss.recommend(1, _make_similarity())
    assert result.swap_decisions == []
    assert result.swap_targets == []
    assert result.swap_reasons == {}
    assert result.n_swaps == 0


def test_result_recommendations_is_dataframe():
    cdss = CDSS(scoring=_make_scoring())
    result = cdss.recommend(1, _make_similarity())
    assert isinstance(result.recommendations, pd.DataFrame)
    assert len(result.recommendations) > 0


def test_result_final_protocols_property():
    cdss = CDSS(scoring=_make_scoring())
    result = cdss.recommend(1, _make_similarity())
    final = result.final_protocols
    assert isinstance(final, list)
    assert all(isinstance(p, int) for p in final)
    assert final == sorted(final)


def test_result_trace_matches_attrs():
    """The structured trace is accessible both as `.trace` and via the
    legacy `.attrs["trace"]` pathway."""
    cdss = CDSS(scoring=_make_scoring())
    result = cdss.recommend(1, _make_similarity())
    assert result.trace is result.recommendations.attrs["trace"]
    assert result.attrs["trace"] is result.trace


def test_result_patient_state_is_patientstate():
    cdss = CDSS(scoring=_make_scoring())
    result = cdss.recommend(1, _make_similarity())
    assert isinstance(result.patient_state, PatientState)
    assert result.patient_state.patient_id == 1


# ---------------------------------------------------------------------------
# Update branch result — swap decisions populated.

def test_result_update_branch_label():
    cdss = CDSS(scoring=_make_scoring(prescriptions_have_days=True))
    result = cdss.recommend(1, _make_similarity())
    assert result.branch == "update"


def test_result_update_branch_has_mvt_mean():
    cdss = CDSS(scoring=_make_scoring(prescriptions_have_days=True))
    result = cdss.recommend(1, _make_similarity())
    assert result.mvt_mean is not None
    assert isinstance(result.mvt_mean, float)


def test_result_update_branch_swap_decisions_typed():
    cdss = CDSS(scoring=_make_scoring(prescriptions_have_days=True))
    result = cdss.recommend(1, _make_similarity())
    assert len(result.swap_decisions) > 0
    for swap in result.swap_decisions:
        assert isinstance(swap, SubstituteResult)
        assert swap.removed_id is not None
        assert swap.reason in (
            "below_mean_score", "aisn_min_one_swap", "unknown",
        )


def test_result_candidate_pool_for_returns_pool():
    cdss = CDSS(scoring=_make_scoring(prescriptions_have_days=True))
    result = cdss.recommend(1, _make_similarity())
    if result.swap_decisions:
        first = result.swap_decisions[0]
        pool = result.candidate_pool_for(first.removed_id)
        assert pool == first.candidates


def test_result_candidate_pool_for_unknown_returns_empty():
    cdss = CDSS(scoring=_make_scoring(prescriptions_have_days=True))
    result = cdss.recommend(1, _make_similarity())
    assert result.candidate_pool_for(99999) == []


# ---------------------------------------------------------------------------
# Backward-compat: subscriptable / iterable / len.

def test_result_subscript_proxies_to_recommendations():
    """`result[col]` reads the underlying recommendations DataFrame."""
    cdss = CDSS(scoring=_make_scoring())
    result = cdss.recommend(1, _make_similarity())
    protocol_col = result[PROTOCOL_ID]
    assert isinstance(protocol_col, pd.Series)


def test_result_iter_proxies_to_recommendations():
    cdss = CDSS(scoring=_make_scoring())
    result = cdss.recommend(1, _make_similarity())
    columns = list(iter(result))
    assert PROTOCOL_ID in columns


def test_result_len_proxies_to_recommendations():
    cdss = CDSS(scoring=_make_scoring())
    result = cdss.recommend(1, _make_similarity())
    assert len(result) == len(result.recommendations)


def test_result_to_dataframe_explicit_unwrap():
    cdss = CDSS(scoring=_make_scoring())
    result = cdss.recommend(1, _make_similarity())
    assert result.to_dataframe() is result.recommendations


# ---------------------------------------------------------------------------
# Topup events surface as both a list and via the trace.

def test_result_topup_events_match_trace_topup():
    cdss = CDSS(scoring=_make_scoring(prescriptions_have_days=True))
    result = cdss.recommend(1, _make_similarity())
    assert result.topup_events == (result.trace.get("topup") or [])


def test_result_n_topup_matches_event_count():
    cdss = CDSS(scoring=_make_scoring(prescriptions_have_days=True))
    result = cdss.recommend(1, _make_similarity())
    assert result.n_topup == len(result.topup_events)


# ---------------------------------------------------------------------------
# Scoring attrs propagated.

def test_result_scoring_attrs_carries_subscales():
    cdss = CDSS(scoring=_make_scoring())
    result = cdss.recommend(1, _make_similarity())
    assert result.scoring_attrs.get("SUBSCALES") == ["motor", "cognitive"]
