"""Tests for the substrate-agnostic engine API (phase F2).

Proves that the recommendation engine accepts:
  1. A plain `pd.DataFrame` (production path) — already covered by
     test_cdss_recommend.py + test_recommendation_result.py.
  2. A `DictBackedState` with a `DictSimilarity` — pandas-free
     synthetic path. This is the "10-line synthetic backtest" goal of
     the functionality refactor.

Also verifies the boundary coercers (`coerce_engine_state`,
`coerce_similarity`) wrap raw inputs correctly.
"""
from __future__ import annotations

import pandas as pd
import pytest

from ai_cdss.engine import (
    DataFrameBackedState,
    DataFrameSimilarity,
    DictBackedState,
    DictSimilarity,
    ProtocolRow,
    coerce_engine_state,
    coerce_similarity,
)
from ai_cdss.recommend import CDSS, RecommendationResult


# ---------------------------------------------------------------------------
# DictBackedState — basic protocol satisfaction.

def _synthetic_state(
    patient_id: int = 1, prescribed_subset: list[int] | None = None,
) -> DictBackedState:
    """Build a 12-protocol synthetic state. By default no protocol is
    prescribed (bootstrap branch). Pass `prescribed_subset` to mark
    those protocols as having DAYS — engine then takes update branch."""
    prescribed = set(prescribed_subset or [])
    rows = {}
    for i, pid in enumerate(range(200, 212)):
        days = [i % 7, (i + 2) % 7, (i + 4) % 7] if pid in prescribed else []
        rows[pid] = ProtocolRow(
            patient_id=patient_id,
            protocol_id=pid,
            score=1.0 + 0.05 * i,
            days=days,
            usage=i,
            usage_week=i % 3 if pid in prescribed else 0,
            ppf=0.5 + 0.02 * i,
            recent_adherence=0.7,
            delta_dm=0.0,
        )
    return DictBackedState(patient_id=patient_id, rows=rows)


def _synthetic_similarity(min_id: int = 200, max_id: int = 211) -> DictSimilarity:
    """All-pairs similarity for the synthetic protocol range."""
    pairs = {}
    for a in range(min_id, max_id + 1):
        for b in range(min_id, max_id + 1):
            if a == b:
                continue
            pairs[(a, b)] = 0.5 + 0.01 * (b - a)
    return DictSimilarity(pairs)


def test_dict_backed_state_satisfies_engine_state_protocol():
    state = _synthetic_state()
    # Required attributes / methods.
    assert state.has_data is True
    assert isinstance(state.all_protocols, list)
    assert isinstance(state.prescribed_rows, list)
    assert state.is_week_skipped() is False
    assert isinstance(state.top_protocols(3), list)
    assert isinstance(state.protocols_with_zero_usage, list)
    row = state.score_row(200)
    assert isinstance(row, ProtocolRow)
    assert state.usage_of(200) == 0
    assert state.usage_of(99999) == 0  # unknown → 0


def test_dict_backed_state_top_protocols_sorted_by_score_desc():
    state = _synthetic_state()
    top3 = state.top_protocols(3)
    # Scores are 1.0 + 0.05*i; highest = i=11 (pid 211)
    assert top3 == [211, 210, 209]


def test_dict_backed_state_prescribed_rows_filters_empty_days():
    state = _synthetic_state(prescribed_subset=[200, 205, 209])
    prescribed_ids = {r.protocol_id for r in state.prescribed_rows}
    assert prescribed_ids == {200, 205, 209}


def test_dict_backed_state_is_week_skipped_when_all_usage_week_zero():
    """USAGE_WEEK=0 on every prescribed row → engine treats as skipped."""
    rows = {
        200: ProtocolRow(patient_id=1, protocol_id=200, score=1.0,
                         days=[0, 2, 4], usage_week=0),
        201: ProtocolRow(patient_id=1, protocol_id=201, score=1.1,
                         days=[1, 3, 5], usage_week=0),
    }
    state = DictBackedState(patient_id=1, rows=rows)
    assert state.is_week_skipped() is True


def test_dict_backed_state_lowest_scoring_prescribed():
    state = _synthetic_state(prescribed_subset=[200, 211, 205])
    # Among prescribed (200, 205, 211), pid 200 has the lowest score.
    assert state.lowest_scoring_prescribed == 200


def test_dict_backed_state_protocols_with_zero_usage():
    """Only pid 200 has usage=0 (i=0); others have usage=i."""
    state = _synthetic_state()
    assert state.protocols_with_zero_usage == [200]


def test_dict_backed_state_with_prescribed_set_clones_with_days():
    """`with_prescribed_set` returns a fresh state with DAYS overrides
    — used for chained-mode backtest injection."""
    state = _synthetic_state()
    chained = state.with_prescribed_set({200: [0, 2, 4], 201: [1, 3, 5]})
    # Original unchanged.
    assert state.score_row(200).days == []
    # New state has DAYS applied.
    assert chained.score_row(200).days == [0, 2, 4]
    assert chained.score_row(201).days == [1, 3, 5]
    # Protocols not in the override get empty DAYS.
    assert chained.score_row(205).days == []


# ---------------------------------------------------------------------------
# End-to-end: CDSS.recommend with DictBackedState + DictSimilarity
# — no pandas inside the engine.

def test_recommend_with_dict_state_bootstrap_branch():
    state = _synthetic_state()
    similarity = _synthetic_similarity()
    cdss = CDSS(scoring=state, n=12)
    result = cdss.recommend(state.patient_id, similarity)
    assert isinstance(result, RecommendationResult)
    assert result.branch == "bootstrap"
    assert len(result.final_protocols) > 0


def test_recommend_with_dict_state_update_branch():
    state = _synthetic_state(prescribed_subset=list(range(200, 212)))
    similarity = _synthetic_similarity()
    cdss = CDSS(scoring=state, n=12)
    result = cdss.recommend(state.patient_id, similarity)
    assert result.branch == "update"
    assert isinstance(result.mvt_mean, float)


def test_recommend_with_dict_state_repeat_branch():
    """All prescribed rows with USAGE_WEEK=0 → repeat branch fires."""
    rows = {
        pid: ProtocolRow(
            patient_id=1, protocol_id=pid,
            score=1.0 + 0.05 * i,
            days=[i % 7, (i + 2) % 7, (i + 4) % 7],
            usage_week=0,
        )
        for i, pid in enumerate(range(200, 212))
    }
    state = DictBackedState(patient_id=1, rows=rows)
    similarity = _synthetic_similarity()
    cdss = CDSS(scoring=state, n=12)
    result = cdss.recommend(1, similarity)
    assert result.branch == "repeat_skipped_week"


def test_recommend_with_dict_state_returns_dataframe_output():
    """Even though the input is pandas-free, the output `recommendations`
    is still a pd.DataFrame (boundary materialization)."""
    state = _synthetic_state()
    similarity = _synthetic_similarity()
    cdss = CDSS(scoring=state, n=12)
    result = cdss.recommend(state.patient_id, similarity)
    assert isinstance(result.recommendations, pd.DataFrame)


def test_recommend_dataframe_input_still_works():
    """Backward compat: pd.DataFrame scoring still accepted."""
    from ai_cdss.constants import (
        DAYS, PATIENT_ID, PROTOCOL_ID, PPF, SCORE, USAGE, USAGE_WEEK,
    )
    scoring = pd.DataFrame([
        {
            PATIENT_ID: 1, PROTOCOL_ID: pid,
            SCORE: 1.0 + 0.05 * i,
            USAGE: i, USAGE_WEEK: 0,
            DAYS: [],
            PPF: 0.5 + 0.02 * i,
        }
        for i, pid in enumerate(range(200, 212))
    ])
    sim_df = pd.DataFrame([
        {"PROTOCOL_A": a, "PROTOCOL_B": b, "SIMILARITY": 0.5}
        for a in range(200, 212) for b in range(200, 212) if a != b
    ])
    cdss = CDSS(scoring=scoring, n=12)
    result = cdss.recommend(1, sim_df)
    assert isinstance(result, RecommendationResult)
    assert result.branch == "bootstrap"


def test_dict_similarity_for_protocol_excludes_correctly():
    sim = DictSimilarity({
        (200, 201): 0.8, (200, 202): 0.7, (200, 203): 0.6,
    })
    pairs = sim.similarities_for(200, exclude=[202])
    pair_dict = dict(pairs)
    assert 201 in pair_dict
    assert 203 in pair_dict
    assert 202 not in pair_dict


def test_dict_similarity_top_n_ordering():
    sim = DictSimilarity({
        (200, 201): 0.5, (200, 202): 0.9, (200, 203): 0.7,
    })
    top2 = sim.top_n_similar(200, 2)
    assert top2 == [202, 203]  # highest similarity first


# ---------------------------------------------------------------------------
# Coercion helpers at the boundary.

def test_coerce_engine_state_passes_through_engine_state():
    state = _synthetic_state()
    assert coerce_engine_state(state) is state


def test_coerce_engine_state_wraps_dataframe():
    from ai_cdss.constants import PATIENT_ID, PROTOCOL_ID, SCORE, DAYS, USAGE, USAGE_WEEK
    df = pd.DataFrame({
        PATIENT_ID: [1], PROTOCOL_ID: [200], SCORE: [1.0],
        DAYS: [[]], USAGE: [0], USAGE_WEEK: [0],
    })
    state = coerce_engine_state(df, patient_id=1)
    assert isinstance(state, DataFrameBackedState)
    assert state.patient_id == 1


def test_coerce_engine_state_requires_patient_id_for_dataframe():
    df = pd.DataFrame({"PATIENT_ID": [1], "PROTOCOL_ID": [200]})
    with pytest.raises(ValueError, match="patient_id"):
        coerce_engine_state(df)  # missing patient_id


def test_coerce_similarity_passes_through():
    sim = DictSimilarity({(200, 201): 0.8})
    assert coerce_similarity(sim) is sim


def test_coerce_similarity_wraps_dataframe():
    df = pd.DataFrame({"PROTOCOL_A": [200], "PROTOCOL_B": [201], "SIMILARITY": [0.8]})
    assert isinstance(coerce_similarity(df), DataFrameSimilarity)


def test_coerce_similarity_wraps_dict():
    assert isinstance(coerce_similarity({(200, 201): 0.8}), DictSimilarity)


def test_coerce_similarity_rejects_unknown_type():
    with pytest.raises(TypeError, match="SimilarityMatrix"):
        coerce_similarity(42)


# ---------------------------------------------------------------------------
# The "10-line synthetic backtest" — proves the goal is met.

def test_synthetic_backtest_in_ten_lines():
    """Demonstration: synthetic recommendation, no pandas imports
    needed by the caller (engine handles materialization at output)."""
    state = DictBackedState.from_rows(patient_id=4378, rows=[
        ProtocolRow(patient_id=4378, protocol_id=200, score=1.8, ppf=0.7),
        ProtocolRow(patient_id=4378, protocol_id=201, score=1.7, ppf=0.6),
        ProtocolRow(patient_id=4378, protocol_id=202, score=1.6, ppf=0.6),
        ProtocolRow(patient_id=4378, protocol_id=203, score=1.5, ppf=0.5),
    ])
    sim = DictSimilarity({
        (200, 201): 0.8, (200, 202): 0.7, (200, 203): 0.6,
        (201, 200): 0.8, (201, 202): 0.5, (201, 203): 0.4,
        (202, 200): 0.7, (202, 201): 0.5, (202, 203): 0.9,
        (203, 200): 0.6, (203, 201): 0.4, (203, 202): 0.9,
    })
    result = CDSS(scoring=state, n=4, days=7, protocols_per_day=2).recommend(
        4378, sim,
    )
    assert result.branch == "bootstrap"
    assert isinstance(result, RecommendationResult)
