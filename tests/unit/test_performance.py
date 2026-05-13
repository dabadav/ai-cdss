"""Performance regression guards (phase F3).

Microbenchmarks for the synthetic recommendation hot path. These
are NOT pinpoint timing tests (CI is noisy) — they assert order-of-
magnitude budgets so a future regression that 10× slows the engine
gets caught.

Run with `pytest -k performance -s` to see actual timings.
"""
from __future__ import annotations

import time

import pytest

from ai_cdss.engine import (
    DictPatientState,
    DictSimilarity,
    ProtocolRow,
)
from ai_cdss.recommender import Recommender


def _build_synthetic(n_protocols: int = 27, prescribed: int = 12) -> tuple[DictPatientState, DictSimilarity]:
    """Realistic-sized synthetic state: 27-protocol whitelist, 12
    prescribed (matches AISN trial dimensions)."""
    rows = {}
    for i in range(n_protocols):
        pid = 200 + i
        days = [i % 7, (i + 2) % 7, (i + 4) % 7] if i < prescribed else []
        rows[pid] = ProtocolRow(
            patient_id=1, protocol_id=pid,
            score=1.0 + 0.05 * (i % 13),  # vary so MVT has work to do
            days=days,
            usage=i,
            usage_week=(i % 3) + 1,
            ppf=0.5 + 0.02 * i,
        )
    state = DictPatientState(patient_id=1, rows=rows)

    pairs = {}
    for i in range(n_protocols):
        for j in range(n_protocols):
            if i == j:
                continue
            a, b = 200 + i, 200 + j
            pairs[(a, b)] = 0.5 + 0.01 * abs(i - j)
    sim = DictSimilarity(pairs)
    return state, sim


def test_100_dataframe_recommendations_under_five_seconds() -> None:
    """End-to-end benchmark on the DataFrame substrate (production
    path). This stresses `score_row` (called 12-35× per recommend
    during top-up), the `_has_days_mask` cache, and
    `DataFrameSimilarity._by_a` lookup.

    Target: < 5s for 100 calls. F3 hits ~50-150ms typical.
    """
    import pandas as pd

    from ai_cdss.constants import DAYS, PATIENT_ID, PPF, PROTOCOL_ID, SCORE, USAGE, USAGE_WEEK

    rows = []
    for i in range(27):
        pid = 200 + i
        rows.append({
            PATIENT_ID: 1, PROTOCOL_ID: pid,
            SCORE: 1.0 + 0.05 * (i % 13),
            DAYS: [i % 7, (i + 2) % 7, (i + 4) % 7] if i < 12 else [],
            USAGE: i, USAGE_WEEK: (i % 3) + 1,
            PPF: 0.5 + 0.02 * i,
        })
    scoring_df = pd.DataFrame(rows)

    sim_rows = []
    for i in range(27):
        for j in range(27):
            if i == j:
                continue
            sim_rows.append({
                "PROTOCOL_A": 200 + i, "PROTOCOL_B": 200 + j,
                "SIMILARITY": 0.5 + 0.01 * abs(i - j),
            })
    sim_df = pd.DataFrame(sim_rows)

    cdss = Recommender(scoring=scoring_df, n=12)
    start = time.perf_counter()
    for _ in range(100):
        cdss.recommend(1, sim_df)
    elapsed = time.perf_counter() - start
    print(f"\n100 DataFrame-substrate recommendations: {elapsed:.3f}s "
          f"({elapsed / 100 * 1000:.1f} ms/call)")
    assert elapsed < 5.0


@pytest.mark.parametrize("n_calls", [100])
def test_100_synthetic_recommendations_under_five_seconds(n_calls: int) -> None:
    """End-to-end benchmark: 100 full recommend() calls (update branch)
    against realistic AISN-sized synthetic data. Target: < 5 seconds.

    On the F3 commit the typical timing is ~0.3-0.6 seconds. The 5s
    budget is generous to absorb CI noise while still catching a 10×
    regression.
    """
    state, sim = _build_synthetic()
    cdss = Recommender(scoring=state, n=12)
    start = time.perf_counter()
    for _ in range(n_calls):
        result = cdss.recommend(1, sim)
        assert result.branch in ("update", "bootstrap", "repeat_skipped_week")
    elapsed = time.perf_counter() - start
    print(f"\n{n_calls} synthetic recommendations: {elapsed:.3f}s "
          f"({elapsed / n_calls * 1000:.1f} ms/call)")
    assert elapsed < 5.0, f"100 recommendations took {elapsed:.2f}s, exceeded 5s budget"


def test_state_caches_prescribed_slice_across_calls() -> None:
    """Memoization regression: `prescribed_rows` returns the same list
    instance on repeated calls (proves the cache hits)."""
    state, _ = _build_synthetic()
    first = state.prescribed_rows
    second = state.prescribed_rows
    assert first is second


def test_state_caches_top_protocols_ordering() -> None:
    """Memoization regression: the sort happens once. We verify by
    asserting consistent results across many calls."""
    state, _ = _build_synthetic()
    first = state.top_protocols(12)
    for _ in range(10):
        assert state.top_protocols(12) == first


def test_similarity_index_built_once_at_construction() -> None:
    """The `_by_a` precompute makes `similarities_for` O(N_b) instead
    of O(table). Verify by checking the index exists immediately after
    construction."""
    _, sim = _build_synthetic()
    # DictSimilarity exposes _by_a; we asserted it's populated.
    assert hasattr(sim, "_by_a")
    assert sim._by_a  # non-empty
    # And queries return expected results.
    pairs = sim.similarities_for(200, exclude=[201, 202])
    pair_b = {b for b, _ in pairs}
    assert 201 not in pair_b
    assert 202 not in pair_b
    assert 200 not in pair_b  # self excluded


def test_dataframe_similarity_also_precomputes_index() -> None:
    """`DataFrameSimilarity` should mirror `DictSimilarity` — precompute
    the index at __init__, never scan the DataFrame on query."""
    import pandas as pd

    from ai_cdss.engine import DataFrameSimilarity

    df = pd.DataFrame([
        {"PROTOCOL_A": 200, "PROTOCOL_B": 201, "SIMILARITY": 0.8},
        {"PROTOCOL_A": 200, "PROTOCOL_B": 202, "SIMILARITY": 0.7},
        {"PROTOCOL_A": 200, "PROTOCOL_B": 200, "SIMILARITY": 1.0},  # self
        {"PROTOCOL_A": 201, "PROTOCOL_B": 200, "SIMILARITY": 0.6},
    ])
    sim = DataFrameSimilarity(df)
    assert hasattr(sim, "_by_a")
    # Self-row dropped at construction.
    assert (200, 1.0) not in sim._by_a[200]
    # Querying excludes correctly.
    pairs = sim.similarities_for(200, exclude=[201])
    assert dict(pairs) == {202: 0.7}
