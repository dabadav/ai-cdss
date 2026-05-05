"""Behaviour tests for CDSS.recommend output shape.

These tests pin the AISN trial invariant: every weekly recommendation must
cover ``days × protocols_per_day`` slots with ``n`` distinct protocols, no
matter how thin the inherited DAYS lists are. Patient compliance from the
prior week must not collapse next week's coverage.
"""

import pandas as pd
import pytest

from ai_cdss.cdss import CDSS
from ai_cdss.constants import (
    DAYS,
    DELTA_DM,
    PATIENT_ID,
    PPF,
    PROTOCOL_A,
    PROTOCOL_B,
    PROTOCOL_ID,
    RECENT_ADHERENCE,
    SCORE,
    SESSION_INDEX,
    SIMILARITY,
    USAGE,
    USAGE_WEEK,
    WEEKS_SINCE_START,
)


def _scoring_frame(
    n_protocols: int,
    days_inherited: list[int],
    n_prescribed: int = 12,
    patient_id: int = 1,
) -> pd.DataFrame:
    """Build a minimal scoring frame for one patient with `n_protocols`
    candidate protocols. The first ``n_prescribed`` are pre-prescribed
    (have non-empty ``DAYS``); the rest are unused candidates available
    as substitutes / top-ups.
    """
    rows = []
    for i in range(n_protocols):
        protocol_id = 200 + i
        prescribed = i < n_prescribed
        rows.append(
            {
                PATIENT_ID:        patient_id,
                PROTOCOL_ID:       protocol_id,
                SCORE:             1.0 - i * 0.02,
                PPF:               0.5,
                DELTA_DM:          0.01,
                RECENT_ADHERENCE:  0.9,
                USAGE:             3 if prescribed else 0,
                USAGE_WEEK:        2 if prescribed else 0,
                SESSION_INDEX:     1,
                WEEKS_SINCE_START: 4,
                # Inherited DAYS comes from build_prescription_days. Only
                # rows with non-empty list count as "current prescriptions".
                DAYS:              list(days_inherited) if prescribed else [],
            }
        )
    return pd.DataFrame(rows)


def _similarity_frame(scoring: pd.DataFrame) -> pd.DataFrame:
    """All-pairs similarity, descending by protocol_id distance."""
    proto = scoring[PROTOCOL_ID].unique().tolist()
    rows = []
    for a in proto:
        for b in proto:
            if a == b:
                continue
            rows.append({PROTOCOL_A: a, PROTOCOL_B: b, SIMILARITY: 1.0 / (1 + abs(a - b))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Full-grid invariant: update branch must always emit days × protocols_per_day
# ---------------------------------------------------------------------------


def test_update_branch_emits_full_grid_when_inherited_days_thin():
    """Patient previously prescribed only Tuesday (DAYS=[1]) for 5 protocols.
    Update branch must still produce 7 days × 5/day coverage with 12 distinct
    protocols.
    """
    n, days, ppd = 12, 7, 5
    scoring = _scoring_frame(n_protocols=20, days_inherited=[1])
    similarity = _similarity_frame(scoring)

    cdss = CDSS(scoring=scoring, n=n, days=days, protocols_per_day=ppd)
    rec = cdss.recommend(patient_id=1, protocol_similarity=similarity)

    # Distinct protocols: must equal n
    assert rec[PROTOCOL_ID].nunique() == n, (
        f"expected {n} distinct protocols, got {rec[PROTOCOL_ID].nunique()}"
    )

    # Total day-slots across all protocols must equal days × protocols_per_day
    total_slots = rec[DAYS].apply(len).sum()
    assert total_slots == days * ppd, (
        f"expected {days * ppd} slots, got {total_slots}"
    )

    # Every day index in [0, days) must be covered, and each day should have
    # exactly protocols_per_day protocols assigned.
    flat_days = [d for lst in rec[DAYS] for d in lst]
    counts = pd.Series(flat_days).value_counts().sort_index()
    assert set(counts.index) == set(range(days))
    assert counts.tolist() == [ppd] * days, f"per-day counts: {counts.tolist()}"


def test_update_branch_emits_full_grid_when_inherited_days_six():
    """Patient previously prescribed Tue-Sun (DAYS=[1..6], no Mon). Update
    branch must still emit 7-day coverage including Monday (index 0).
    """
    n, days, ppd = 12, 7, 5
    scoring = _scoring_frame(n_protocols=20, days_inherited=[1, 2, 3, 4, 5, 6])
    similarity = _similarity_frame(scoring)
    cdss = CDSS(scoring=scoring, n=n, days=days, protocols_per_day=ppd)
    rec = cdss.recommend(patient_id=1, protocol_similarity=similarity)

    flat_days = [d for lst in rec[DAYS] for d in lst]
    assert 0 in flat_days, "Monday (day 0) must be covered after re-fan"
    assert set(flat_days) == set(range(days))
    assert pd.Series(flat_days).value_counts().tolist() == [ppd] * days


def test_bootstrap_branch_emits_full_grid():
    """Bootstrap path (no prior prescriptions) must also yield full coverage."""
    n, days, ppd = 12, 7, 5
    scoring = _scoring_frame(n_protocols=20, days_inherited=[])  # all empty -> bootstrap
    # Still need a few non-empty rows? No — bootstrap branch triggers when
    # _get_prescriptions returns empty (DAYS lists all empty).
    similarity = _similarity_frame(scoring)
    cdss = CDSS(scoring=scoring, n=n, days=days, protocols_per_day=ppd)
    rec = cdss.recommend(patient_id=1, protocol_similarity=similarity)

    assert rec[PROTOCOL_ID].nunique() == n
    flat_days = [d for lst in rec[DAYS] for d in lst]
    assert set(flat_days) == set(range(days))
    assert pd.Series(flat_days).value_counts().tolist() == [ppd] * days
