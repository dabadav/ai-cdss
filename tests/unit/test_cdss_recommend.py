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
    (every prescribed protocol shares the same ``days_inherited`` list).

    For tests that need a realistic distribution (protocols spread across
    multiple days with at most ``protocols_per_day`` per day), use
    ``_scoring_frame_distributed`` instead.
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


def _scoring_frame_distributed(
    n_protocols: int,
    n_prescribed: int,
    days_available: list[int],
    protocols_per_day: int = 5,
    patient_id: int = 1,
) -> pd.DataFrame:
    """Build a scoring frame where prescribed protocols are spread across
    ``days_available`` round-robin, capped at ``protocols_per_day`` per
    day — i.e. mirrors what a healthy prior week would look like."""
    day_to_protos: dict[int, list[int]] = {d: [] for d in days_available}
    proto_to_days: dict[int, list[int]] = {}
    cursor = 0
    for slot in range(len(days_available) * protocols_per_day):
        if cursor >= n_prescribed:
            break
        day = days_available[slot % len(days_available)]
        if len(day_to_protos[day]) >= protocols_per_day:
            continue
        protocol_id = 200 + cursor
        day_to_protos[day].append(protocol_id)
        proto_to_days.setdefault(protocol_id, []).append(day)
        cursor += 1
    # Refill remaining day slots by cycling already-prescribed protocols
    proto_iter = list(proto_to_days.keys())
    pi = 0
    for day in days_available:
        while len(day_to_protos[day]) < protocols_per_day and proto_iter:
            cand = proto_iter[pi % len(proto_iter)]
            pi += 1
            if cand not in day_to_protos[day]:
                day_to_protos[day].append(cand)
                proto_to_days[cand].append(day)

    rows = []
    for i in range(n_protocols):
        protocol_id = 200 + i
        prescribed_days = sorted(set(proto_to_days.get(protocol_id, [])))
        rows.append(
            {
                PATIENT_ID:        patient_id,
                PROTOCOL_ID:       protocol_id,
                SCORE:             1.0 - i * 0.02,
                PPF:               0.5,
                DELTA_DM:          0.01,
                RECENT_ADHERENCE:  0.9,
                USAGE:             3 if prescribed_days else 0,
                USAGE_WEEK:        2 if prescribed_days else 0,
                SESSION_INDEX:     1,
                WEEKS_SINCE_START: 4,
                DAYS:              prescribed_days,
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


def test_update_branch_tops_up_when_inherited_thin_4904_case():
    """Mirror of the patient-4904 incident: 5 protocols all on Tuesday.
    Top-up must spread the 5 inherited protocols (and pull from the top-N
    pool if needed) to fill every day with `protocols_per_day` protocols.
    Tuesday's existing 5 stay as-is; other days reach 5 too.
    """
    n, days, ppd = 12, 7, 5
    scoring = _scoring_frame(n_protocols=20, n_prescribed=5, days_inherited=[1])
    similarity = _similarity_frame(scoring)

    cdss = CDSS(scoring=scoring, n=n, days=days, protocols_per_day=ppd)
    rec = cdss.recommend(patient_id=1, protocol_similarity=similarity)

    flat_days = [d for lst in rec[DAYS] for d in lst]
    counts = pd.Series(flat_days).value_counts().sort_index()
    assert set(counts.index) == set(range(days))
    assert (counts >= ppd).all(), (
        f"every day must reach {ppd} protocols after top-up, got {counts.tolist()}"
    )
    assert rec[PROTOCOL_ID].nunique() >= ppd, "expected at least ppd distinct protocols"


def test_update_branch_tops_up_when_inherited_six_days_distributed():
    """Realistic prior week: 12 protocols distributed across Tue-Sun (no Mon)
    at 5/day. After top-up, Monday must be filled with `ppd` protocols and
    no day's count may drop below `ppd`.
    """
    n, days, ppd = 12, 7, 5
    scoring = _scoring_frame_distributed(
        n_protocols=20, n_prescribed=12,
        days_available=[1, 2, 3, 4, 5, 6], protocols_per_day=ppd,
    )
    similarity = _similarity_frame(scoring)
    cdss = CDSS(scoring=scoring, n=n, days=days, protocols_per_day=ppd)
    rec = cdss.recommend(patient_id=1, protocol_similarity=similarity)

    flat_days = [d for lst in rec[DAYS] for d in lst]
    counts = pd.Series(flat_days).value_counts().sort_index()
    assert 0 in counts.index, "Monday must be covered after top-up"
    assert (counts >= ppd).all(), f"every day >= {ppd}, got {counts.tolist()}"


def test_update_branch_full_grid_distributed_seven_days():
    """Healthy prior week: 12 protocols across 7 days × 5/day. After update
    + top-up, output must still be exactly 35 slots (no inflation)."""
    n, days, ppd = 12, 7, 5
    scoring = _scoring_frame_distributed(
        n_protocols=20, n_prescribed=12,
        days_available=list(range(7)), protocols_per_day=ppd,
    )
    similarity = _similarity_frame(scoring)
    cdss = CDSS(scoring=scoring, n=n, days=days, protocols_per_day=ppd)
    rec = cdss.recommend(patient_id=1, protocol_similarity=similarity)

    flat_days = [d for lst in rec[DAYS] for d in lst]
    counts = pd.Series(flat_days).value_counts().sort_index()
    assert counts.tolist() == [ppd] * days, (
        f"expected exactly {ppd}/day, got {counts.tolist()}"
    )
    assert rec[PROTOCOL_ID].nunique() == n


def test_update_branch_preserves_kept_protocol_days():
    """Stability: a protocol that survives the swap must keep every (proto,
    day) pair it had last week. Top-up may *add* day slots but never remove
    nor relocate inherited ones."""
    n, days, ppd = 12, 7, 5
    scoring = _scoring_frame_distributed(
        n_protocols=20, n_prescribed=12,
        days_available=[1, 2, 3, 4, 5, 6], protocols_per_day=ppd,
    )
    similarity = _similarity_frame(scoring)
    cdss = CDSS(scoring=scoring, n=n, days=days, protocols_per_day=ppd)

    # Identify which prescribed protocols would survive the swap. Swap rule:
    # SCORE < mean(SCORE) among current prescriptions. Kept = score >= mean.
    prior = scoring[scoring[DAYS].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    mean_score = prior[SCORE].mean()
    kept = prior[prior[SCORE] >= mean_score]
    kept_days_by_proto = {row[PROTOCOL_ID]: set(row[DAYS]) for _, row in kept.iterrows()}

    rec = cdss.recommend(patient_id=1, protocol_similarity=similarity)

    for p, original_days in kept_days_by_proto.items():
        sub = rec[rec[PROTOCOL_ID] == p]
        assert not sub.empty, f"kept protocol {p} dropped"
        post_days = set(sub.iloc[0][DAYS])
        assert original_days.issubset(post_days), (
            f"kept protocol {p} lost inherited days; "
            f"inherited={sorted(original_days)} got={sorted(post_days)}"
        )


def test_recommend_attaches_full_trace_for_update_branch():
    """The trace dict on `out.attrs['trace']` must be rich enough to
    reconstruct the run: branch, top_protocols, prior, every swap event,
    every top-up addition, final schedule."""
    n, days, ppd = 12, 7, 5
    scoring = _scoring_frame_distributed(
        n_protocols=20, n_prescribed=12,
        days_available=[1, 2, 3, 4, 5, 6], protocols_per_day=ppd,
    )
    similarity = _similarity_frame(scoring)
    cdss = CDSS(scoring=scoring, n=n, days=days, protocols_per_day=ppd)
    rec = cdss.recommend(patient_id=1, protocol_similarity=similarity)

    trace = rec.attrs["trace"]
    assert trace["branch"] == "update"
    assert trace["config"] == {"n": n, "days": days, "protocols_per_day": ppd}
    assert isinstance(trace["top_protocols"], list) and len(trace["top_protocols"]) == n
    assert trace["prior"], "prior must capture the prior-week prescription state"
    assert all({"protocol_id", "days", "score", "usage_week"}.issubset(p) for p in trace["prior"])
    # At least one swap event (AISN min-1-swap rule guarantees this when n_prescribed>=1)
    assert len(trace["swaps"]) >= 1
    swap = trace["swaps"][0]
    assert {"removed", "added", "similarity", "inherited_days", "candidate_pool", "reason"}.issubset(swap)
    # Top-up should add Monday slots (day=0) since inherited window had no Mon
    assert any(t["day"] == 0 for t in trace["topup"]), "expected Monday top-ups"
    # final schedule covers exactly days x ppd
    final_slots = sum(len(p["days"]) for p in trace["final"])
    assert final_slots == days * ppd


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
