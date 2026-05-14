"""Tests for `CDSS._normalize_input` — pre-swap shape trim.

Three independent trims (distinct count, per-day count, empty DAYS)
applied to the patient's prior-week prescription before the swap loop
sees it. Every drop should be visible in trace["trimmed"].
"""

import pandas as pd
import pytest

from ai_cdss.cdss import CDSS
from ai_cdss.constants import (
    DAYS,
    DELTA_DM,
    PATIENT_ID,
    PPF,
    PROTOCOL_ID,
    RECENT_ADHERENCE,
    SCORE,
    SESSION_INDEX,
    USAGE,
    USAGE_WEEK,
    WEEKS_SINCE_START,
)


def _scoring_with(prescriptions: list[dict], extra_unused: int = 5, patient_id: int = 1) -> pd.DataFrame:
    """Build a minimal scoring frame: each `prescriptions` entry is a dict
    `{protocol_id, score, days}`. Pads with `extra_unused` non-prescribed
    protocols (empty DAYS) so the patient has a realistic candidate pool."""
    rows = []
    for p in prescriptions:
        rows.append({
            PATIENT_ID:        patient_id,
            PROTOCOL_ID:       int(p["protocol_id"]),
            SCORE:             float(p["score"]),
            PPF:               0.5,
            DELTA_DM:          0.01,
            RECENT_ADHERENCE:  0.9,
            USAGE:             3,
            USAGE_WEEK:        2,
            SESSION_INDEX:     1,
            WEEKS_SINCE_START: 4,
            DAYS:              list(p["days"]),
        })
    base_pid = max(int(p["protocol_id"]) for p in prescriptions) + 1
    for i in range(extra_unused):
        rows.append({
            PATIENT_ID:        patient_id,
            PROTOCOL_ID:       base_pid + i,
            SCORE:             0.3,
            PPF:               0.5,
            DELTA_DM:          0.0,
            RECENT_ADHERENCE:  0.0,
            USAGE:             0,
            USAGE_WEEK:        0,
            SESSION_INDEX:     0,
            WEEKS_SINCE_START: 4,
            DAYS:              [],
        })
    return pd.DataFrame(rows)


def _make_cdss(scoring: pd.DataFrame, n: int = 12, days: int = 7, ppd: int = 5) -> CDSS:
    cdss = CDSS(scoring=scoring, n=n, days=days, protocols_per_day=ppd)
    cdss._trace = {"trimmed": []}
    return cdss


def test_normalize_input_trim_distinct():
    """n=14 prescribed → 12 kept, 2 lowest-scoring dropped, both recorded.

    Day distribution is at-spec (≤ ppd per day) so only Trim 1 fires —
    isolates the distinct-count trim from per-day trim.
    """
    # 14 protocols × 1 day each, cycling days 0..6 → each day has exactly 2.
    prescriptions = [
        {"protocol_id": 200 + i, "score": 1.00 - i * 0.05, "days": [i % 7]}
        for i in range(14)
    ]
    scoring = _scoring_with(prescriptions)
    cdss   = _make_cdss(scoring)
    prior  = cdss._get_prescriptions(1)
    out    = cdss._normalize_input(1, prior)

    assert len(out) == 12
    kept_ids = set(out[PROTOCOL_ID].astype(int))
    # Top-12 by SCORE = first 12 in the list (scores descending by index)
    assert kept_ids == {200 + i for i in range(12)}
    trimmed = cdss._trace["trimmed"]
    assert len(trimmed) == 2
    assert {t["protocol_id"] for t in trimmed} == {212, 213}
    assert all(t["reason"] == "n_distinct_over_max" for t in trimmed)


def test_normalize_input_trim_per_day():
    """Day 0 has 7 prescribed protocols → 5 kept (top by score), 2 lose day 0.

    Other days kept under ppd cap so only the day-0 trim fires."""
    # Day 0: 7 protocols. Day 1 / day 2 / day 3: at-spec (≤ 5).
    prescriptions = [
        {"protocol_id": 200, "score": 1.00, "days": [0, 1]},
        {"protocol_id": 201, "score": 0.95, "days": [0, 1]},
        {"protocol_id": 202, "score": 0.90, "days": [0, 1]},
        {"protocol_id": 203, "score": 0.85, "days": [0, 1]},
        {"protocol_id": 204, "score": 0.80, "days": [0, 1]},
        # The two extras on day 0 also live on day 2 (where there's room)
        {"protocol_id": 205, "score": 0.75, "days": [0, 2]},
        {"protocol_id": 206, "score": 0.70, "days": [0, 2]},  # 7th on day 0
        # Filler so n is reasonable and day 2 has ≤ 5
        {"protocol_id": 207, "score": 0.65, "days": [3]},
        {"protocol_id": 208, "score": 0.60, "days": [3]},
        {"protocol_id": 209, "score": 0.55, "days": [3]},
    ]
    scoring = _scoring_with(prescriptions)
    cdss   = _make_cdss(scoring)
    prior  = cdss._get_prescriptions(1)
    out    = cdss._normalize_input(1, prior)

    # Build day-protocol map from output
    day_count: dict[int, set[int]] = {}
    for _, r in out.iterrows():
        for d in r[DAYS]:
            day_count.setdefault(int(d), set()).add(int(r[PROTOCOL_ID]))
    assert len(day_count[0]) == 5
    # The 2 lowest-scoring on day 0 (205, 206) should have lost it
    assert 205 not in day_count[0]
    assert 206 not in day_count[0]
    # And kept their other-day prescriptions
    assert 205 in day_count[2]
    assert 206 in day_count[2]

    trimmed = cdss._trace["trimmed"]
    per_day = [t for t in trimmed if t["reason"] == "per_day_over_max"]
    assert len(per_day) == 2
    assert {t["protocol_id"] for t in per_day} == {205, 206}


def test_normalize_input_empty_after_trim():
    """Protocol whose only day gets trimmed by per-day rule is dropped entirely."""
    # Day 0 has 6 protocols, one of which has DAYS=[0] only. Trim 1 leaves it.
    # Trim 2 removes day 0 from the lowest-scoring on day 0 — that's also the
    # one with only [0], so its DAYS goes empty → Trim 3 drops it.
    prescriptions = [
        {"protocol_id": 200, "score": 1.00, "days": [0, 1]},
        {"protocol_id": 201, "score": 0.95, "days": [0, 1]},
        {"protocol_id": 202, "score": 0.90, "days": [0, 1]},
        {"protocol_id": 203, "score": 0.85, "days": [0, 1]},
        {"protocol_id": 204, "score": 0.80, "days": [0, 2]},
        {"protocol_id": 205, "score": 0.40, "days": [0]},      # only day 0; will be empty after Trim 2
    ]
    scoring = _scoring_with(prescriptions)
    cdss   = _make_cdss(scoring)
    prior  = cdss._get_prescriptions(1)
    out    = cdss._normalize_input(1, prior)

    out_ids = set(out[PROTOCOL_ID].astype(int))
    assert 205 not in out_ids
    trimmed = cdss._trace["trimmed"]
    reasons = [t["reason"] for t in trimmed if t["protocol_id"] == 205]
    assert "per_day_over_max" in reasons
    assert "empty_after_per_day_trim" in reasons


def test_normalize_input_passthrough_at_spec():
    """n=12 with all days at ≤ ppd: no trim, trace.trimmed stays empty."""
    prescriptions = [
        {"protocol_id": 200 + i, "score": 1.0 - i * 0.02, "days": [i % 7]}
        for i in range(12)
    ]
    scoring = _scoring_with(prescriptions)
    cdss   = _make_cdss(scoring)
    prior  = cdss._get_prescriptions(1)
    out    = cdss._normalize_input(1, prior)

    assert len(out) == 12
    assert cdss._trace["trimmed"] == []
