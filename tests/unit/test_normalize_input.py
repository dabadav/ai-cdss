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
    """Day 0 over cap → protocol with MOST current days loses day 0
    (regardless of score). Preserves diversity for single-day protocols."""
    # Day 0: 6 protocols. Day 1, 2, 3 used as filler so 200's multi-day
    # presence is unambiguous. Cap = 5 → drop 1 from day 0.
    prescriptions = [
        # 200 is on 4 days total → should be the day-0 victim under
        # "most days" policy even though it has the highest score.
        {"protocol_id": 200, "score": 1.00, "days": [0, 1, 2, 3]},
        {"protocol_id": 201, "score": 0.90, "days": [0, 1]},
        {"protocol_id": 202, "score": 0.80, "days": [0]},
        {"protocol_id": 203, "score": 0.70, "days": [0]},
        {"protocol_id": 204, "score": 0.60, "days": [0]},
        # 6th on day 0, lowest score — under OLD policy this would be
        # the victim. Under NEW policy 200 (most days) loses day 0
        # instead, and 205 survives.
        {"protocol_id": 205, "score": 0.50, "days": [0]},
    ]
    scoring = _scoring_with(prescriptions)
    cdss   = _make_cdss(scoring)
    prior  = cdss._get_prescriptions(1)
    out    = cdss._normalize_input(1, prior)

    # Build day-protocol map
    day_count: dict[int, set[int]] = {}
    for _, r in out.iterrows():
        for d in r[DAYS]:
            day_count.setdefault(int(d), set()).add(int(r[PROTOCOL_ID]))

    assert len(day_count[0]) == 5
    # 200 (most days) lost day 0
    assert 200 not in day_count[0]
    # 205 (lowest score) is preserved
    assert 205 in day_count[0]
    # 200 still alive on its other days
    assert 200 in day_count[1] and 200 in day_count[2] and 200 in day_count[3]

    trimmed = cdss._trace["trimmed"]
    per_day = [t for t in trimmed if t["reason"] == "per_day_over_max"]
    assert len(per_day) == 1
    assert per_day[0]["protocol_id"] == 200
    assert per_day[0]["removed_days"] == [0]
    # New victim_day_count field records load at decision time
    assert per_day[0]["victim_day_count"] == 4


def test_normalize_input_trim_per_day_iterates_when_needed():
    """If trim 1 isn't enough (day still over cap), the inner loop picks
    a second victim — each pass re-evaluates current day counts."""
    # Two protocols tied for "most days" on day 0; both have 3 days,
    # day 0 has 7 protocols → need to drop 2 from day 0.
    prescriptions = [
        {"protocol_id": 200, "score": 1.00, "days": [0, 1, 2]},
        {"protocol_id": 201, "score": 0.95, "days": [0, 3, 4]},
        {"protocol_id": 202, "score": 0.90, "days": [0]},
        {"protocol_id": 203, "score": 0.80, "days": [0]},
        {"protocol_id": 204, "score": 0.70, "days": [0]},
        {"protocol_id": 205, "score": 0.60, "days": [0]},
        {"protocol_id": 206, "score": 0.50, "days": [0]},
    ]
    scoring = _scoring_with(prescriptions)
    cdss   = _make_cdss(scoring)
    prior  = cdss._get_prescriptions(1)
    out    = cdss._normalize_input(1, prior)

    day_count: dict[int, set[int]] = {}
    for _, r in out.iterrows():
        for d in r[DAYS]:
            day_count.setdefault(int(d), set()).add(int(r[PROTOCOL_ID]))
    assert len(day_count[0]) == 5
    # 200 and 201 (the 3-day protocols) should each lose day 0 — they
    # were tied on |DAYS|=3, then 200 loses first (lower protocol_id),
    # after which 201 still has 3 days and 200 has 2 → 201 is next.
    assert 200 not in day_count[0]
    assert 201 not in day_count[0]
    trimmed_per_day = [t for t in cdss._trace["trimmed"] if t["reason"] == "per_day_over_max"]
    assert len(trimmed_per_day) == 2
    assert {t["protocol_id"] for t in trimmed_per_day} == {200, 201}


def test_normalize_input_empty_after_trim():
    """Under the 'most days' policy, Trim 3 only fires when EVERY protocol
    on the over-cap day has the same minimal day count. Tiebreak by
    PROTOCOL_ID ascending picks the smallest id as victim, which then
    becomes empty and is dropped by Trim 3.
    """
    # 6 protocols all on day 0 only — every protocol has |DAYS|=1.
    # Tie on day count → tiebreak picks 200 → 200 loses day 0 → empty.
    prescriptions = [
        {"protocol_id": 200, "score": 1.00, "days": [0]},
        {"protocol_id": 201, "score": 0.95, "days": [0]},
        {"protocol_id": 202, "score": 0.90, "days": [0]},
        {"protocol_id": 203, "score": 0.85, "days": [0]},
        {"protocol_id": 204, "score": 0.80, "days": [0]},
        {"protocol_id": 205, "score": 0.75, "days": [0]},
    ]
    scoring = _scoring_with(prescriptions)
    cdss   = _make_cdss(scoring)
    prior  = cdss._get_prescriptions(1)
    out    = cdss._normalize_input(1, prior)

    out_ids = set(out[PROTOCOL_ID].astype(int))
    # 200 was picked (lowest pid on tie), stripped to empty, dropped
    assert 200 not in out_ids
    trimmed = cdss._trace["trimmed"]
    reasons = [t["reason"] for t in trimmed if t["protocol_id"] == 200]
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
