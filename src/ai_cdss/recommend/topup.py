"""Top-up coverage — universal post-step to fill the (day × slot) grid.

The AISN trial mandates 7 days × `protocols_per_day` protocols per day
(35 slots when ppd=5). Branch outputs (bootstrap / update / repeat) may
under-cover this grid. The top-up step fills any gap **additively**:

  * Tier 1 ("existing"): give an already-prescribed protocol an
    additional day, preferring days it does not yet cover.
  * Tier 2 ("top_pool"): pull in a top-N-by-SCORE protocol that is
    NOT currently prescribed. Schedule it on the deficit day only.
  * If both tiers are dry, the day stays under-filled and we record
    a `source="exhausted"` trace entry.

Important: top-up NEVER removes or moves existing (protocol, day) pairs.
It only adds.

Behavior preserved exactly from the v0.3.1 implementation in
`CDSS._top_up_coverage`.
"""
from __future__ import annotations

from typing import Any

from ai_cdss.constants import DAYS, PATIENT_ID, PROTOCOL_ID
from ai_cdss.recommend.state import PatientState


def fill_grid_coverage(
    patient: PatientState,
    rows: list[dict[str, Any]],
    *,
    n_days: int,
    protocols_per_day: int,
    n: int,
    trace: dict | None = None,
) -> list[dict[str, Any]]:
    """Fill any day-slot gap so every day reaches `protocols_per_day`
    distinct protocols.

    `n` is the bootstrap top-N — same value used in
    `bootstrap.build_recommendations` — which feeds the tier-2 pool.

    Returns a fresh list of row dicts; the input `rows` may be mutated
    in place (existing protocols get extended DAYS lists).
    """
    proto_to_row = _index_rows_by_protocol(rows)
    day_protos = _index_protocols_by_day(rows, n_days=n_days)
    filler = _build_filler_pool(patient, proto_to_row, n)

    for day in range(n_days):
        while len(day_protos[day]) < protocols_per_day:
            pick = _next_filler(filler, day_protos[day])
            if pick is None:
                _record_exhaustion(trace, day, day_protos[day], protocols_per_day)
                break
            _apply_filler(
                patient=patient,
                pick=pick,
                day=day,
                proto_to_row=proto_to_row,
                day_protos=day_protos,
                trace=trace,
            )

    return list(proto_to_row.values())


# ---------------------------------------------------------------------------
# Indexing helpers.

def _index_rows_by_protocol(rows: list[dict[str, Any]]) -> dict[int, dict]:
    return {r[PROTOCOL_ID]: r for r in rows}


def _index_protocols_by_day(
    rows: list[dict[str, Any]], *, n_days: int
) -> dict[int, list[int]]:
    by_day: dict[int, list[int]] = {d: [] for d in range(n_days)}
    for row in rows:
        protocol_id = row[PROTOCOL_ID]
        for d in row.get(DAYS, []) or []:
            if protocol_id not in by_day[d]:
                by_day[d].append(protocol_id)
    return by_day


def _build_filler_pool(
    patient: PatientState,
    existing_rows: dict[int, dict],
    n: int,
) -> list[int]:
    """Existing protocols first (extends DAYS), then top-N protocols not
    already in the recommendation set (tier-2 "soft adds")."""
    existing = list(existing_rows.keys())
    top_pool = [p for p in patient.top_protocols(n) if p not in existing_rows]
    return existing + top_pool


# ---------------------------------------------------------------------------
# Per-iteration helpers.

def _next_filler(filler: list[int], already_on_day: list[int]) -> int | None:
    """First filler protocol that is not yet on this day. None if the
    pool is exhausted for the day."""
    return next((p for p in filler if p not in already_on_day), None)


def _apply_filler(
    *,
    patient: PatientState,
    pick: int,
    day: int,
    proto_to_row: dict[int, dict],
    day_protos: dict[int, list[int]],
    trace: dict | None,
) -> None:
    """Apply the chosen filler to `day`. Updates indices in place.

    Records the source label (`existing` or `top_pool`) in the trace.
    """
    day_protos[day].append(pick)
    if pick in proto_to_row:
        _extend_existing_days(proto_to_row[pick], day)
        source = "existing"
    else:
        proto_to_row[pick] = _new_row_for_top_pool(patient, pick, day)
        source = "top_pool"

    if trace is not None:
        trace["topup"].append({
            "day":         day,
            "protocol_id": int(pick),
            "source":      source,
        })


def _extend_existing_days(row: dict[str, Any], day: int) -> None:
    """Add `day` to an existing row's DAYS list (sorted, deduped)."""
    current = list(row.get(DAYS, []) or [])
    row[DAYS] = sorted(set(current + [day]))


def _new_row_for_top_pool(
    patient: PatientState, protocol_id: int, day: int
) -> dict[str, Any]:
    """Seed a new row for a tier-2 ("top_pool") protocol — scheduled
    on a single day to start."""
    row = patient.score_row(protocol_id)
    row[PROTOCOL_ID] = protocol_id
    row[PATIENT_ID] = patient.patient_id
    row[DAYS] = [day]
    return row


# ---------------------------------------------------------------------------
# Exhaustion bookkeeping.

def _record_exhaustion(
    trace: dict | None,
    day: int,
    on_day: list[int],
    protocols_per_day: int,
) -> None:
    if trace is None:
        return
    trace["topup"].append({
        "day":         day,
        "protocol_id": None,
        "source":      "exhausted",
        "deficit":     protocols_per_day - len(on_day),
    })
