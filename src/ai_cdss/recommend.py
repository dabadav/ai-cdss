"""Recommendation engine — one file, read top-to-bottom.

The engine takes a `scoring` DataFrame (one row per (patient, protocol))
and a `protocol_similarity` table, and produces a weekly schedule for
ONE patient at a time. The schedule is `n` distinct protocols laid out
across `days` × `protocols_per_day` slots.

This file is organized in **sections that mirror the algorithm's steps**:

    1.  PatientState        — wrap the scoring frame for one patient
    2.  trace               — build the structured audit dict
    3.  branch dispatch     — pick bootstrap / repeat / update
    4.  bootstrap branch    — first-ever schedule
    5.  repeat branch       — week was skipped, copy prior
    6.  update branch       — MVT-driven swap loop
    7.  swap targets (MVT)  — which prescribed protocols to swap
    8.  substitute search   — two-tier pick: unused / least-used-similar
    9.  topup               — fill the 7×ppd grid post-step
   10.  CDSS orchestrator   — entry-point class wiring everything

Each section is preceded by a banner. Read top-to-bottom for the full
algorithm; jump to a section title to dig into one part.

Behavior is byte-for-byte identical to v0.3.1 of the original
`cdss.py`. The 21 existing unit tests pass unchanged.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ai_cdss.constants import (
    DAYS,
    N,
    N_DAYS,
    PATIENT_ID,
    PROTOCOL_A,
    PROTOCOL_B,
    PROTOCOL_ID,
    PROTOCOLS_PER_DAY,
    SCORE,
    SIMILARITY,
    USAGE,
    USAGE_WEEK,
)

logger = logging.getLogger(__name__)


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1 — PatientState                                            ║
# ║                                                                      ║
# ║  The scoring DataFrame has one row per (patient, protocol). The      ║
# ║  engine only cares about ONE patient. Wrapping that single-patient   ║
# ║  slice gives every helper a single argument with intent-named        ║
# ║  accessors. Pure read-only view — does not mutate `scoring`.         ║
# ╚═════════════════════════════════════════════════════════════════════╝

class PatientState:
    """All scoring data for a single patient. Read-only view."""

    def __init__(self, scoring: pd.DataFrame, patient_id: int) -> None:
        self._scoring = scoring
        self.patient_id = patient_id
        self.rows = scoring.loc[scoring[PATIENT_ID] == patient_id]

    @property
    def has_data(self) -> bool:
        return not self.rows.empty

    @property
    def prescriptions(self) -> pd.DataFrame:
        """Subset of `rows` engine treats as 'currently prescribed'
        — rows with a non-empty DAYS list. Empty DAYS = in cohort but
        not on this week's schedule."""
        has_days = self.rows[DAYS].apply(
            lambda d: isinstance(d, list) and len(d) > 0
        )
        return self.rows.loc[has_days]

    def is_week_skipped(self) -> bool:
        """True iff every scheduled prescription this week recorded
        USAGE_WEEK == 0. Returns False on empty prior."""
        scheduled = self.prescriptions
        if scheduled.empty:
            return False
        return bool((scheduled[USAGE_WEEK] == 0).all())

    def top_protocols(self, n: int) -> list[int]:
        """Top-N protocols by SCORE."""
        return self.rows.nlargest(n, SCORE)[PROTOCOL_ID].tolist()

    @property
    def lowest_scoring_prescribed(self) -> int:
        """Protocol with the lowest SCORE among currently-prescribed.
        Used to satisfy the AISN min-1-swap rule when no protocol falls
        below the swap threshold."""
        pres = self.prescriptions
        return int(pres.loc[pres[SCORE].idxmin(), PROTOCOL_ID])

    @property
    def usage(self) -> pd.Series:
        """Per-protocol usage count, indexed by PROTOCOL_ID."""
        return self.rows.set_index(PROTOCOL_ID)[USAGE]

    def score_row(self, protocol_id: int) -> dict[str, Any]:
        """Scoring row for `(patient_id, protocol_id)` as a plain dict."""
        match = self.rows.loc[self.rows[PROTOCOL_ID] == protocol_id]
        return match.iloc[0].to_dict()


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2 — Trace construction                                      ║
# ║                                                                      ║
# ║  The recommendation pipeline emits a structured `trace` dict that    ║
# ║  captures every decision. Downstream (supervisor backtest, decision  ║
# ║  view) reads this dict to reconstruct the run. Shape preserved from  ║
# ║  v0.3.1.                                                             ║
# ╚═════════════════════════════════════════════════════════════════════╝

def _init_trace(
    *, patient: PatientState, n: int, n_days: int, protocols_per_day: int,
) -> dict[str, Any]:
    return {
        "patient_id":     patient.patient_id,
        "config":         {"n": n, "days": n_days, "protocols_per_day": protocols_per_day},
        "top_protocols":  patient.top_protocols(n),
        "branch":         None,
        "prior":          [],
        "swaps":          [],
        "topup":          [],
        "final":          [],
    }


def _serialize_prior(prescriptions: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "protocol_id": int(row[PROTOCOL_ID]),
            "days":        list(row[DAYS]) if isinstance(row[DAYS], list) else [],
            "score":       _safe_float(row.get(SCORE)),
            "usage_week":  _safe_int(row.get(USAGE_WEEK)),
        }
        for _, row in prescriptions.iterrows()
    ]


def _serialize_final(recommendations: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "protocol_id": int(row[PROTOCOL_ID]),
            "days":        sorted(int(d) for d in (row.get(DAYS) or [])),
        }
        for _, row in recommendations.iterrows()
    ]


def _safe_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return float(value) if pd.notna(value) else None


def _safe_int(value: Any) -> int:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0
    return int(value) if pd.notna(value) else 0


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3 — Bootstrap branch                                        ║
# ║                                                                      ║
# ║  When the patient has no prior week (`_get_prescriptions` empty),    ║
# ║  pick top-N by SCORE and round-robin across the week.                ║
# ╚═════════════════════════════════════════════════════════════════════╝

def _bootstrap_branch(
    patient: PatientState, *, n: int, n_days: int, protocols_per_day: int,
) -> pd.DataFrame:
    """Top-N protocols laid out round-robin across the week."""
    top_protocols = patient.top_protocols(n)
    schedule = _round_robin_across_days(
        top_protocols, n_days=n_days, protocols_per_day=protocols_per_day,
    )
    rows_by_protocol = _seed_rows_from_schedule(patient, schedule)

    df = pd.DataFrame(rows_by_protocol.values()).sort_values(
        by=PROTOCOL_ID
    ).reset_index(drop=True)
    df.attrs = patient.rows.attrs
    return df


def _round_robin_across_days(
    protocols: list[int], *, n_days: int, protocols_per_day: int,
) -> dict[int, list[int]]:
    """Distribute `protocols` round-robin across `n_days`. Each day caps
    at `protocols_per_day` distinct protocols. Day 0 = Monday."""
    schedule: dict[int, list[int]] = {d: [] for d in range(n_days)}
    if not protocols:
        return schedule

    total_slots = n_days * protocols_per_day
    repeats = math.ceil(total_slots / len(protocols))
    sequence = (protocols * repeats)[:total_slots]

    for i, protocol in enumerate(sequence):
        day = i % n_days
        if protocol not in schedule[day]:
            schedule[day].append(protocol)
    return schedule


def _seed_rows_from_schedule(
    patient: PatientState, schedule: dict[int, list[int]],
) -> dict[int, dict]:
    """Walk the per-day schedule; accumulate one row per protocol with
    growing DAYS list."""
    rows: dict[int, dict] = {}
    for day, protocol_ids in schedule.items():
        for protocol_id in protocol_ids:
            if protocol_id in rows:
                rows[protocol_id][DAYS].append(day)
                continue
            row = patient.score_row(protocol_id)
            row[DAYS] = [day]
            row[PROTOCOL_ID] = protocol_id
            row[PATIENT_ID] = patient.patient_id
            rows[protocol_id] = row
    return rows


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4 — Repeat branch                                           ║
# ║                                                                      ║
# ║  When every prescribed row this week recorded USAGE_WEEK == 0, the   ║
# ║  patient skipped the entire week. Repeat the prior schedule as-is.   ║
# ╚═════════════════════════════════════════════════════════════════════╝

def _repeat_branch(patient: PatientState) -> pd.DataFrame:
    """Return a copy of the prior prescriptions unchanged."""
    if patient.prescriptions.empty:
        logger.info("repeat_week called with empty prescriptions for patient=%s",
                    patient.patient_id)
    else:
        logger.info("Patient %s, skipped the whole week, cdss repeating prescriptions.",
                    patient.patient_id)
    df = patient.prescriptions.copy()
    df.attrs = patient.rows.attrs
    return df


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5 — MVT swap criterion                                      ║
# ║                                                                      ║
# ║  Marginal Value Theorem: a prescribed protocol is a swap candidate   ║
# ║  if its SCORE is strictly below the mean SCORE of the currently-     ║
# ║  prescribed set. Strict `<` — ties at the mean stay.                 ║
# ║                                                                      ║
# ║  v0.3.1 semantics preserved here. Env-wide alternative explored on   ║
# ║  the original repo's `feat/env-wide-mvt-threshold` branch.           ║
# ╚═════════════════════════════════════════════════════════════════════╝

def _below_mean_protocols(prescriptions: pd.DataFrame) -> list[int]:
    """Protocol IDs whose SCORE is strictly below the prescribed-set
    mean. DataFrame-row order preserved."""
    if prescriptions.empty:
        return []
    mean = prescriptions[SCORE].mean()
    below_mask = prescriptions[SCORE] < mean
    return prescriptions.loc[below_mask, PROTOCOL_ID].tolist()


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 6 — Similarity queries                                      ║
# ║                                                                      ║
# ║  Slice and rank the protocol-similarity table. Pure functions; the   ║
# ║  substitute search depends on these.                                 ║
# ╚═════════════════════════════════════════════════════════════════════╝

def _similarities_for(
    protocol_id: int,
    similarity_table: pd.DataFrame,
    excluded: list[int] | None = None,
) -> pd.DataFrame:
    """Rows describing how similar `protocol_id` is to other protocols.
    Excludes self and any protocol in `excluded`."""
    rows = similarity_table.loc[similarity_table[PROTOCOL_A] == protocol_id]
    rows = rows.loc[rows[PROTOCOL_A] != rows[PROTOCOL_B]]
    if excluded:
        rows = rows.loc[~rows[PROTOCOL_B].isin(excluded)]
    return rows


def _top_n_similar(similarities: pd.DataFrame, n: int = 5) -> list[int]:
    return similarities.nlargest(n, SIMILARITY)[PROTOCOL_B].tolist()


def _most_similar_within(
    candidates: list[int], similarities: pd.DataFrame,
) -> int | None:
    """Highest-similarity candidate in `candidates`, or None if no
    candidate is in `similarities`. Ties broken by first-row order."""
    matched = similarities.loc[similarities[PROTOCOL_B].isin(candidates)]
    if matched.empty:
        return None
    peak = matched[SIMILARITY].max()
    return int(matched.loc[matched[SIMILARITY] == peak, PROTOCOL_B].iloc[0])


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 7 — Substitute search (two-tier)                            ║
# ║                                                                      ║
# ║  Tier 1: pick most-similar protocol the patient has NEVER used.      ║
# ║  Tier 2: if every candidate has been used, pick the least-used       ║
# ║          among the top-5 most-similar.                               ║
# ║  Both fail → return None. Caller decides what to do; the v0.3.1      ║
# ║  engine falls back to keeping the original protocol in place         ║
# ║  (the "swap_returned_same_protocol" trace marker).                   ║
# ╚═════════════════════════════════════════════════════════════════════╝

@dataclass
class SubstituteResult:
    """Outcome of a substitute search. Auditable — carries which tier
    matched, which candidates were considered."""
    protocol_id: int | None
    tier:        str            # "unused" | "least_used_top_similar" | "exhausted"
    candidates:  list[int]


def _find_substitute(
    *,
    usage: pd.Series,
    similarities: pd.DataFrame,
    removed_protocol_id: int,
) -> SubstituteResult:
    # Tier 1: unused most-similar.
    unused = usage.loc[usage == 0].index.tolist()
    if unused:
        logger.info("No usage for %s, selecting most similar from %s",
                    removed_protocol_id, unused)
        pick = _most_similar_within(unused, similarities)
        if pick is not None:
            return SubstituteResult(pick, "unused", unused)

    # Tier 2: least-used among top-5 most-similar.
    top5 = _top_n_similar(similarities, n=5)
    least_used = _least_used_among(usage, top5)
    if least_used:
        logger.info("No unused protocols for %s, selecting least used from %s",
                    removed_protocol_id, least_used)
        pick = _most_similar_within(least_used, similarities)
        if pick is not None:
            return SubstituteResult(pick, "least_used_top_similar", least_used)

    return SubstituteResult(None, "exhausted", [])


def _least_used_among(usage: pd.Series, candidates: list[int]) -> list[int]:
    """Protocols in `candidates` sharing the minimum usage value."""
    sub = usage.loc[usage.index.isin(candidates)]
    if sub.empty:
        return []
    floor = sub.min()
    return sub.loc[sub == floor].index.tolist()


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 8 — Update branch                                           ║
# ║                                                                      ║
# ║  Prior week exists and wasn't skipped. Identify swap targets via     ║
# ║  MVT, run substitute search for each, assemble the new schedule     ║
# ║  (kept rows + swapped rows). Universal top-up runs afterwards.       ║
# ╚═════════════════════════════════════════════════════════════════════╝

def _update_branch(
    patient: PatientState,
    similarity_table: pd.DataFrame,
    *, trace: dict | None = None,
) -> pd.DataFrame:
    prior = patient.prescriptions
    targets, reasons = _select_swap_targets(patient)

    kept_rows = _rows_kept_unchanged(prior, targets)
    swapped_rows = _build_swap_rows(
        patient=patient, prior=prior, similarity_table=similarity_table,
        swap_targets=targets, reasons=reasons, trace=trace,
    )

    combined = pd.DataFrame(kept_rows + swapped_rows).sort_values(
        by=PROTOCOL_ID
    ).reset_index(drop=True)
    combined.attrs = patient.rows.attrs
    return combined


def _select_swap_targets(
    patient: PatientState,
) -> tuple[list[int], dict[int, str]]:
    """Return `(targets, reason_by_protocol)`. If MVT yields no
    candidate, force a single swap (AISN min-1-swap rule)."""
    targets = _below_mean_protocols(patient.prescriptions)
    reasons = {p: "below_mean_score" for p in targets}
    if not targets:
        forced = patient.lowest_scoring_prescribed
        targets = [forced]
        reasons = {forced: "aisn_min_one_swap"}
    return targets, reasons


def _rows_kept_unchanged(
    prior: pd.DataFrame, swap_targets: list[int],
) -> list[dict[str, Any]]:
    """Prior rows whose protocol is NOT in `swap_targets` — kept
    verbatim, DAYS preserved."""
    keep_mask = ~prior[PROTOCOL_ID].isin(swap_targets)
    return prior.loc[keep_mask].to_dict("records")


def _build_swap_rows(
    *,
    patient: PatientState,
    prior: pd.DataFrame,
    similarity_table: pd.DataFrame,
    swap_targets: list[int],
    reasons: dict[int, str],
    trace: dict | None,
) -> list[dict[str, Any]]:
    """Greedy substitute loop. `excluded` grows as substitutes are picked
    so the pool depletes deterministically."""
    excluded = prior[PROTOCOL_ID].tolist()
    swapped: list[dict[str, Any]] = []

    for removed_id in swap_targets:
        sims = _similarities_for(removed_id, similarity_table, excluded)
        result = _find_substitute(
            usage=patient.usage, similarities=sims,
            removed_protocol_id=removed_id,
        )

        new_row = _materialize_swap_row(
            patient=patient, prior=prior,
            removed_id=removed_id, substitute_id=result.protocol_id,
        )
        new_id = int(new_row[PROTOCOL_ID])

        if trace is not None:
            trace["swaps"].append(_swap_trace_event(
                removed_id=removed_id, new_id=new_id, prior=prior,
                sims=sims, new_row=new_row, reason=reasons.get(removed_id, "unknown"),
            ))

        swapped.append(new_row)
        excluded.append(new_id)
    return swapped


def _materialize_swap_row(
    *,
    patient: PatientState,
    prior: pd.DataFrame,
    removed_id: int,
    substitute_id: int | None,
) -> dict[str, Any]:
    """Build the row that replaces `removed_id`'s slot.

    Substitute found → it inherits the removed protocol's DAYS verbatim.
    No substitute (pool exhausted) → fall back to keeping the removed
    protocol in place. This fallback (v0.3.1 commit `cf733c4`,
    2025-07-01) is what produces the `swap_returned_same_protocol`
    backtest warning."""
    if substitute_id is None:
        return patient.score_row(removed_id)

    new_row = patient.score_row(substitute_id)
    new_row[DAYS] = prior.loc[
        prior[PROTOCOL_ID] == removed_id, DAYS
    ].values[0]
    new_row[PROTOCOL_ID] = substitute_id
    new_row[PATIENT_ID] = patient.patient_id
    return new_row


def _swap_trace_event(
    *,
    removed_id: int,
    new_id: int,
    prior: pd.DataFrame,
    sims: pd.DataFrame,
    new_row: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    """The `{removed, removed_score, added, similarity, …}` entry for
    `trace["swaps"]`. Also emits the v0.3.1 info-level log line."""
    removed_score = (
        float(prior.loc[prior[PROTOCOL_ID] == removed_id, SCORE].iloc[0])
        if SCORE in prior.columns and not prior.loc[prior[PROTOCOL_ID] == removed_id].empty
        else None
    )
    sim_row = sims.loc[sims[PROTOCOL_B] == new_id]
    sim_value = float(sim_row[SIMILARITY].iloc[0]) if not sim_row.empty else None
    inherited_days = sorted(int(d) for d in (new_row.get(DAYS) or []))

    logger.info(
        "Swap patient=%s removed=%s (score=%s) -> added=%s (sim=%s) days=%s reason=%s",
        new_row.get(PATIENT_ID), removed_id, removed_score,
        new_id, sim_value, inherited_days, reason,
    )
    return {
        "removed":        int(removed_id),
        "removed_score":  removed_score,
        "added":          new_id,
        "similarity":     sim_value,
        "inherited_days": inherited_days,
        "candidate_pool": sims[PROTOCOL_B].astype(int).tolist(),
        "reason":         reason,
    }


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 9 — Top-up coverage (universal post-step)                   ║
# ║                                                                      ║
# ║  AISN mandates 7 days × ppd protocols. Branch outputs may under-     ║
# ║  cover. Top-up fills additively: existing protocols get extra days,  ║
# ║  then top-N unused protocols ("top_pool" — un-MVT-audited soft adds).║
# ║  Never removes or moves existing (protocol, day) pairs.              ║
# ╚═════════════════════════════════════════════════════════════════════╝

def _fill_grid_coverage(
    patient: PatientState,
    rows: list[dict[str, Any]],
    *, n_days: int, protocols_per_day: int, n: int,
    trace: dict | None = None,
) -> list[dict[str, Any]]:
    proto_to_row = {r[PROTOCOL_ID]: r for r in rows}
    day_protos = _index_protocols_by_day(rows, n_days=n_days)
    filler = _build_filler_pool(patient, proto_to_row, n)

    for day in range(n_days):
        while len(day_protos[day]) < protocols_per_day:
            pick = next((p for p in filler if p not in day_protos[day]), None)
            if pick is None:
                _record_exhaustion(trace, day, day_protos[day], protocols_per_day)
                break
            _apply_filler(
                patient=patient, pick=pick, day=day,
                proto_to_row=proto_to_row, day_protos=day_protos, trace=trace,
            )

    return list(proto_to_row.values())


def _index_protocols_by_day(
    rows: list[dict[str, Any]], *, n_days: int,
) -> dict[int, list[int]]:
    by_day: dict[int, list[int]] = {d: [] for d in range(n_days)}
    for row in rows:
        protocol_id = row[PROTOCOL_ID]
        for d in row.get(DAYS, []) or []:
            if protocol_id not in by_day[d]:
                by_day[d].append(protocol_id)
    return by_day


def _build_filler_pool(
    patient: PatientState, existing_rows: dict[int, dict], n: int,
) -> list[int]:
    """Existing protocols first (extends DAYS); then top-N protocols
    not yet in the recommendation set (tier-2 'soft adds')."""
    existing = list(existing_rows.keys())
    top_pool = [p for p in patient.top_protocols(n) if p not in existing_rows]
    return existing + top_pool


def _apply_filler(
    *,
    patient: PatientState,
    pick: int,
    day: int,
    proto_to_row: dict[int, dict],
    day_protos: dict[int, list[int]],
    trace: dict | None,
) -> None:
    day_protos[day].append(pick)
    if pick in proto_to_row:
        current = list(proto_to_row[pick].get(DAYS, []) or [])
        proto_to_row[pick][DAYS] = sorted(set(current + [day]))
        source = "existing"
    else:
        row = patient.score_row(pick)
        row[PROTOCOL_ID] = pick
        row[PATIENT_ID] = patient.patient_id
        row[DAYS] = [day]
        proto_to_row[pick] = row
        source = "top_pool"

    if trace is not None:
        trace["topup"].append({
            "day": day, "protocol_id": int(pick), "source": source,
        })


def _record_exhaustion(
    trace: dict | None, day: int, on_day: list[int], protocols_per_day: int,
) -> None:
    if trace is None:
        return
    trace["topup"].append({
        "day": day, "protocol_id": None,
        "source": "exhausted", "deficit": protocols_per_day - len(on_day),
    })


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 10 — CDSS orchestrator (entry-point class)                  ║
# ║                                                                      ║
# ║  The class everyone imports. Holds engine config (n / days / ppd),   ║
# ║  exposes `recommend(patient_id, similarity)`. Inside, dispatches to  ║
# ║  the right branch, runs top-up, attaches the trace.                  ║
# ╚═════════════════════════════════════════════════════════════════════╝

class CDSS:
    """Clinical Decision Support System.

    Recommends a 7-day × `protocols_per_day` schedule of rehab protocols
    for ONE patient at a time, using the patient's PPF + session
    history baked into `scoring`.
    """

    def __init__(
        self,
        scoring: pd.DataFrame,
        n: int = N,
        days: int = N_DAYS,
        protocols_per_day: int = PROTOCOLS_PER_DAY,
    ) -> None:
        self.scoring = scoring
        self.n = n
        self.days = days
        self.protocols_per_day = protocols_per_day

    def recommend(
        self, patient_id: int, protocol_similarity: pd.DataFrame,
    ) -> pd.DataFrame:
        """Full recommendation pipeline. Attaches a structured `trace`
        dict to `.attrs['trace']`."""
        patient = PatientState(self.scoring, patient_id)
        if not patient.has_data:
            raise ValueError(f"Patient {patient_id} has no data.")

        trace = _init_trace(
            patient=patient, n=self.n,
            n_days=self.days, protocols_per_day=self.protocols_per_day,
        )

        recommendations = self._dispatch_branch(patient, protocol_similarity, trace)
        recommendations = self._apply_topup(patient, recommendations, trace)
        return self._finalize(recommendations, trace)

    # ------------------------------------------------------------------
    # Branch dispatch — three mutually exclusive code paths.

    def _dispatch_branch(
        self,
        patient: PatientState,
        protocol_similarity: pd.DataFrame,
        trace: dict,
    ) -> pd.DataFrame:
        if patient.prescriptions.empty:
            trace["branch"] = "bootstrap"
            return _bootstrap_branch(
                patient, n=self.n, n_days=self.days,
                protocols_per_day=self.protocols_per_day,
            )

        if patient.is_week_skipped():
            trace["branch"] = "repeat_skipped_week"
            return _repeat_branch(patient)

        trace["branch"] = "update"
        trace["prior"] = _serialize_prior(patient.prescriptions)
        return _update_branch(patient, protocol_similarity, trace=trace)

    # ------------------------------------------------------------------
    # Universal post-step + finalize.

    def _apply_topup(
        self, patient: PatientState, recommendations: pd.DataFrame, trace: dict,
    ) -> pd.DataFrame:
        rows = recommendations.to_dict("records")
        rows = _fill_grid_coverage(
            patient, rows, n_days=self.days,
            protocols_per_day=self.protocols_per_day,
            n=self.n, trace=trace,
        )
        return pd.DataFrame(rows).sort_values(
            by=PROTOCOL_ID
        ).reset_index(drop=True)

    def _finalize(
        self, recommendations: pd.DataFrame, trace: dict,
    ) -> pd.DataFrame:
        trace["final"] = _serialize_final(recommendations)
        attrs = dict(self.scoring.attrs)
        attrs["trace"] = trace
        recommendations.attrs = attrs
        return recommendations
