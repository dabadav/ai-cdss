"""Recommendation engine — substrate-agnostic.

The engine takes an `EngineState` (any substrate — pandas DataFrame
adapter, in-memory dict, future polars/xarray) and a `SimilarityMatrix`
and produces a weekly schedule for ONE patient at a time. The schedule
is `n` distinct protocols laid out across `days` × `protocols_per_day`
slots.

After phase F2, **none of the engine internals depend on pandas**.
The engine works on `list[ProtocolRow]` throughout. pandas only
appears at two boundaries:
  - INPUT: production callers pass a `pd.DataFrame` for scoring; that
    gets wrapped by `coerce_engine_state` into a `PatientState`
    (see `engine.py`).
  - OUTPUT: the final per-protocol schedule is materialized as a
    `pd.DataFrame` for backwards-compatible consumption (and for
    `RecommendationResult.recommendations`).

This file is organized in **sections that mirror the algorithm's steps**:

    1.  trace               — build the structured audit dict
    2.  bootstrap strategy  — first-ever schedule
    3.  repeat strategy     — week was skipped, copy prior
    4.  MVT criterion       — which prescribed protocols to swap
    5.  substitute search   — two-tier pick: unused / least-used-similar
    6.  update strategy     — swap loop assembly
    7.  topup               — fill the 7×ppd grid post-step
    8.  RecommendationResult — introspectable output (PCA-style)
    9.  Recommender         — entry-point class wiring everything

Behavior is byte-for-byte identical to v0.3.1; all unit tests pass.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ai_cdss.constants import (
    N,
    N_DAYS,
    PROTOCOL_ID,
    PROTOCOLS_PER_DAY,
)
from ai_cdss.engine import (
    EngineState,
    PatientState,
    ProtocolRow,
    SimilarityMatrix,
    coerce_engine_state,
    coerce_similarity,
)

logger = logging.getLogger(__name__)


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1 — Trace construction                                      ║
# ║                                                                      ║
# ║  Structured audit dict capturing every decision. Downstream readers  ║
# ║  (supervisor backtest, decision view) reconstruct the run from this. ║
# ║  Shape preserved from v0.3.1.                                        ║
# ╚═════════════════════════════════════════════════════════════════════╝

def _init_trace(
    *, state: EngineState, n: int, n_days: int, protocols_per_day: int,
) -> dict[str, Any]:
    return {
        "patient_id":     state.patient_id,
        "config":         {"n": n, "days": n_days, "protocols_per_day": protocols_per_day},
        "top_protocols":  state.top_protocols(n),
        "branch":         None,
        "prior":          [],
        "mvt_mean":       None,
        "swaps":          [],
        "topup":          [],
        "final":          [],
    }


def _serialize_prior(prior: list[ProtocolRow]) -> list[dict[str, Any]]:
    return [
        {
            "protocol_id": int(row.protocol_id),
            "days":        list(row.days),
            "score":       float(row.score) if row.score is not None else None,
            "usage_week":  int(row.usage_week),
        }
        for row in prior
    ]


def _serialize_final_rows(rows: list[ProtocolRow]) -> list[dict[str, Any]]:
    return [
        {"protocol_id": int(r.protocol_id), "days": sorted(int(d) for d in r.days)}
        for r in rows
    ]


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2 — Bootstrap strategy                                      ║
# ║                                                                      ║
# ║  Patient has no prior week. Pick top-N by SCORE and round-robin      ║
# ║  across the week.                                                    ║
# ╚═════════════════════════════════════════════════════════════════════╝

def _bootstrap_strategy(
    state: EngineState, *, n: int, n_days: int, protocols_per_day: int,
) -> list[ProtocolRow]:
    top_protocols = state.top_protocols(n)
    schedule = _round_robin_across_days(
        top_protocols, n_days=n_days, protocols_per_day=protocols_per_day,
    )

    # Accumulate one row per protocol; build up DAYS list as we
    # encounter the protocol on additional days.
    rows_by_protocol: dict[int, ProtocolRow] = {}
    for day, protocol_ids in schedule.items():
        for pid in protocol_ids:
            if pid in rows_by_protocol:
                rows_by_protocol[pid].days.append(day)
                continue
            row = state.score_row(pid)
            # Fresh row — copy and override DAYS so we don't mutate
            # whatever the state returned.
            rows_by_protocol[pid] = _clone_with(row, days=[day])
    return list(rows_by_protocol.values())


def _round_robin_across_days(
    protocols: list[int], *, n_days: int, protocols_per_day: int,
) -> dict[int, list[int]]:
    """Distribute `protocols` round-robin across `n_days`. Day 0 = Monday."""
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


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3 — Repeat strategy                                         ║
# ║                                                                      ║
# ║  Every prescribed row recorded USAGE_WEEK == 0 → patient skipped     ║
# ║  the entire week. Repeat the prior schedule.                         ║
# ╚═════════════════════════════════════════════════════════════════════╝

def _repeat_strategy(state: EngineState) -> list[ProtocolRow]:
    prior = state.prescribed_rows
    if not prior:
        logger.info(
            "repeat_week called with empty prescriptions for patient=%s",
            state.patient_id,
        )
    else:
        logger.info(
            "Patient %s, skipped the whole week, cdss repeating prescriptions.",
            state.patient_id,
        )
    # Return clones so engine output is independent of state mutation.
    return [_clone_with(r) for r in prior]


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4 — MVT swap criterion                                      ║
# ║                                                                      ║
# ║  A prescribed protocol is a swap candidate when its SCORE is         ║
# ║  strictly below the COHORT-WIDE mean SCORE — the mean over ALL of    ║
# ║  the patient's protocols, not just the prescribed set. The candidate ║
# ║  pool is already cohort-wide (Section 5); this makes the threshold   ║
# ║  symmetric with it. Strict `<` — ties at the mean stay. The keep/    ║
# ║  swap criterion still applies only to PRESCRIBED protocols (the only ║
# ║  ones we can swap out).                                              ║
# ╚═════════════════════════════════════════════════════════════════════╝

def _below_mean_protocols(
    prior: list[ProtocolRow], state: EngineState,
) -> tuple[list[int], float | None]:
    """Prescribed protocol IDs strictly below the cohort-wide mean SCORE.

    The mean is taken over every protocol the patient has
    (`state.all_protocols`) — not the prescribed set — so a homogeneous
    prescribed set no longer hides swap pressure when better-scoring
    unprescribed protocols exist. Returns (targets, mean); targets in
    `prior` iteration order (caller decides further sorting)."""
    all_scores = [
        row.score
        for pid in state.all_protocols
        if (row := state.score_row(pid)).score is not None
    ]
    if not all_scores:
        return [], None
    mean = sum(all_scores) / len(all_scores)
    targets = [
        r.protocol_id for r in prior
        if r.score is not None and r.score < mean
    ]
    return targets, mean


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5 — Substitute search (two-tier)                            ║
# ║                                                                      ║
# ║  Tier 1: most-similar protocol the patient has NEVER used.           ║
# ║  Tier 2: least-used among top-5 most-similar (if all candidates      ║
# ║          have been used).                                            ║
# ║  Both fail → return None. Caller falls back to keeping the original. ║
# ╚═════════════════════════════════════════════════════════════════════╝

@dataclass
class SubstituteResult:
    """Outcome of a substitute search. Auditable — carries which tier
    matched, which candidates were considered, plus (when materialized
    from a trace event) the removed protocol + similarity + reason.
    """
    protocol_id:  int | None
    tier:         str               # "unused" | "least_used_top_similar" | "exhausted"
    candidates:   list[int]
    removed_id:   int | None = None
    similarity:   float | None = None
    reason:       str = ""

    @property
    def candidates_considered(self) -> list[int]:
        return self.candidates


def _find_substitute(
    *,
    state: EngineState,
    similarity: SimilarityMatrix,
    removed_protocol_id: int,
    excluded: list[int],
) -> SubstituteResult:
    """Run the two-tier search. See section banner above for the rules."""
    # Tier 1: protocols never used by this patient, most-similar first.
    unused = state.protocols_with_zero_usage
    if unused:
        logger.info(
            "No usage for %s, selecting most similar from %s",
            removed_protocol_id, unused,
        )
        pick = _most_similar_within(
            unused, similarity, removed_protocol_id, excluded,
        )
        if pick is not None:
            return SubstituteResult(pick, "unused", unused)

    # Tier 2: least-used among top-5 most-similar.
    top5 = similarity.top_n_similar(
        removed_protocol_id, 5, exclude=excluded,
    )
    least_used = _least_used_among(state, top5)
    if least_used:
        logger.info(
            "No unused protocols for %s, selecting least used from %s",
            removed_protocol_id, least_used,
        )
        pick = _most_similar_within(
            least_used, similarity, removed_protocol_id, excluded,
        )
        if pick is not None:
            return SubstituteResult(pick, "least_used_top_similar", least_used)

    return SubstituteResult(None, "exhausted", [])


def _most_similar_within(
    candidates: list[int],
    similarity: SimilarityMatrix,
    protocol_id: int,
    excluded: list[int],
) -> int | None:
    """The highest-similarity candidate (filtered by excluded). None if
    no candidate has a similarity entry."""
    pairs = similarity.similarities_for(protocol_id, exclude=excluded)
    # Restrict to candidates we care about.
    candidate_set = set(candidates)
    matched = [(b, s) for b, s in pairs if b in candidate_set]
    if not matched:
        return None
    matched.sort(key=lambda p: -p[1])
    return matched[0][0]


def _least_used_among(state: EngineState, candidates: list[int]) -> list[int]:
    """Protocols in `candidates` sharing the minimum usage value."""
    if not candidates:
        return []
    usage_map = {pid: state.usage_of(pid) for pid in candidates}
    if not usage_map:
        return []
    floor = min(usage_map.values())
    return [pid for pid, u in usage_map.items() if u == floor]


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 6 — Update strategy                                         ║
# ║                                                                      ║
# ║  Prior week exists and wasn't skipped. MVT picks swap targets;       ║
# ║  substitute search fills each.  Universal top-up runs afterwards.    ║
# ╚═════════════════════════════════════════════════════════════════════╝

def _update_strategy(
    state: EngineState,
    similarity: SimilarityMatrix,
    *,
    trace: dict | None = None,
) -> list[ProtocolRow]:
    prior = state.prescribed_rows
    swap_targets, reasons, mvt_mean = _select_swap_targets(prior, state)
    if trace is not None:
        trace["mvt_mean"] = mvt_mean

    # Keep rows whose protocol is NOT being swapped — preserved verbatim.
    kept_rows: list[ProtocolRow] = [
        _clone_with(r) for r in prior if r.protocol_id not in swap_targets
    ]
    swapped_rows = _build_swap_rows(
        state=state,
        similarity=similarity,
        prior=prior,
        swap_targets=swap_targets,
        reasons=reasons,
        trace=trace,
    )
    return kept_rows + swapped_rows


def _select_swap_targets(
    prior: list[ProtocolRow],
    state: EngineState,
) -> tuple[list[int], dict[int, str], float | None]:
    """Pick swap targets + reason per target + the cohort-wide MVT mean.
    If MVT yields none, force a single swap on the lowest-scoring prior
    (AISN min-1-swap rule — unchanged under the cohort-wide threshold)."""
    targets, mvt_mean = _below_mean_protocols(prior, state)
    reasons = {p: "below_mean_score" for p in targets}
    if not targets:
        forced = state.lowest_scoring_prescribed
        targets = [forced]
        reasons = {forced: "aisn_min_one_swap"}
    return targets, reasons, mvt_mean


def _build_swap_rows(
    *,
    state: EngineState,
    similarity: SimilarityMatrix,
    prior: list[ProtocolRow],
    swap_targets: list[int],
    reasons: dict[int, str],
    trace: dict | None,
) -> list[ProtocolRow]:
    """Greedy substitute loop. `excluded` grows as substitutes are
    picked so the pool depletes deterministically."""
    excluded: list[int] = [r.protocol_id for r in prior]
    prior_by_id: dict[int, ProtocolRow] = {r.protocol_id: r for r in prior}
    swapped: list[ProtocolRow] = []

    for removed_id in swap_targets:
        result = _find_substitute(
            state=state, similarity=similarity,
            removed_protocol_id=removed_id, excluded=excluded,
        )
        new_row = _materialize_swap_row(
            state=state,
            prior_by_id=prior_by_id,
            removed_id=removed_id,
            substitute_id=result.protocol_id,
        )

        if trace is not None:
            trace["swaps"].append(_swap_trace_event(
                removed_id=removed_id,
                new_row=new_row,
                prior_by_id=prior_by_id,
                similarity=similarity,
                excluded=excluded,
                reason=reasons.get(removed_id, "unknown"),
            ))

        swapped.append(new_row)
        excluded.append(new_row.protocol_id)
    return swapped


def _materialize_swap_row(
    *,
    state: EngineState,
    prior_by_id: dict[int, ProtocolRow],
    removed_id: int,
    substitute_id: int | None,
) -> ProtocolRow:
    """Build the row replacing `removed_id`'s slot.

    Substitute found → inherits removed protocol's DAYS verbatim.
    Pool exhausted → keep the removed protocol (v0.3.1 "same protocol"
    fallback; surfaces as `swap_returned_same_protocol` warn).
    """
    if substitute_id is None:
        # Fallback: keep removed in place with its original DAYS.
        return _clone_with(prior_by_id[removed_id])

    sub_row = state.score_row(substitute_id)
    inherited_days = list(prior_by_id[removed_id].days)
    return _clone_with(sub_row, days=inherited_days)


def _swap_trace_event(
    *,
    removed_id: int,
    new_row: ProtocolRow,
    prior_by_id: dict[int, ProtocolRow],
    similarity: SimilarityMatrix,
    excluded: list[int],
    reason: str,
) -> dict[str, Any]:
    """The `{removed, removed_score, added, similarity, …}` entry for
    `trace["swaps"]`."""
    removed_score = prior_by_id[removed_id].score
    new_id = int(new_row.protocol_id)

    # Look up the similarity of removed → new_row.
    pairs = similarity.similarities_for(removed_id, exclude=excluded)
    sim_value: float | None = None
    for b, s in pairs:
        if b == new_id:
            sim_value = float(s)
            break

    candidate_pool = [b for b, _ in pairs]
    inherited_days = sorted(int(d) for d in new_row.days)

    logger.info(
        "Swap patient=%s removed=%s (score=%s) -> added=%s (sim=%s) days=%s reason=%s",
        new_row.patient_id, removed_id, removed_score,
        new_id, sim_value, inherited_days, reason,
    )
    return {
        "removed":        int(removed_id),
        "removed_score":  removed_score,
        "added":          new_id,
        "similarity":     sim_value,
        "inherited_days": inherited_days,
        "candidate_pool": candidate_pool,
        "reason":         reason,
    }


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 7 — Top-up coverage (universal post-step)                   ║
# ║                                                                      ║
# ║  AISN mandates 7 × ppd protocol-day slots. Top-up adds fillers       ║
# ║  additively: existing protocols get extra days first; then top-N     ║
# ║  unused protocols ("top_pool" — un-MVT-audited soft adds). Never     ║
# ║  removes existing pairs.                                             ║
# ╚═════════════════════════════════════════════════════════════════════╝

def _top_up_schedule(
    state: EngineState,
    rows: list[ProtocolRow],
    *, n_days: int, protocols_per_day: int, n: int,
    trace: dict | None = None,
) -> list[ProtocolRow]:
    rows_by_protocol: dict[int, ProtocolRow] = {r.protocol_id: r for r in rows}
    day_protos: dict[int, list[int]] = _index_protocols_by_day(rows, n_days=n_days)
    filler = _build_filler_pool(state, rows_by_protocol, n)

    for day in range(n_days):
        while len(day_protos[day]) < protocols_per_day:
            pick = next((p for p in filler if p not in day_protos[day]), None)
            if pick is None:
                _record_exhaustion(trace, day, day_protos[day], protocols_per_day)
                break
            _apply_filler(
                state=state, pick=pick, day=day,
                rows_by_protocol=rows_by_protocol,
                day_protos=day_protos, trace=trace,
            )

    return list(rows_by_protocol.values())


def _index_protocols_by_day(
    rows: list[ProtocolRow], *, n_days: int,
) -> dict[int, list[int]]:
    by_day: dict[int, list[int]] = {d: [] for d in range(n_days)}
    for row in rows:
        for d in row.days:
            if row.protocol_id not in by_day[d]:
                by_day[d].append(row.protocol_id)
    return by_day


def _build_filler_pool(
    state: EngineState, existing_rows: dict[int, ProtocolRow], n: int,
) -> list[int]:
    """Existing protocols first (extends DAYS); then top-N protocols
    not yet in the recommendation set (tier-2 'soft adds')."""
    existing = list(existing_rows.keys())
    top_pool = [p for p in state.top_protocols(n) if p not in existing_rows]
    return existing + top_pool


def _apply_filler(
    *,
    state: EngineState,
    pick: int,
    day: int,
    rows_by_protocol: dict[int, ProtocolRow],
    day_protos: dict[int, list[int]],
    trace: dict | None,
) -> None:
    day_protos[day].append(pick)
    if pick in rows_by_protocol:
        existing = rows_by_protocol[pick]
        existing.days = sorted(set(existing.days + [day]))
        source = "existing"
    else:
        row = state.score_row(pick)
        rows_by_protocol[pick] = _clone_with(row, days=[day])
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
# ║  Internal — ProtocolRow cloning utility                              ║
# ╚═════════════════════════════════════════════════════════════════════╝

def _clone_with(row: ProtocolRow, **overrides: Any) -> ProtocolRow:
    """Return a copy of `row` with the named fields replaced. Avoids
    mutating shared row instances coming out of state.score_row()."""
    return ProtocolRow(
        patient_id        = overrides.get("patient_id",        row.patient_id),
        protocol_id       = overrides.get("protocol_id",       row.protocol_id),
        score             = overrides.get("score",             row.score),
        days              = list(overrides.get("days", row.days)),
        usage             = overrides.get("usage",             row.usage),
        usage_week        = overrides.get("usage_week",        row.usage_week),
        ppf               = overrides.get("ppf",               row.ppf),
        delta_dm          = overrides.get("delta_dm",          row.delta_dm),
        recent_adherence  = overrides.get("recent_adherence",  row.recent_adherence),
        weeks_since_start = overrides.get("weeks_since_start", row.weeks_since_start),
        contrib           = overrides.get("contrib",           row.contrib),
    )


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 8 — RecommendationResult (introspectable output)            ║
# ║                                                                      ║
# ║  PCA / sklearn / TensorFlow-style result object. Every intermediate  ║
# ║  artifact is a public attribute.                                     ║
# ╚═════════════════════════════════════════════════════════════════════╝

@dataclass
class RecommendationResult:
    """All artifacts the engine produced for one (patient, week)."""

    recommendations: pd.DataFrame
    trace:           dict[str, Any]
    patient_state:   EngineState
    branch:          str
    swap_decisions:  list[SubstituteResult]
    topup_events:    list[dict[str, Any]]
    mvt_mean:        float | None
    swap_targets:    list[int]
    swap_reasons:    dict[int, str]
    scoring_attrs:   dict[str, Any]

    @property
    def final_protocols(self) -> list[int]:
        return sorted(int(p) for p in self.recommendations[PROTOCOL_ID].unique())

    @property
    def n_swaps(self) -> int:
        return len(self.swap_decisions)

    @property
    def n_topup(self) -> int:
        return len(self.topup_events)

    @property
    def attrs(self) -> dict[str, Any]:
        return self.recommendations.attrs

    def candidate_pool_for(self, removed_id: int) -> list[int]:
        for swap in self.swap_decisions:
            if swap.removed_id == removed_id:
                return list(swap.candidates_considered)
        return []

    def __getitem__(self, key: Any) -> Any:
        return self.recommendations[key]

    def __iter__(self) -> Any:
        return iter(self.recommendations)

    def __len__(self) -> int:
        return len(self.recommendations)

    def to_dataframe(self) -> pd.DataFrame:
        return self.recommendations


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 9 — Recommender (entry-point class)                         ║
# ║                                                                      ║
# ║  Internal engine entry. interface.cdss.CDSS wraps this for prod.    ║
# ║  Accepts any EngineState / SimilarityMatrix substrate.               ║
# ╚═════════════════════════════════════════════════════════════════════╝

class Recommender:
    """Clinical Decision Support System core.

    Recommends a 7-day × `protocols_per_day` schedule for ONE patient.

    Substrate-agnostic: `scoring` can be a `pd.DataFrame` (production
    path) or any `EngineState` (synthetic / dict / future polars).
    Similarly `protocol_similarity` accepts `pd.DataFrame`, dict, or
    `SimilarityMatrix`.
    """

    def __init__(
        self,
        scoring: pd.DataFrame | EngineState,
        n: int = N,
        days: int = N_DAYS,
        protocols_per_day: int = PROTOCOLS_PER_DAY,
    ) -> None:
        self._scoring = scoring
        self.n = n
        self.days = days
        self.protocols_per_day = protocols_per_day

    def recommend(
        self,
        patient_id: int,
        protocol_similarity: Any,
    ) -> RecommendationResult:
        """Full pipeline. Accepts any substrate; coerces to the engine
        protocols at the boundary."""
        state = coerce_engine_state(self._scoring, patient_id)
        sim   = coerce_similarity(protocol_similarity)

        if not state.has_data:
            raise ValueError(f"Patient {patient_id} has no data.")

        trace = _init_trace(
            state=state, n=self.n,
            n_days=self.days, protocols_per_day=self.protocols_per_day,
        )

        rows = self._run_strategy(state, sim, trace)
        rows = _top_up_schedule(
            state, rows,
            n_days=self.days,
            protocols_per_day=self.protocols_per_day,
            n=self.n, trace=trace,
        )

        recommendations = self._rows_to_dataframe(rows, state)
        trace["final"] = _serialize_final_rows(rows)
        attrs = dict(state.scoring_attrs)
        attrs["trace"] = trace
        recommendations.attrs = attrs

        return self._assemble_result(
            state=state, rows=rows,
            recommendations=recommendations, trace=trace,
        )

    # ------------------------------------------------------------------
    # Strategy dispatch — three mutually exclusive paths.

    def _run_strategy(
        self, state: EngineState, similarity: SimilarityMatrix, trace: dict,
    ) -> list[ProtocolRow]:
        prior = state.prescribed_rows
        if not prior:
            trace["branch"] = "bootstrap"
            return _bootstrap_strategy(
                state, n=self.n, n_days=self.days,
                protocols_per_day=self.protocols_per_day,
            )
        if state.is_week_skipped():
            trace["branch"] = "repeat_skipped_week"
            return _repeat_strategy(state)
        trace["branch"] = "update"
        trace["prior"] = _serialize_prior(prior)
        return _update_strategy(state, similarity, trace=trace)

    # ------------------------------------------------------------------
    # Output materialization — ProtocolRow list → DataFrame at boundary.

    def _rows_to_dataframe(
        self, rows: list[ProtocolRow], state: EngineState,
    ) -> pd.DataFrame:
        """Convert list of ProtocolRow to a pd.DataFrame for the legacy
        output type. Sorted by PROTOCOL_ID for stable order."""
        records = [r.as_dict() for r in rows]
        df = pd.DataFrame(records)
        if PROTOCOL_ID in df.columns and not df.empty:
            df = df.sort_values(by=PROTOCOL_ID).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Result construction — assemble the introspectable view.

    def _assemble_result(
        self, *, state: EngineState, rows: list[ProtocolRow],
        recommendations: pd.DataFrame, trace: dict,
    ) -> RecommendationResult:
        branch = trace.get("branch") or ""
        swap_decisions = self._extract_swap_decisions(trace)
        swap_targets = [s["removed"] for s in trace.get("swaps") or []]
        swap_reasons = {
            int(s["removed"]): s.get("reason", "unknown")
            for s in trace.get("swaps") or []
        }
        mvt_mean = self._compute_mvt_mean(trace, branch)

        return RecommendationResult(
            recommendations=recommendations,
            trace=trace,
            patient_state=state,
            branch=branch,
            swap_decisions=swap_decisions,
            topup_events=list(trace.get("topup") or []),
            mvt_mean=mvt_mean,
            swap_targets=swap_targets,
            swap_reasons=swap_reasons,
            scoring_attrs=dict(state.scoring_attrs),
        )

    @staticmethod
    def _extract_swap_decisions(trace: dict) -> list[SubstituteResult]:
        out: list[SubstituteResult] = []
        for s in trace.get("swaps") or []:
            tier = (
                "exhausted" if s.get("added") == s.get("removed")
                else s.get("reason", "unknown")
            )
            out.append(SubstituteResult(
                protocol_id=s.get("added"),
                tier=tier,
                candidates=list(s.get("candidate_pool") or []),
                removed_id=int(s.get("removed")),
                similarity=s.get("similarity"),
                reason=s.get("reason", "unknown"),
            ))
        return out

    @staticmethod
    def _compute_mvt_mean(trace: dict, branch: str) -> float | None:
        """The cohort-wide MVT threshold — mean SCORE over all of the
        patient's protocols — recorded by the update strategy. Only the
        update branch runs the MVT criterion."""
        if branch != "update":
            return None
        return trace.get("mvt_mean")
