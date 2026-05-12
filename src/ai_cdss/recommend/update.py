"""Update branch — swap underperforming protocols in an existing schedule.

When the patient has a prior week (`_get_prescriptions` non-empty) and
the week was NOT entirely skipped, the engine runs the update branch:

  1. Identify swap candidates via MVT — every prescribed protocol with
     SCORE < mean(prescribed SCORE). See `mvt.below_mean_protocols`.
  2. If no candidate is below the mean, force a single swap on the
     lowest-scoring protocol (AISN RCT min-1-swap rule).
  3. For each swap candidate, search for a substitute via the two-tier
     algorithm in `substitute.find_substitute`. If found, the substitute
     inherits the removed protocol's DAYS. If not found, the removed
     protocol stays (legacy v0.3.1 "Else return same protocol" fallback —
     surfaced explicitly here so the behavior is auditable).
  4. Kept-untouched + swapped rows form the recommendation set.
     The universal top-up step (run from `cdss.recommend`) then fills
     the 7×ppd grid.

Behavior preserved exactly from the v0.3.1 implementation in
`CDSS._update_existing_recommendations` + `_swap_protocol`.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ai_cdss.constants import DAYS, PATIENT_ID, PROTOCOL_B, PROTOCOL_ID, SCORE, SIMILARITY
from ai_cdss.recommend import mvt, similarity as sim, substitute
from ai_cdss.recommend.state import PatientState

logger = logging.getLogger(__name__)


def build_recommendations(
    patient: PatientState,
    similarity_table: pd.DataFrame,
    *,
    trace: dict | None = None,
) -> pd.DataFrame:
    """Build the next week's recommendation set by swapping below-MVT
    protocols out of the prior week.

    Parameters
    ----------
    patient
        Patient-scoped view of the scoring DataFrame.
    similarity_table
        The full protocol-similarity DataFrame.
    trace
        Optional trace dict; when provided, each swap event is appended
        to `trace["swaps"]`.
    """
    prior = patient.prescriptions
    swap_targets, reasons = _select_swap_targets(patient)

    kept_rows = _rows_kept_unchanged(prior, swap_targets)
    swapped_rows = _build_swap_rows(
        patient=patient,
        prior=prior,
        similarity_table=similarity_table,
        swap_targets=swap_targets,
        reasons=reasons,
        trace=trace,
    )

    combined = pd.DataFrame(kept_rows + swapped_rows).sort_values(
        by=PROTOCOL_ID
    ).reset_index(drop=True)
    combined.attrs = patient.rows.attrs
    return combined


# ---------------------------------------------------------------------------
# Step 1 — pick swap targets + record the reason for each.

def _select_swap_targets(
    patient: PatientState,
) -> tuple[list[int], dict[int, str]]:
    """Return `(targets, reason_by_protocol)`.

    `targets` is the ordered list of protocols to swap (DataFrame-row
    order, as v0.3.1). `reason_by_protocol` maps each target to one of:
      * "below_mean_score" — fell under the MVT mean.
      * "aisn_min_one_swap" — fallback to satisfy AISN RCT.
    """
    targets = mvt.below_mean_protocols(patient.prescriptions)
    reasons = {p: "below_mean_score" for p in targets}

    if not targets:
        forced = patient.lowest_scoring_prescribed
        targets = [forced]
        reasons = {forced: "aisn_min_one_swap"}
    return targets, reasons


# ---------------------------------------------------------------------------
# Step 2 — pass through prescribed rows we're keeping.

def _rows_kept_unchanged(
    prior: pd.DataFrame, swap_targets: list[int]
) -> list[dict[str, Any]]:
    """Prescribed rows whose protocol is NOT in `swap_targets` — kept
    verbatim (DAYS preserved)."""
    keep_mask = ~prior[PROTOCOL_ID].isin(swap_targets)
    return prior.loc[keep_mask].to_dict("records")


# ---------------------------------------------------------------------------
# Step 3 — for each swap target, run substitute search + record trace event.

def _build_swap_rows(
    *,
    patient: PatientState,
    prior: pd.DataFrame,
    similarity_table: pd.DataFrame,
    swap_targets: list[int],
    reasons: dict[int, str],
    trace: dict | None,
) -> list[dict[str, Any]]:
    """Run the swap loop and return the new rows (substitute or
    same-protocol fallback) for each target.

    `protocols_excluded` grows as we pick substitutes — each chosen
    substitute is excluded from later searches so the pool depletes
    deterministically.
    """
    excluded = prior[PROTOCOL_ID].tolist()
    swapped: list[dict[str, Any]] = []

    for removed_id in swap_targets:
        sims = sim.similarities_for(removed_id, similarity_table, excluded)
        result = substitute.find_substitute(
            usage=patient.usage,
            similarities=sims,
            removed_protocol_id=removed_id,
        )

        new_row = _materialize_swap_row(
            patient=patient,
            prior=prior,
            removed_id=removed_id,
            substitute_id=result.protocol_id,
        )
        new_id = int(new_row[PROTOCOL_ID])

        if trace is not None:
            trace["swaps"].append(
                _trace_event(
                    removed_id=removed_id,
                    new_id=new_id,
                    prior=prior,
                    sims=sims,
                    new_row=new_row,
                    reason=reasons.get(removed_id, "unknown"),
                )
            )

        swapped.append(new_row)
        excluded.append(new_id)
    return swapped


# ---------------------------------------------------------------------------
# Step 3a — assemble the row that replaces a removed protocol.

def _materialize_swap_row(
    *,
    patient: PatientState,
    prior: pd.DataFrame,
    removed_id: int,
    substitute_id: int | None,
) -> dict[str, Any]:
    """Build the row that takes `removed_id`'s slot.

    If a substitute was found, it inherits the removed protocol's DAYS
    verbatim (the substitute's own DAYS column may be empty since it
    wasn't prescribed). If `substitute_id` is None (search exhausted),
    we fall back to the removed protocol itself — legacy v0.3.1
    behavior. This fallback is the source of the `swap_returned_same_protocol`
    backtest warning when the candidate pool starves.
    """
    if substitute_id is None:
        return patient.score_row(removed_id)

    new_row = patient.score_row(substitute_id)
    new_row[DAYS] = prior.loc[
        prior[PROTOCOL_ID] == removed_id, DAYS
    ].values[0]
    new_row[PROTOCOL_ID] = substitute_id
    new_row[PATIENT_ID] = patient.patient_id
    return new_row


# ---------------------------------------------------------------------------
# Step 3b — emit a structured trace entry per swap.

def _trace_event(
    *,
    removed_id: int,
    new_id: int,
    prior: pd.DataFrame,
    sims: pd.DataFrame,
    new_row: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    """Build the `{removed, removed_score, added, similarity,
    inherited_days, candidate_pool, reason}` entry for `trace["swaps"]`.

    Also emits the v0.3.1 info-level log line so existing log consumers
    don't break.
    """
    removed_score = _score_of(prior, removed_id)
    sim_row = sims.loc[sims[PROTOCOL_B] == new_id]
    similarity_value = float(sim_row[SIMILARITY].iloc[0]) if not sim_row.empty else None
    inherited_days = sorted(int(d) for d in (new_row.get(DAYS) or []))

    logger.info(
        "Swap patient=%s removed=%s (score=%s) -> added=%s (sim=%s) days=%s reason=%s",
        new_row.get(PATIENT_ID), removed_id, removed_score,
        new_id, similarity_value, inherited_days, reason,
    )
    return {
        "removed":        int(removed_id),
        "removed_score":  removed_score,
        "added":          new_id,
        "similarity":     similarity_value,
        "inherited_days": inherited_days,
        "candidate_pool": sims[PROTOCOL_B].astype(int).tolist(),
        "reason":         reason,
    }


def _score_of(prior: pd.DataFrame, protocol_id: int) -> float | None:
    """SCORE of a protocol in the prior set, or None if the column is
    missing (rare; defensive for older test fixtures)."""
    if SCORE not in prior.columns:
        return None
    match = prior.loc[prior[PROTOCOL_ID] == protocol_id, SCORE]
    return float(match.iloc[0]) if not match.empty else None
