"""`CDSS` — top-level orchestrator for one patient's weekly recommendation.

This file is intentionally small. All algorithm logic lives in the
`recommend/` subpackage, split by concern (mvt, substitute, bootstrap,
update, repeat_week, topup, trace). This class is the **glue**: it
inspects the patient's prior state, dispatches to the right branch
builder, runs the universal grid top-up, and attaches the structured
trace to the returned DataFrame.

Compared to the pre-refactor v0.3.1 `cdss.py` (588 lines), this file is
~80 lines and reads top-to-bottom as a description of the algorithm:

    if no prior          → bootstrap
    elif week skipped    → repeat
    else                 → update (MVT-driven swaps)
    [always]             → top-up to fill the 7×ppd grid
    [always]             → attach trace

The behavior is byte-for-byte identical to v0.3.1 — the existing test
suite passes unchanged.
"""
from __future__ import annotations

import pandas as pd

from ai_cdss.constants import N, N_DAYS, PROTOCOL_ID, PROTOCOLS_PER_DAY
from ai_cdss.recommend import (
    bootstrap,
    repeat_week,
    topup,
    update,
)
from ai_cdss.recommend.state import PatientState
from ai_cdss.recommend.trace import init_trace, serialize_final, serialize_prior


class CDSS:
    """Clinical Decision Support System.

    Recommends a 7-day × `protocols_per_day` schedule of rehab
    protocols for one patient at a time, using the patient's PPF +
    session history baked into `scoring`.

    Parameters
    ----------
    scoring
        DataFrame with one row per (patient, protocol). Required
        columns: PATIENT_ID, PROTOCOL_ID, SCORE, USAGE, USAGE_WEEK,
        DAYS. The presence of a non-empty DAYS list marks a protocol
        as "currently prescribed" (prior week).
    n
        How many distinct protocols to recommend per week. AISN trial
        = 12.
    days
        How many days in the schedule. AISN trial = 7.
    protocols_per_day
        How many protocols per day. AISN trial = 5.
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

    # ------------------------------------------------------------------

    def recommend(
        self,
        patient_id: int,
        protocol_similarity: pd.DataFrame,
    ) -> pd.DataFrame:
        """Run the full recommendation pipeline for `patient_id`.

        Attaches a structured `trace` dict to the returned DataFrame's
        `.attrs['trace']` capturing every decision: branch chosen,
        prior-week state, swap events, top-up events, final schedule.
        """
        patient = PatientState(self.scoring, patient_id)
        if not patient.has_data:
            raise ValueError(f"Patient {patient_id} has no data.")

        trace = init_trace(
            patient=patient,
            n=self.n,
            n_days=self.days,
            protocols_per_day=self.protocols_per_day,
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
        """Select the right branch and run it. Writes `trace["branch"]`
        and (for the update branch) `trace["prior"]`."""
        if patient.prescriptions.empty:
            trace["branch"] = "bootstrap"
            return bootstrap.build_recommendations(
                patient,
                n=self.n,
                n_days=self.days,
                protocols_per_day=self.protocols_per_day,
            )

        if patient.is_week_skipped():
            trace["branch"] = "repeat_skipped_week"
            return repeat_week.build_recommendations(patient)

        trace["branch"] = "update"
        trace["prior"] = serialize_prior(patient.prescriptions)
        return update.build_recommendations(
            patient, protocol_similarity, trace=trace,
        )

    # ------------------------------------------------------------------
    # Universal post-step.

    def _apply_topup(
        self,
        patient: PatientState,
        recommendations: pd.DataFrame,
        trace: dict,
    ) -> pd.DataFrame:
        """Fill any gap so every day reaches `protocols_per_day`
        protocols. Records each filler in `trace["topup"]`."""
        rows = recommendations.to_dict("records")
        rows = topup.fill_grid_coverage(
            patient,
            rows,
            n_days=self.days,
            protocols_per_day=self.protocols_per_day,
            n=self.n,
            trace=trace,
        )
        return pd.DataFrame(rows).sort_values(
            by=PROTOCOL_ID
        ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Finalize — write trace["final"] and attach to the output.

    def _finalize(
        self, recommendations: pd.DataFrame, trace: dict,
    ) -> pd.DataFrame:
        trace["final"] = serialize_final(recommendations)
        attrs = dict(self.scoring.attrs)
        attrs["trace"] = trace
        recommendations.attrs = attrs
        return recommendations
