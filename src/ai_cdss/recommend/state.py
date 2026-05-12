"""`PatientState` — patient-scoped view of the scoring DataFrame.

The scoring DataFrame produced by `DataProcessor.process_data` has one
row per (patient, protocol) for every patient in the cohort. The CDSS
engine only ever cares about ONE patient at a time. Wrapping that
single-patient slice in a small object lets every downstream helper
take a `PatientState` instead of `(scoring, patient_id)` everywhere —
fewer arguments to thread, intent obvious at the call site.

The class is **read-only** — it does not mutate `scoring`. Methods
return new DataFrames / lists / scalars.

Behavior preserved exactly from the v0.3.1 implementations in
`CDSS._get_top_protocols`, `_get_prescriptions`,
`_get_patient_protocol_usage`, `_get_lowest_performing_protocol`,
`_get_scores`, `_has_patient_data`, `_is_week_skipped`.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from ai_cdss.constants import (
    DAYS,
    PATIENT_ID,
    PROTOCOL_ID,
    SCORE,
    USAGE,
    USAGE_WEEK,
)


class PatientState:
    """All scoring data for a single patient.

    Parameters
    ----------
    scoring : pd.DataFrame
        The full scoring DataFrame (multiple patients).
    patient_id : int
        The patient we are recommending for.

    Attributes
    ----------
    patient_id : int
    rows : pd.DataFrame
        The slice of `scoring` for this patient. Empty if the patient
        has no row.
    """

    def __init__(self, scoring: pd.DataFrame, patient_id: int) -> None:
        self._scoring = scoring
        self.patient_id = patient_id
        self.rows = scoring.loc[scoring[PATIENT_ID] == patient_id]

    # ------------------------------------------------------------------
    # Existence

    @property
    def has_data(self) -> bool:
        """True iff the patient has at least one scoring row."""
        return not self.rows.empty

    # ------------------------------------------------------------------
    # Prescriptions (currently-prescribed protocols — DAYS non-empty)

    @property
    def prescriptions(self) -> pd.DataFrame:
        """Subset of `rows` that the engine treats as 'currently
        prescribed' — rows with a non-empty DAYS list.

        Empty DAYS means the protocol is in the patient's PPF cohort
        but is not on this week's schedule.
        """
        has_days = self.rows[DAYS].apply(
            lambda d: isinstance(d, list) and len(d) > 0
        )
        return self.rows.loc[has_days]

    def is_week_skipped(self) -> bool:
        """True iff every scheduled prescription this week recorded
        zero sessions (USAGE_WEEK == 0).

        Mirrors the v0.3.1 semantics: only counts rows that actually
        have days scheduled; if nothing is scheduled, the week is NOT
        considered skipped (returns False).
        """
        scheduled = self.prescriptions
        if scheduled.empty:
            return False
        return bool((scheduled[USAGE_WEEK] == 0).all())

    # ------------------------------------------------------------------
    # Score-based queries

    def top_protocols(self, n: int) -> list[int]:
        """Top N protocols by SCORE."""
        return self.rows.nlargest(n, SCORE)[PROTOCOL_ID].tolist()

    @property
    def lowest_scoring_prescribed(self) -> int:
        """Protocol with the lowest SCORE among currently-prescribed.

        Used to satisfy the AISN min-1-swap rule when no protocol falls
        below the swap threshold.
        """
        pres = self.prescriptions
        return int(pres.loc[pres[SCORE].idxmin(), PROTOCOL_ID])

    # ------------------------------------------------------------------
    # Usage

    @property
    def usage(self) -> pd.Series:
        """Per-protocol usage count, indexed by PROTOCOL_ID."""
        return self.rows.set_index(PROTOCOL_ID)[USAGE]

    # ------------------------------------------------------------------
    # Single-row lookup

    def score_row(self, protocol_id: int) -> dict[str, Any]:
        """Return the scoring row for `(patient_id, protocol_id)` as a
        plain dict — used to seed a recommendation/substitute row.

        Raises IndexError if the protocol isn't in the patient's
        scoring rows (PPF cohort).
        """
        match = self.rows.loc[self.rows[PROTOCOL_ID] == protocol_id]
        return match.iloc[0].to_dict()
