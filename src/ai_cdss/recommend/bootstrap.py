"""Bootstrap branch — first-ever schedule for a patient with no prior.

When `_get_prescriptions` is empty (no row has any DAYS), the engine
takes the *bootstrap* path: pick the patient's top-N protocols by SCORE
and round-robin them across the 7-day week.

Behavior preserved exactly from the v0.3.1 implementation in
`CDSS._generate_new_recommendations`.
"""
from __future__ import annotations

import pandas as pd

from ai_cdss.constants import DAYS, PATIENT_ID, PROTOCOL_ID
from ai_cdss.recommend.schedule import round_robin_across_days
from ai_cdss.recommend.state import PatientState


def build_recommendations(
    patient: PatientState,
    *,
    n: int,
    n_days: int,
    protocols_per_day: int,
) -> pd.DataFrame:
    """Top-N protocols by SCORE laid out across the week.

    Returns a DataFrame ready to feed into the universal top-up step.
    The DataFrame's `.attrs` are pre-populated from `patient` so any
    subscale metadata propagates.
    """
    top_protocols = patient.top_protocols(n)
    schedule_by_day = round_robin_across_days(
        top_protocols,
        n_days=n_days,
        protocols_per_day=protocols_per_day,
    )
    rows_by_protocol = _seed_rows_from_schedule(patient, schedule_by_day)

    df = pd.DataFrame(rows_by_protocol.values()).sort_values(
        by=PROTOCOL_ID
    ).reset_index(drop=True)
    df.attrs = patient.rows.attrs
    return df


# ---------------------------------------------------------------------------
# Internal — small helpers so `build_recommendations` reads as prose.

def _seed_rows_from_schedule(
    patient: PatientState,
    schedule_by_day: dict[int, list[int]],
) -> dict[int, dict]:
    """Walk the per-day schedule and accumulate one row per protocol.

    Each row starts from the patient's scoring row (preserves SCORE,
    PPF, USAGE etc.) and the DAYS list is built up as we encounter the
    protocol on additional days.
    """
    rows: dict[int, dict] = {}
    for day, protocol_ids in schedule_by_day.items():
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
