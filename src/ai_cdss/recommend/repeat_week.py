"""Repeat-week branch — week was skipped, prior schedule repeats as-is.

When every prescribed row this week recorded zero sessions
(`USAGE_WEEK == 0`), the engine assumes the patient skipped the entire
week. Rather than reshuffle anything, it repeats the prior schedule.

Behavior preserved exactly from the v0.3.1 implementation in
`CDSS._repeat_prescriptions`.
"""
from __future__ import annotations

import logging

import pandas as pd

from ai_cdss.constants import PATIENT_ID
from ai_cdss.recommend.state import PatientState

logger = logging.getLogger(__name__)


def build_recommendations(patient: PatientState) -> pd.DataFrame:
    """Return a copy of the patient's prior prescriptions, unchanged."""
    if patient.prescriptions.empty:
        logger.info("repeat_week called with empty prescriptions for patient=%s",
                    patient.patient_id)
        df = patient.prescriptions.copy()
    else:
        logger.info("Patient %s, skipped the whole week, cdss repeating prescriptions.",
                    patient.patient_id)
        df = patient.prescriptions.copy()
    df.attrs = patient.rows.attrs
    return df
