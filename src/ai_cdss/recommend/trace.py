"""Trace-dict construction helpers.

The `CDSS.recommend` method builds a `trace` dict that captures every
decision so a downstream consumer can reconstruct the run from the
JSON alone. These helpers shape that dict — they DO NOT carry algorithm
logic.

The dict layout (preserved from v0.3.1):

    {
      "patient_id":    int,
      "config":        {"n": ..., "days": ..., "protocols_per_day": ...},
      "top_protocols": [int, ...],            # top-N by SCORE
      "branch":        "bootstrap" | "repeat_skipped_week" | "update",
      "prior":         [ {protocol_id, days, score, usage_week}, ... ],
      "swaps":         [ {removed, added, similarity, ...}, ... ],
      "topup":         [ {day, protocol_id, source}, ... ],
      "final":         [ {protocol_id, days}, ... ],
    }
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from ai_cdss.constants import DAYS, PROTOCOL_ID, SCORE, USAGE_WEEK
from ai_cdss.recommend.state import PatientState


def init_trace(
    *,
    patient: PatientState,
    n: int,
    n_days: int,
    protocols_per_day: int,
) -> dict[str, Any]:
    """Build the empty trace skeleton with the metadata fields filled."""
    return {
        "patient_id":     patient.patient_id,
        "config":         {
            "n":                 n,
            "days":              n_days,
            "protocols_per_day": protocols_per_day,
        },
        "top_protocols":  patient.top_protocols(n),
        "branch":         None,
        "prior":          [],
        "swaps":          [],
        "topup":          [],
        "final":          [],
    }


def serialize_prior(prescriptions: pd.DataFrame) -> list[dict[str, Any]]:
    """Render the prior-week prescriptions in the trace format."""
    return [
        {
            "protocol_id": int(row[PROTOCOL_ID]),
            "days":        list(row[DAYS]) if isinstance(row[DAYS], list) else [],
            "score":       _safe_float(row.get(SCORE)),
            "usage_week":  _safe_int(row.get(USAGE_WEEK)),
        }
        for _, row in prescriptions.iterrows()
    ]


def serialize_final(recommendations: pd.DataFrame) -> list[dict[str, Any]]:
    """Render the final schedule in the trace format."""
    return [
        {
            "protocol_id": int(row[PROTOCOL_ID]),
            "days":        sorted(int(d) for d in (row.get(DAYS) or [])),
        }
        for _, row in recommendations.iterrows()
    ]


# ---------------------------------------------------------------------------
# Small NaN-safe coercions so the trace JSON has clean types.

def _safe_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return float(value) if pd.notna(value) else None


def _safe_int(value: Any) -> int:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0
    return int(value) if pd.notna(value) else 0
