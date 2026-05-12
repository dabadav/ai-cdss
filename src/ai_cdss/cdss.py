"""Backward-compat re-export.

The `CDSS` class and `PatientState` view live in `ai_cdss.recommend`
after the readability refactor. This module re-exports them so the
public API `from ai_cdss.cdss import CDSS` keeps working unchanged.
"""
from __future__ import annotations

from ai_cdss.recommend import CDSS, PatientState, SubstituteResult

__all__ = ["CDSS", "PatientState", "SubstituteResult"]
