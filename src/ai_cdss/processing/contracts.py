"""Typed contracts between processing-pipeline stages.

The data pipeline is a chain of DataFrame transformations. Without
explicit contracts, each stage takes `pd.DataFrame` and trusts the
caller to have populated the right columns. That's the root of the
opacity problem: the column expectations are buried in method bodies
and constants.

This module defines a small frozen dataclass per pipeline boundary.
Each dataclass:
  1. wraps the DataFrame for that stage,
  2. lists the columns it must carry,
  3. validates its contract on construction (when `validate=True`).

A stage's public signature becomes self-documenting:

    def build_features(inputs: PreparedInputs, ...) -> ProtocolFeatures:
        ...

Compare to the v0.3.1 form:

    def _build_features(self, patient_data, session_data, ppf_data, ...):
        ...   # callers had to read the body to know the column contract

These dataclasses do NOT change the DataFrame's identity — they're thin
wrappers around the same frames. Mutating `obj.df` mutates the
underlying frame. The wrapper is for documentation + validation, not
isolation.

Each contract has a `required` class attribute listing the minimum
columns. Stages may add more columns; `required` is the floor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Iterable

import pandas as pd

from ai_cdss.constants import (
    ADHERENCE,
    BY_PP,
    BY_PPS,
    CLINICAL_END,
    CLINICAL_START,
    CONTRIB,
    DAYS,
    DELTA_DM,
    DM_VALUE,
    PATIENT_ID,
    PPF,
    PROTOCOL_ID,
    RECENT_ADHERENCE,
    SCORE,
    SESSION_DATE,
    SESSION_INDEX,
    STATUS,
    USAGE,
    USAGE_WEEK,
    WEEKS_SINCE_START,
)


class ContractError(ValueError):
    """Raised when a frame doesn't carry its declared required columns."""


def _validate_columns(
    df: pd.DataFrame, required: Iterable[str], stage_name: str
) -> None:
    """Raise ContractError if any required column is missing."""
    required = list(required)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ContractError(
            f"{stage_name}: missing required columns {missing}. "
            f"Got: {list(df.columns)}"
        )


# ---------------------------------------------------------------------------
# Stage 1 — Pipeline entry: three cleaned input frames.

@dataclass(frozen=True)
class PreparedInputs:
    """Cleaned input frames at the start of the pipeline.

    `patient`  : one row per patient — anchors clinical window.
    `session`  : one row per (patient, protocol, session) — raw signal,
                 windowed to [CLINICAL_START, min(CLINICAL_END, scoring_date)].
                 Includes prescription metadata via the prescription_plus
                 LEFT JOIN done upstream (see DataLoader).
    `ppf`      : one row per (patient, protocol) the patient is eligible
                 for — the patient's PPF cohort. Drives the env-wide
                 "all available protocols" set.
    """
    patient: pd.DataFrame
    session: pd.DataFrame
    ppf:     pd.DataFrame
    validate_on_init: bool = field(default=True)

    PATIENT_REQUIRED: ClassVar[list[str]] = [PATIENT_ID, CLINICAL_START, CLINICAL_END]
    SESSION_REQUIRED: ClassVar[list[str]] = [PATIENT_ID, PROTOCOL_ID, SESSION_DATE]
    PPF_REQUIRED:     ClassVar[list[str]] = [PATIENT_ID, PROTOCOL_ID, PPF]

    def __post_init__(self) -> None:
        if not self.validate_on_init:
            return
        _validate_columns(self.patient, self.PATIENT_REQUIRED, "PreparedInputs.patient")
        _validate_columns(self.session, self.SESSION_REQUIRED, "PreparedInputs.session")
        _validate_columns(self.ppf,     self.PPF_REQUIRED,     "PreparedInputs.ppf")

    @property
    def has_sessions(self) -> bool:
        """True iff at least one session row survives the date window."""
        return not self.session.empty


# ---------------------------------------------------------------------------
# Stage 2a — Session-level features (one row per (PP, session_date)).

@dataclass(frozen=True)
class SessionLevelFeatures:
    """Per-session features keyed on (patient, protocol, session_date).

    Built by `FeatureBuilder.build_recent_adherence` + `build_delta_dm`
    and merged. Each session has its own RECENT_ADHERENCE (rolling
    window) and DELTA_DM (first-order difference) values.
    """
    df: pd.DataFrame
    validate_on_init: bool = field(default=True)

    REQUIRED: ClassVar[list[str]] = BY_PP + [SESSION_DATE, RECENT_ADHERENCE, DELTA_DM]

    def __post_init__(self) -> None:
        if self.validate_on_init:
            _validate_columns(self.df, self.REQUIRED, "SessionLevelFeatures")


# ---------------------------------------------------------------------------
# Stage 2b — Protocol-level features (one row per (PP)).

@dataclass(frozen=True)
class ProtocolLevelFeatures:
    """Per-protocol features keyed on (patient, protocol).

    Built by joining PPF rows with `build_usage`, `build_week_usage`,
    `build_prescription_days`, then `build_week_since_start`. One row
    per patient-protocol-pair, regardless of session history.

    The presence of PPF here is what defines the patient's PPF cohort —
    only protocols with a PPF row reach scoring.
    """
    df: pd.DataFrame
    validate_on_init: bool = field(default=True)

    REQUIRED: ClassVar[list[str]] = BY_PP + [
        PPF, USAGE, USAGE_WEEK, DAYS, WEEKS_SINCE_START,
    ]

    def __post_init__(self) -> None:
        if self.validate_on_init:
            _validate_columns(self.df, self.REQUIRED, "ProtocolLevelFeatures")


# ---------------------------------------------------------------------------
# Stage 2c — Merged features (session × protocol).

@dataclass(frozen=True)
class MergedFeatures:
    """Session-level rows broadcast against protocol-level metadata.

    One row per (patient, protocol, session_date). Each row carries
    everything the imputer needs: session-level DELTA_DM /
    RECENT_ADHERENCE plus protocol-level USAGE / DAYS / WEEKS_SINCE_START.
    """
    df: pd.DataFrame
    validate_on_init: bool = field(default=True)

    REQUIRED: ClassVar[list[str]] = (
        ProtocolLevelFeatures.REQUIRED + [SESSION_DATE, RECENT_ADHERENCE, DELTA_DM]
    )

    def __post_init__(self) -> None:
        if self.validate_on_init:
            _validate_columns(self.df, self.REQUIRED, "MergedFeatures")


# ---------------------------------------------------------------------------
# Stage 3 — Imputed scoring input (one row per (PP), all metrics ready).

@dataclass(frozen=True)
class ScoringInput:
    """One row per (patient, protocol) ready for the Scorer.

    Built by:
      * `groupby(BY_PP).agg('last')` to collapse session rows to
        one-per-protocol (taking the most recent session's values),
      * imputing missing DELTA_DM / RECENT_ADHERENCE per-patient median,
      * `Imputer.init_metrics` to seed defaults for missing PPF
        components.
    """
    df: pd.DataFrame
    validate_on_init: bool = field(default=True)

    REQUIRED: ClassVar[list[str]] = BY_PP + [
        PPF, DELTA_DM, RECENT_ADHERENCE, USAGE, USAGE_WEEK, DAYS, WEEKS_SINCE_START,
    ]

    def __post_init__(self) -> None:
        if self.validate_on_init:
            _validate_columns(self.df, self.REQUIRED, "ScoringInput")


# ---------------------------------------------------------------------------
# Stage 4 — Scoring output (one row per (PP), with SCORE).

@dataclass(frozen=True)
class ScoringOutput:
    """Final scored output — one row per (patient, protocol) with SCORE.

    Carries the same columns as `ScoringInput` plus SCORE. This is what
    `DataProcessor.process_data` returns; CDSS.recommend consumes it.
    """
    df: pd.DataFrame
    validate_on_init: bool = field(default=True)

    REQUIRED: ClassVar[list[str]] = BY_PP + [
        PPF, DELTA_DM, RECENT_ADHERENCE, USAGE, USAGE_WEEK, DAYS,
        WEEKS_SINCE_START, SCORE,
    ]

    def __post_init__(self) -> None:
        if self.validate_on_init:
            _validate_columns(self.df, self.REQUIRED, "ScoringOutput")

    @property
    def attrs(self) -> dict[str, Any]:
        """Pass-through to the underlying frame's attrs (used to carry
        subscale metadata + the eventual recommendation trace)."""
        return self.df.attrs
