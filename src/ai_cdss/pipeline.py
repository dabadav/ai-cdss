"""Data pipeline — feature-build → impute → score, in that order.

Turns a `Cohort` (sessions / patient metadata / PPF cohort, supplied
by `MySQLCohortRepository` or any other `CohortRepository`
implementation) into a one-row-per-(patient, protocol) scoring
DataFrame that `recommend.CDSS.recommend` consumes.

The flow (each arrow is a typed contract — see SECTION 1 below):

    PreparedInputs                    Stage 1 — cleaned, windowed
        ├── SessionLevelFeatures      Stage 2a — per-session features
        ├── ProtocolLevelFeatures     Stage 2b — per-protocol features
        └── MergedFeatures            Stage 2c — session × protocol
            └── ScoringInput          Stage 3 — imputed, one-per-PP
                └── ScoringOutput     Stage 4 — final SCORE

This file is sectioned by the algorithm's logical phases:

    SECTION 1  Typed contracts (PreparedInputs / SessionLevelFeatures /
               ProtocolLevelFeatures / MergedFeatures / ScoringInput /
               ScoringOutput) — column expectations declared inline so
               you never have to grep constants to know what each stage
               expects.
    SECTION 2  get_nth — generic helper used by the imputer for first/
               last per-group lookups.
    SECTION 3  DataPipeline — orchestrator class; one method per stage.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import reduce
from typing import Any, ClassVar, Iterable

import pandas as pd
from pandas import Timestamp

from ai_cdss.constants import (
    BY_PP,
    BY_PPS,
    CLINICAL_END,
    CLINICAL_START,
    DAYS,
    DELTA_DM,
    DM_VALUE,
    FINAL_METRICS,
    PATIENT_ID,
    PPF,
    PROTOCOL_ID,
    RECENT_ADHERENCE,
    SCORE,
    SESSION_DATE,
    SESSION_INDEX,
    USAGE,
    USAGE_WEEK,
    WEEKS_SINCE_START,
)
from ai_cdss.data import Cohort
from ai_cdss.feature import (
    build_delta_dm,
    build_prescription_days,
    build_recent_adherence,
    build_usage,
    build_week_since_start,
    build_week_usage,
    include_missing_sessions,
)
from ai_cdss.score import Imputer, Scorer

logger = logging.getLogger(__name__)


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1 — Typed contracts                                         ║
# ║                                                                      ║
# ║  Every DataFrame that crosses a stage boundary is wrapped in a       ║
# ║  small frozen dataclass that lists its required columns. Each        ║
# ║  contract validates on construction (skippable via                   ║
# ║  validate_on_init=False for hot-path).                               ║
# ║                                                                      ║
# ║  The contracts replace the v0.3.1 positional-tuple returns           ║
# ║  (patient_data, session_data, ppf_data) and the implicit column      ║
# ║  contracts buried in method bodies.                                  ║
# ╚═════════════════════════════════════════════════════════════════════╝

class ContractError(ValueError):
    """Raised when a frame doesn't carry its declared required columns."""


def _validate_columns(
    df: pd.DataFrame, required: Iterable[str], stage_name: str,
) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ContractError(
            f"{stage_name}: missing required columns {missing}. "
            f"Got: {list(df.columns)}"
        )


@dataclass(frozen=True)
class PreparedInputs:
    """Three cleaned input frames at pipeline entry.

    `patient`  one row per patient — anchors clinical window.
    `session`  one row per (patient, protocol, session) — windowed to
               [CLINICAL_START, min(CLINICAL_END, scoring_date)].
    `ppf`      one row per (patient, protocol) — patient's PPF cohort,
               the env-wide alternative set.
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


@dataclass(frozen=True)
class SessionLevelFeatures:
    """One row per (patient, protocol, session_date). Carries
    per-session RECENT_ADHERENCE and DELTA_DM."""
    df: pd.DataFrame
    validate_on_init: bool = field(default=True)

    REQUIRED: ClassVar[list[str]] = BY_PP + [SESSION_DATE, RECENT_ADHERENCE, DELTA_DM]

    def __post_init__(self) -> None:
        if self.validate_on_init:
            _validate_columns(self.df, self.REQUIRED, "SessionLevelFeatures")


@dataclass(frozen=True)
class ProtocolLevelFeatures:
    """One row per (patient, protocol). Per-protocol metadata
    (PPF + USAGE + USAGE_WEEK + DAYS + WEEKS_SINCE_START)."""
    df: pd.DataFrame
    validate_on_init: bool = field(default=True)

    REQUIRED: ClassVar[list[str]] = BY_PP + [
        PPF, USAGE, USAGE_WEEK, DAYS, WEEKS_SINCE_START,
    ]

    def __post_init__(self) -> None:
        if self.validate_on_init:
            _validate_columns(self.df, self.REQUIRED, "ProtocolLevelFeatures")


@dataclass(frozen=True)
class MergedFeatures:
    """Session-level rows broadcast against protocol-level metadata.
    One row per (patient, protocol, session_date), all columns."""
    df: pd.DataFrame
    validate_on_init: bool = field(default=True)

    REQUIRED: ClassVar[list[str]] = (
        ProtocolLevelFeatures.REQUIRED + [SESSION_DATE, RECENT_ADHERENCE, DELTA_DM]
    )

    def __post_init__(self) -> None:
        if self.validate_on_init:
            _validate_columns(self.df, self.REQUIRED, "MergedFeatures")


@dataclass(frozen=True)
class ScoringInput:
    """One row per (patient, protocol), all metrics imputed. Ready for
    the Scorer."""
    df: pd.DataFrame
    validate_on_init: bool = field(default=True)

    REQUIRED: ClassVar[list[str]] = BY_PP + [
        PPF, DELTA_DM, RECENT_ADHERENCE, USAGE, USAGE_WEEK, DAYS, WEEKS_SINCE_START,
    ]

    def __post_init__(self) -> None:
        if self.validate_on_init:
            _validate_columns(self.df, self.REQUIRED, "ScoringInput")


@dataclass(frozen=True)
class ScoringOutput:
    """Final scored output — one row per (patient, protocol) with SCORE."""
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
        return self.df.attrs


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2 — get_nth helper                                          ║
# ║                                                                      ║
# ║  Generic 'first/last value per group' lookup used by the imputer.    ║
# ║  Lives here (not in feature.py) because only the pipeline stages     ║
# ║  call it.                                                            ║
# ╚═════════════════════════════════════════════════════════════════════╝

def get_nth(
    df: pd.DataFrame, col: str,
    groupby_col: str | list[str], session_index_col: str, n: int,
) -> pd.DataFrame:
    """Return the nth row per group, sorted by `session_index_col`.

    `n` can be negative (e.g. `-1` = last). Returns BY_PP +
    [session_index_col, col], NaN rows dropped.
    """
    sorted_df = df.sort_values(by=[session_index_col])
    nth = sorted_df.groupby(groupby_col).nth(n)
    return nth[BY_PP + [session_index_col, col]].dropna()


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3 — DataPipeline (orchestrator)                             ║
# ║                                                                      ║
# ║  One method per stage. `process` is the public entry; everything     ║
# ║  underscore-prefixed is internal.                                    ║
# ╚═════════════════════════════════════════════════════════════════════╝

class DataPipeline:
    """End-to-end pipeline from raw DataLoader output to scored frame."""

    def __init__(
        self,
        imputer: Imputer | None = None,
        scorer:  Scorer  | None = None,
    ) -> None:
        self.imputer = imputer or Imputer()
        self.scorer = scorer or Scorer()

    # ------------------------------------------------------------------
    # Public entry.

    def process(
        self, cohort: "Cohort", scoring_date: Timestamp,
    ) -> pd.DataFrame:
        """Run the pipeline and return the scored DataFrame.

        Consumes the three frames the pipeline cares about (`patient`,
        `session`, `ppf`) off the `Cohort` — the bundle also carries
        `similarity` / `whitelist` / `missing_ppf`, which the engine
        consumes downstream, not us.
        """
        inputs = self._prepare(cohort, scoring_date)

        if not inputs.has_sessions:
            logger.info("Bootstrapping system, no session data available for patients.")
            scoring_input = self._bootstrap_scoring_input(inputs)
        else:
            features = self._build_features(inputs, scoring_date)
            scoring_input = self._impute_features(features)

        return self._score(scoring_input, inputs).df

    # ==================================================================
    # Stage 1 — prepare: clean and window the inputs.

    def _prepare(
        self, cohort: "Cohort", scoring_date: Timestamp,
    ) -> PreparedInputs:
        """Attach clinical window to sessions, clamp session_date to
        [CLINICAL_START, min(CLINICAL_END, scoring_date)]."""
        session = include_missing_sessions(cohort.session)
        session = self._attach_clinical_window(session, cohort.patient)
        session = self._clamp_to_window(session, scoring_date)

        return PreparedInputs(
            patient=cohort.patient, session=session, ppf=cohort.ppf,
            validate_on_init=False,
        )

    def _attach_clinical_window(
        self, session: pd.DataFrame, patient: pd.DataFrame,
    ) -> pd.DataFrame:
        """Left-join CLINICAL_START + CLINICAL_END onto each session row
        and normalize those dates."""
        session = session.merge(
            patient[[PATIENT_ID, CLINICAL_START, CLINICAL_END]],
            on=PATIENT_ID, how="left",
        )
        for col in (CLINICAL_START, CLINICAL_END):
            if not pd.api.types.is_datetime64_any_dtype(session[col]):
                session[col] = pd.to_datetime(session[col], errors="coerce")
            session[col] = session[col].dt.normalize()
        return session

    def _clamp_to_window(
        self, session: pd.DataFrame, scoring_date: Timestamp,
    ) -> pd.DataFrame:
        """Drop sessions outside [CLINICAL_START, min(CLINICAL_END,
        scoring_date)]."""
        upper_bound = session[CLINICAL_END].where(
            session[CLINICAL_END] < scoring_date, scoring_date
        )
        in_window = (
            (session[SESSION_DATE] >= session[CLINICAL_START])
            & (session[SESSION_DATE] <= upper_bound)
        )
        return session.loc[in_window]

    # ==================================================================
    # Stage 2 — build features (session-level, protocol-level, merge).

    def _build_features(
        self, inputs: PreparedInputs, scoring_date: Timestamp,
    ) -> MergedFeatures:
        """Run the three feature substages and merge."""
        session_features  = self._session_level_features(inputs.session)
        protocol_features = self._protocol_level_features(inputs, scoring_date)
        return self._broadcast_session_onto_protocol(session_features, protocol_features)

    def _session_level_features(
        self, session: pd.DataFrame,
    ) -> SessionLevelFeatures:
        """RECENT_ADHERENCE + DELTA_DM, keyed on (PP, session_date)."""
        adherence_df = build_recent_adherence(session)
        dm_rows = session[BY_PPS + [SESSION_DATE, DM_VALUE]].dropna()
        delta_df = build_delta_dm(dm_rows)
        merged = pd.merge(
            adherence_df, delta_df, on=BY_PP + [SESSION_DATE], how="left",
        )
        return SessionLevelFeatures(df=merged, validate_on_init=False)

    def _protocol_level_features(
        self, inputs: PreparedInputs, scoring_date: Timestamp,
    ) -> ProtocolLevelFeatures:
        """USAGE / USAGE_WEEK / DAYS plus patient-broadcast WEEKS_SINCE_START."""
        per_protocol_frames = [
            inputs.ppf,
            build_usage(inputs.session),
            build_week_usage(
                inputs.session, inputs.patient, scoring_date,
            ),
            build_prescription_days(
                inputs.session, inputs.patient, scoring_date,
            ),
        ]
        merged = reduce(
            lambda left, right: pd.merge(left, right, on=BY_PP, how="left"),
            per_protocol_frames,
        )
        weeks_since_start = build_week_since_start(
            inputs.patient, scoring_date,
        )
        merged = pd.merge(merged, weeks_since_start, on=PATIENT_ID, how="left")
        return ProtocolLevelFeatures(df=merged, validate_on_init=False)

    def _broadcast_session_onto_protocol(
        self,
        session_level: SessionLevelFeatures,
        protocol_level: ProtocolLevelFeatures,
    ) -> MergedFeatures:
        """Broadcast every per-(PP) protocol row across every per-session
        row. Sorted by (PP, session_date) for the downstream groupby."""
        merged = pd.merge(
            protocol_level.df, session_level.df, on=BY_PP, how="left",
        ).sort_values(by=BY_PP + [SESSION_DATE])
        return MergedFeatures(df=merged, validate_on_init=False)

    # ==================================================================
    # Stage 3 — impute missing values.

    def _impute_features(self, features: MergedFeatures) -> ScoringInput:
        """Collapse session rows to one-per-(PP), fill missing DELTA_DM
        + RECENT_ADHERENCE with per-patient median.

        `groupby(BY_PP).agg("last")` takes the most-recent session's
        values per (PP). After this, exactly one row per (PP)
        regardless of session-history depth.
        """
        scoring = features.df.groupby(BY_PP).agg("last").reset_index()
        scoring = self.imputer.init_metrics(scoring)
        scoring = self._impute_per_patient_median(
            scoring, features.df, column=DELTA_DM, position="first",
        )
        scoring = self._impute_per_patient_median(
            scoring, features.df, column=RECENT_ADHERENCE, position="last",
        )
        return ScoringInput(df=scoring, validate_on_init=False)

    def _impute_per_patient_median(
        self,
        scoring: pd.DataFrame, per_session: pd.DataFrame,
        *, column: str, position: str,
    ) -> pd.DataFrame:
        """Fill `column` with per-patient median.

        `position` controls which session per (PP) supplies the value
        before computing the patient median:
          "first" → n=1 (matches v0.3.1 DELTA_DM rule)
          "last"  → n=-1 (matches v0.3.1 RECENT_ADHERENCE rule)
        """
        n = 1 if position == "first" else -1
        per_pp = get_nth(per_session, column, BY_PP, SESSION_INDEX, n=n)
        medians = per_pp.groupby(PATIENT_ID)[column].median().reset_index()
        return self.imputer.impute_metrics(scoring, column, medians)

    # ==================================================================
    # Stage 3b — bootstrap path (no sessions in the window).

    def _bootstrap_scoring_input(self, inputs: PreparedInputs) -> ScoringInput:
        """No sessions to learn from — assemble a scoring frame from
        the PPF cohort with NaN metrics, let the imputer seed defaults."""
        scoring_columns = BY_PP + [
            DELTA_DM, RECENT_ADHERENCE, WEEKS_SINCE_START, SESSION_INDEX,
            USAGE, USAGE_WEEK, DAYS,
        ]
        empty = pd.DataFrame(columns=scoring_columns)
        scoring = inputs.ppf.merge(empty, on=BY_PP, how="left")
        scoring = self.imputer.init_metrics(scoring)
        return ScoringInput(df=scoring, validate_on_init=False)

    # ==================================================================
    # Stage 4 — compute SCORE, slim columns, propagate attrs.

    def _score(
        self, scoring_input: ScoringInput, inputs: PreparedInputs,
    ) -> ScoringOutput:
        scored = self.scorer.compute_score(scoring_input.df)
        scored.attrs = inputs.ppf.attrs
        final = scored[BY_PP + FINAL_METRICS]
        return ScoringOutput(df=final, validate_on_init=False)
