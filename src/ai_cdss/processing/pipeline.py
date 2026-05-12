"""`DataPipeline` — feature-build, impute, score, in that order.

The pipeline turns the raw inputs from `DataLoader` (sessions / patient
metadata / PPF cohort) into a one-row-per-(patient, protocol) scoring
DataFrame that `CDSS.recommend` consumes.

The flow:

    PreparedInputs                    (Stage 1 — cleaned, windowed)
        ├── SessionLevelFeatures      (Stage 2a — per-session features)
        ├── ProtocolLevelFeatures     (Stage 2b — per-protocol features)
        └── MergedFeatures            (Stage 2c — session × protocol)
            └── ScoringInput          (Stage 3 — imputed, one-per-PP)
                └── ScoringOutput     (Stage 4 — final SCORE)

Each stage's input and output is a typed dataclass from `contracts.py`
with documented column requirements. No more raw `pd.DataFrame` passed
via positional arguments — every method signature self-documents the
expected schema.

Compared to the v0.3.1 implementation (247 lines, dense
`reduce(lambda merge)` chains, opaque 3-tuple returns), this file is
structured as a sequence of small named steps. Behavior preserved
byte-for-byte; existing tests pass unchanged.
"""
from __future__ import annotations

import logging
from functools import reduce

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
    PATIENT_ID,
    RECENT_ADHERENCE,
    SESSION_DATE,
    SESSION_INDEX,
    USAGE,
    USAGE_WEEK,
    WEEKS_SINCE_START,
)
from ai_cdss.models import DataUnitName, DataUnitSet
from ai_cdss.processing.contracts import (
    MergedFeatures,
    PreparedInputs,
    ProtocolLevelFeatures,
    ScoringInput,
    ScoringOutput,
    SessionLevelFeatures,
)
from ai_cdss.processing.feature_builder import FeatureBuilder
from ai_cdss.processing.features import include_missing_sessions
from ai_cdss.processing.imputer import Imputer
from ai_cdss.processing.scorer import Scorer
from ai_cdss.processing.utils import get_nth

logger = logging.getLogger(__name__)


class DataPipeline:
    """End-to-end pipeline from raw DataLoader output to scored frame.

    Constructor injects the three processors. Default instances are
    used when omitted — convenient for tests and ad-hoc callers.
    """

    def __init__(
        self,
        feature_builder: FeatureBuilder | None = None,
        imputer:         Imputer | None = None,
        scorer:          Scorer | None = None,
    ) -> None:
        self.feature_builder = feature_builder or FeatureBuilder()
        self.imputer = imputer or Imputer()
        self.scorer = scorer or Scorer()

    # ------------------------------------------------------------------
    # Public entry point.

    def process(
        self, data: DataUnitSet, scoring_date: Timestamp,
    ) -> pd.DataFrame:
        """Run the full pipeline and return the scored DataFrame.

        The return is a plain `pd.DataFrame` (not `ScoringOutput`) for
        backward compat with the v0.3.1 API. The typed wrappers are
        internal to this module.
        """
        inputs = self._prepare(data, scoring_date)

        if not inputs.has_sessions:
            logger.info("Bootstrapping system, no session data available for patients.")
            scoring_input = self._bootstrap_scoring_input(inputs)
        else:
            features = self._build_features(inputs, scoring_date)
            scoring_input = self._impute_features(features)

        return self._score(scoring_input, inputs).df

    # ==================================================================
    # Stage 1 — prepare: clean and window the input frames.

    def _prepare(
        self, data: DataUnitSet, scoring_date: Timestamp,
    ) -> PreparedInputs:
        """Resolve DataUnitSet, merge clinical windows onto sessions,
        clamp session_date to [CLINICAL_START, min(CLINICAL_END, scoring_date)].

        Returns a `PreparedInputs` carrying the three frames the rest
        of the pipeline needs: `patient`, `session`, `ppf`.
        """
        patient = data.get(DataUnitName.PATIENT).data
        session = data.get(DataUnitName.SESSIONS).data
        ppf     = data.get(DataUnitName.PPF).data

        session = include_missing_sessions(session)
        session = self._attach_clinical_window(session, patient)
        session = self._clamp_to_window(session, scoring_date)

        return PreparedInputs(patient=patient, session=session, ppf=ppf,
                              validate_on_init=False)

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
        """Drop sessions outside [CLINICAL_START, min(CLINICAL_END, scoring_date)].

        Sessions after the patient's clinical end (or after scoring_date,
        whichever is earlier) are removed. Sessions before clinical
        start are also removed.
        """
        upper_bound = session[CLINICAL_END].where(
            session[CLINICAL_END] < scoring_date, scoring_date
        )
        in_window = (
            (session[SESSION_DATE] >= session[CLINICAL_START])
            & (session[SESSION_DATE] <= upper_bound)
        )
        return session.loc[in_window]

    # ==================================================================
    # Stage 2 — build features.

    def _build_features(
        self, inputs: PreparedInputs, scoring_date: Timestamp,
    ) -> MergedFeatures:
        """Run the three feature substages and merge their outputs.

        Returns a `MergedFeatures` with one row per (patient, protocol,
        session_date), carrying both session-level (DELTA_DM,
        RECENT_ADHERENCE) and protocol-level (USAGE, DAYS, …) columns.
        """
        session_features  = self._session_level_features(inputs.session)
        protocol_features = self._protocol_level_features(inputs, scoring_date)
        return self._broadcast_session_onto_protocol(session_features, protocol_features)

    def _session_level_features(
        self, session: pd.DataFrame,
    ) -> SessionLevelFeatures:
        """Per-session features: RECENT_ADHERENCE + DELTA_DM, keyed on
        (patient, protocol, session_date)."""
        adherence_df = self.feature_builder.build_recent_adherence(session)

        # DELTA_DM is computed only on rows with a non-null DM_VALUE.
        dm_rows = session[BY_PPS + [SESSION_DATE, DM_VALUE]].dropna()
        delta_df = self.feature_builder.build_delta_dm(dm_rows)

        merged = pd.merge(adherence_df, delta_df,
                          on=BY_PP + [SESSION_DATE], how="left")
        return SessionLevelFeatures(df=merged, validate_on_init=False)

    def _protocol_level_features(
        self, inputs: PreparedInputs, scoring_date: Timestamp,
    ) -> ProtocolLevelFeatures:
        """Per-protocol features: USAGE / USAGE_WEEK / DAYS, plus
        WEEKS_SINCE_START (patient-level, broadcast onto every
        protocol)."""
        per_protocol_dfs = [
            inputs.ppf,
            self.feature_builder.build_usage(inputs.session),
            self.feature_builder.build_week_usage(
                inputs.session, inputs.patient, scoring_date,
            ),
            self.feature_builder.build_prescription_days(
                inputs.session, inputs.patient, scoring_date,
            ),
        ]
        merged = reduce(
            lambda left, right: pd.merge(left, right, on=BY_PP, how="left"),
            per_protocol_dfs,
        )

        weeks_since_start = self.feature_builder.build_week_since_start(
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
        row. Result is one row per (patient, protocol, session_date),
        carrying both the session-level metrics and the protocol-level
        metadata. Sorted by (patient, protocol, session_date) for the
        groupby step downstream."""
        merged = pd.merge(
            protocol_level.df, session_level.df, on=BY_PP, how="left",
        ).sort_values(by=BY_PP + [SESSION_DATE])
        return MergedFeatures(df=merged, validate_on_init=False)

    # ==================================================================
    # Stage 3 — impute missing values.

    def _impute_features(self, features: MergedFeatures) -> ScoringInput:
        """Collapse per-session rows to one row per (PP) and fill
        missing DELTA_DM / RECENT_ADHERENCE per-patient median.

        The "last" aggregation takes each (PP)'s most-recent session
        row's values. After this step there is one row per (patient,
        protocol) regardless of session history depth.
        """
        scoring = features.df.groupby(BY_PP).agg("last").reset_index()
        scoring = self.imputer.init_metrics(scoring)
        scoring = self._impute_per_patient_median(
            scoring, features.df,
            column=DELTA_DM, position="first",
        )
        scoring = self._impute_per_patient_median(
            scoring, features.df,
            column=RECENT_ADHERENCE, position="last",
        )
        return ScoringInput(df=scoring, validate_on_init=False)

    def _impute_per_patient_median(
        self,
        scoring: pd.DataFrame,
        per_session: pd.DataFrame,
        *,
        column: str,
        position: str,
    ) -> pd.DataFrame:
        """Fill `column` in `scoring` with each patient's median value
        from the per-session frame.

        `position` controls which session per (PP) supplies the value
        before computing the patient median:
          * "first" — the FIRST session per (PP) (n=1, matches v0.3.1
            DELTA_DM rule).
          * "last"  — the LAST session per (PP)  (n=-1, matches v0.3.1
            RECENT_ADHERENCE rule).
        """
        n = 1 if position == "first" else -1
        per_pp = get_nth(per_session, column, BY_PP, SESSION_INDEX, n=n)
        medians = per_pp.groupby(PATIENT_ID)[column].median().reset_index()
        return self.imputer.impute_metrics(scoring, column, medians)

    # ==================================================================
    # Stage 3b — bootstrap path (no sessions in the window).

    def _bootstrap_scoring_input(self, inputs: PreparedInputs) -> ScoringInput:
        """No sessions to learn from — assemble a scoring frame from
        the PPF cohort alone with NaN metrics, then let the imputer
        seed defaults.
        """
        scoring_columns = BY_PP + [
            DELTA_DM, RECENT_ADHERENCE, WEEKS_SINCE_START, SESSION_INDEX,
            USAGE, USAGE_WEEK, DAYS,
        ]
        empty = pd.DataFrame(columns=scoring_columns)
        scoring = inputs.ppf.merge(empty, on=BY_PP, how="left")
        scoring = self.imputer.init_metrics(scoring)
        return ScoringInput(df=scoring, validate_on_init=False)

    # ==================================================================
    # Stage 4 — compute final score.

    def _score(
        self, scoring_input: ScoringInput, inputs: PreparedInputs,
    ) -> ScoringOutput:
        """Run the scorer; propagate PPF attrs (subscale metadata);
        slim to the final column set."""
        scored = self.scorer.compute_score(scoring_input.df)
        scored.attrs = inputs.ppf.attrs
        from ai_cdss.constants import FINAL_METRICS
        final = scored[BY_PP + FINAL_METRICS]
        return ScoringOutput(df=final, validate_on_init=False)
