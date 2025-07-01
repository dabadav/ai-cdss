import logging
from functools import reduce

import pandas as pd
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
    RECENT_ADHERENCE,
    SESSION_DATE,
    SESSION_INDEX,
    USAGE,
    USAGE_WEEK,
    WEEKS_SINCE_START,
)
from ai_cdss.models import DataUnitName, DataUnitSet
from ai_cdss.processing.feature_builder import FeatureBuilder
from ai_cdss.processing.features import include_missing_sessions
from ai_cdss.processing.imputer import Imputer
from ai_cdss.processing.scorer import Scorer
from ai_cdss.processing.utils import get_nth
from pandas import Timestamp

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Orchestrates feature building, imputation, and scoring for patient-protocol data.
    """

    def __init__(self, feature_builder=None, imputer=None, scorer=None):
        """
        Initialize the DataPipeline with optional custom feature builder, imputer, and scorer.
        """
        self.feature_builder = feature_builder or FeatureBuilder()
        self.imputer = imputer or Imputer()
        self.scorer = scorer or Scorer()

    def process(self, data: DataUnitSet, scoring_date: Timestamp) -> pd.DataFrame:
        """
        Run the full data processing pipeline: prepare, build features, impute, and score.

        Args:
            data: DataUnitSet containing session and PPF data.
            scoring_date: Timestamp for scoring reference.

        Returns:
            pd.DataFrame: Final scored recommendations.
        """
        patient_data, session_data, ppf_data = self._prepare_session_data(
            data, scoring_date
        )
        system_bootstrap = session_data.empty

        if system_bootstrap:
            scoring_input = self._handle_empty_case(ppf_data)

        else:
            features_df = self._build_features(
                patient_data=patient_data,
                session_data=session_data,
                ppf_data=ppf_data,
                scoring_date=scoring_date,
            )
            scoring_input = self._impute_features(features_df)

        return self._finalize_scoring(scoring_input, ppf_data)

    def _prepare_session_data(self, data, scoring_date):
        """
        Prepare and clean session and PPF data for processing.

        Args:
            data: DataUnitSet with session and PPF data.
            scoring_date: Reference date for scoring.

        Returns:
            Tuple of (session_data, ppf_data) as DataFrames.
        """
        session_unit = data.get(DataUnitName.SESSIONS)
        ppf_unit = data.get(DataUnitName.PPF)
        patient_unit = data.get(DataUnitName.PATIENT)
        session_data = session_unit.data
        ppf_data = ppf_unit.data
        patient_data = patient_unit.data

        session_data = include_missing_sessions(session_data)
        session_data = session_data.merge(
            patient_data[[PATIENT_ID, CLINICAL_START, CLINICAL_END]],
            on=PATIENT_ID,
            how="left",
        )

        print(session_data)

        for col in [CLINICAL_START, CLINICAL_END]:
            if not pd.api.types.is_datetime64_any_dtype(session_data[col]):
                session_data[col] = pd.to_datetime(session_data[col], errors="coerce")
            session_data[col] = session_data[col].dt.normalize()
        date_upper_bound = session_data[CLINICAL_END].where(
            session_data[CLINICAL_END] < scoring_date, scoring_date
        )
        session_data = session_data[
            (session_data[SESSION_DATE] >= session_data[CLINICAL_START])
            & (session_data[SESSION_DATE] <= date_upper_bound)
        ]
        return patient_data, session_data, ppf_data

    def _build_features(self, patient_data, session_data, ppf_data, scoring_date):
        """
        Build and merge all feature DataFrames required for scoring.

        Args:
            session_data: Cleaned session DataFrame.
            ppf_data: Patient-protocol fit DataFrame.
            weeks_since_start_df: DataFrame with weeks since start.
            scoring_date: Reference date for scoring.

        Returns:
            pd.DataFrame: Combined feature DataFrame.
        """

        # Session-level features
        session_features_df = reduce(
            lambda l, r: pd.merge(l, r, on=BY_PP + [SESSION_DATE], how="left"),
            [
                self.feature_builder.build_recent_adherence(session_data),
                self.feature_builder.build_delta_dm(
                    session_data[BY_PPS + [SESSION_DATE, DM_VALUE]].dropna()
                ),
            ],
        )

        # all_patient_protocols_df contains the full list of patient-protocol pairs
        all_patient_protocols_df = ppf_data

        # Protocol-level features
        protocol_features_df = reduce(
            lambda l, r: pd.merge(l, r, on=BY_PP, how="left"),
            [
                all_patient_protocols_df,
                self.feature_builder.build_usage(session_data),
                self.feature_builder.build_week_usage(session_data, scoring_date),
                self.feature_builder.build_prescription_days(
                    session_data, scoring_date
                ),
            ],
        )

        weeks_since_start_df = self.feature_builder.build_week_since_start(
            patient_data, scoring_date
        )

        protocol_features_df = pd.merge(
            protocol_features_df, weeks_since_start_df, on=PATIENT_ID, how="left"
        )

        # Augment session-level features with protocol-level features
        session_features_augmented_df = pd.merge(
            protocol_features_df, session_features_df, on=BY_PP, how="left"
        ).sort_values(by=BY_PP + [SESSION_DATE])

        return session_features_augmented_df

    def _impute_features(self, feat_pp_df):
        """
        Impute missing feature values and initialize metrics for scoring.

        Args:
            feat_pp_df: Combined feature DataFrame.

        Returns:
            pd.DataFrame: Feature DataFrame ready for scoring.
        """
        scoring_input = feat_pp_df.groupby(BY_PP).agg("last").reset_index()
        scoring_input = self.imputer.init_metrics(scoring_input)

        # Impute delta_dm
        delta_nth = get_nth(feat_pp_df, DELTA_DM, BY_PP, SESSION_INDEX, n=1)
        delta_medians = delta_nth.groupby(PATIENT_ID)[DELTA_DM].median().reset_index()
        scoring_input = self.imputer.impute_metrics(
            scoring_input, DELTA_DM, delta_medians
        )

        # Impute recent_adherence
        adherence_last = get_nth(
            feat_pp_df, RECENT_ADHERENCE, BY_PP, SESSION_INDEX, n=-1
        )
        adherence_medians = (
            adherence_last.groupby(PATIENT_ID)[RECENT_ADHERENCE].median().reset_index()
        )
        scoring_input = self.imputer.impute_metrics(
            scoring_input, RECENT_ADHERENCE, adherence_medians
        )

        return scoring_input

    def _handle_empty_case(self, ppf_data):
        """
        Handle the case where session data is empty (bootstrapping).

        Args:
            ppf_data: Patient-protocol fit DataFrame.

        Returns:
            pd.DataFrame: Feature DataFrame for scoring.
        """
        scoring_columns = BY_PP + [
            DELTA_DM,
            RECENT_ADHERENCE,
            WEEKS_SINCE_START,
            SESSION_INDEX,
            USAGE,
            USAGE_WEEK,
            DAYS,
        ]
        bootstrap_df = pd.DataFrame(columns=scoring_columns)
        all_patient_protocols_df = ppf_data

        scoring_input = all_patient_protocols_df.merge(
            bootstrap_df, on=BY_PP, how="left"
        )
        scoring_input = self.imputer.init_metrics(scoring_input)

        return scoring_input

    def _finalize_scoring(self, scoring_input, ppf_data):
        """
        Compute the final score and format the output DataFrame.

        Args:
            scoring_input: Feature DataFrame ready for scoring.
            ppf_data: Patient-protocol fit DataFrame (for attrs).

        Returns:
            pd.DataFrame: Final scored recommendations.
        """
        scored_df = self.scorer.compute_score(scoring_input)
        scored_df.attrs = ppf_data.attrs
        return scored_df[BY_PP + FINAL_METRICS]
