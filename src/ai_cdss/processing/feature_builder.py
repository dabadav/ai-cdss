import numpy as np
import pandas as pd
from ai_cdss.constants import (
    ADHERENCE,
    BY_PP,
    BY_PPS,
    CLINICAL_START,
    DAYS,
    DELTA_DM,
    DM_SMOOTH,
    DM_VALUE,
    PATIENT_ID,
    PRESCRIPTION_ENDING_DATE,
    PROTOCOL_ID,
    RECENT_ADHERENCE,
    SAVGOL_POLY_ORDER,
    SAVGOL_WINDOW_SIZE,
    SESSION_DATE,
    SESSION_ID,
    SESSION_INDEX,
    STATUS,
    THEILSON_REGRESSION_WINDOW_SIZE,
    TOTAL_PRESCRIBED,
    USAGE,
    USAGE_WEEK,
    WEEKDAY_INDEX,
    WEEKS_SINCE_START,
    SessionStatus,
)
from ai_cdss.processing.features import (
    apply_savgol_filter_groupwise,
    compute_ewma,
    get_rolling_theilsen_slope,
)
from pandas import Timestamp
from pandera.typing import DataFrame


class FeatureBuilder:
    """
    Builds engineered features for patient session and prescription data.
    Provides methods to compute deltas, adherence, usage statistics, and other features
    required for downstream modeling and analysis in the CDSS pipeline.
    """

    def build_delta_dm(self, session: DataFrame) -> pd.DataFrame:
        """
        Compute the delta (trend) of the DM (disease measure) value for each patient-protocol group
        using a Savitzky-Golay filter and Theil-Sen regression.

        Args:
            session (DataFrame): Input session data.

        Returns:
            pd.DataFrame: DataFrame with columns for patient, protocol, session date, DM value, and delta DM.
        """
        grouped = session.copy().sort_values(by=BY_PP + [SESSION_DATE])
        grouped[SESSION_INDEX] = grouped.groupby(BY_PP).cumcount() + 1
        grouped[DM_SMOOTH] = grouped.groupby(BY_PP)[DM_VALUE].transform(
            apply_savgol_filter_groupwise, SAVGOL_WINDOW_SIZE, SAVGOL_POLY_ORDER
        )
        grouped[DELTA_DM] = (
            grouped.groupby(BY_PP)[DM_SMOOTH]
            .transform(
                lambda g: get_rolling_theilsen_slope(
                    g,
                    grouped.loc[g.index, SESSION_INDEX],
                    THEILSON_REGRESSION_WINDOW_SIZE,
                )
            )
            .fillna(0)
        )
        return grouped[BY_PP + [SESSION_DATE, DM_VALUE, DELTA_DM]]

    def build_recent_adherence(self, session_df: DataFrame) -> pd.DataFrame:
        """
        Compute recent adherence for each session, setting adherence to NaN for days where all sessions were not performed.
        Applies an exponentially weighted moving average (EWMA) to adherence values.

        Args:
            session_df (DataFrame): Input session data.

        Returns:
            pd.DataFrame: DataFrame with recent adherence and related columns.
        """
        df = session_df.copy()

        def day_skip_to_nan(group):
            day_skipped = all(group[STATUS] == SessionStatus.NOT_PERFORMED)
            if day_skipped:
                group[ADHERENCE] = np.nan
            return group

        df = df.groupby(by=[PATIENT_ID, SESSION_DATE], group_keys=False).apply(
            day_skip_to_nan
        )
        df = df.sort_values(by=BY_PP + [SESSION_DATE, WEEKDAY_INDEX])
        df[SESSION_INDEX] = (df.groupby(BY_PP).cumcount() + 1).astype("Int64")
        df = compute_ewma(df, ADHERENCE, BY_PP, sufix="_RECENT")
        return df[
            BY_PPS + [SESSION_DATE, STATUS, SESSION_INDEX, ADHERENCE, RECENT_ADHERENCE]
        ]

    def build_usage(self, session_df: DataFrame) -> pd.DataFrame:
        """
        Compute the total number of unique sessions (usage) per patient-protocol pair.

        Args:
            session_df (DataFrame): Input session data.

        Returns:
            pd.DataFrame: DataFrame with usage counts per patient and protocol.
        """
        df = session_df.copy()
        return (
            df.groupby([PATIENT_ID, PROTOCOL_ID], dropna=False)[SESSION_ID]
            .nunique()
            .reset_index(name=USAGE)
            .astype({USAGE: "Int64"})
        )

    def build_week_usage(
        self, session_df: DataFrame, scoring_date: Timestamp
    ) -> pd.DataFrame:
        """
        Compute the number of unique sessions (usage) per patient-protocol pair for the week containing the scoring date.

        Args:
            session_df (DataFrame): Input session data.
            scoring_date (Timestamp): The date to define the week of interest.

        Returns:
            pd.DataFrame: DataFrame with weekly usage counts per patient and protocol.
        """
        df = session_df.copy()

        def week_range(date):
            week_start = date - pd.Timedelta(days=date.weekday())
            week_start = week_start.normalize()
            week_end = week_start + pd.Timedelta(days=7)
            return week_start, week_end

        week_start, week_end = week_range(scoring_date)
        df = df[(df[SESSION_DATE] >= week_start) & (df[SESSION_DATE] < week_end)]
        usage = (
            df.groupby([PATIENT_ID, PROTOCOL_ID], dropna=False)[SESSION_ID]
            .nunique()
            .reset_index(name=USAGE_WEEK)
            .astype({USAGE_WEEK: "Int64"})
        )
        return usage

    def build_week_since_start(self, patient_df: DataFrame) -> pd.DataFrame:
        """
        Compute the number of weeks since the clinical trial start for each session.

        Args:
            patient_df (DataFrame): Input patient/session data.

        Returns:
            pd.DataFrame: DataFrame with weeks since start for each session.
        """
        df = patient_df.copy()
        session_week_start = df[SESSION_DATE] - pd.to_timedelta(
            df[SESSION_DATE].dt.weekday, unit="D"
        )
        trial_week_start = df[CLINICAL_START] - pd.to_timedelta(
            df[CLINICAL_START].dt.weekday, unit="D"
        )
        df[WEEKS_SINCE_START] = (
            (session_week_start - trial_week_start) / pd.Timedelta(weeks=1)
        ).astype("Int64")
        return df[[PATIENT_ID, PROTOCOL_ID, SESSION_DATE, WEEKS_SINCE_START]]

    def build_number_prescriptions(self, session: DataFrame) -> pd.DataFrame:
        """
        Compute the cumulative number of prescriptions for each patient-protocol pair.

        Args:
            session (DataFrame): Input session data.

        Returns:
            pd.DataFrame: DataFrame with a column for total prescribed count.
        """
        df = session.copy()
        df[TOTAL_PRESCRIBED] = df.groupby(BY_PP).cumcount() + 1
        return df

    def build_prescription_days(
        self, session_df: DataFrame, scoring_date: Timestamp
    ) -> pd.DataFrame:
        """
        Get the list of prescribed days (weekday indices) for active prescriptions in the week of the scoring date.

        Args:
            session_df (DataFrame): Input session data.
            scoring_date (Timestamp): The date to define the week of interest.

        Returns:
            pd.DataFrame: DataFrame with prescribed days per patient-protocol pair.
        """
        week_start = scoring_date - pd.Timedelta(days=scoring_date.weekday())
        week_start = week_start.normalize()
        active_prescriptions = session_df[
            session_df[PRESCRIPTION_ENDING_DATE] > week_start
        ]
        prescribed_days = (
            active_prescriptions.groupby(BY_PP)[WEEKDAY_INDEX]
            .agg(lambda x: sorted(x.unique()))
            .rename(DAYS)
            .reset_index()
        )
        return prescribed_days
