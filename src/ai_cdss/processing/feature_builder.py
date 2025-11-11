# %%
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


class FeatureBuilder:
    """
    Builds engineered features for patient session and prescription data.
    Provides methods to compute deltas, adherence, usage statistics, and other features
    required for downstream modeling and analysis in the CDSS pipeline.
    """

    def build_delta_dm(self, session_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the delta (trend) of the DM (disease measure) value for each patient-protocol group
        using a Savitzky-Golay filter and Theil-Sen regression.

        Args:
            session (DataFrame): Input session data.

        Returns:
            pd.DataFrame: DataFrame with columns for patient, protocol, session date, DM value, and delta DM.
        """
        grouped = session_df.copy().sort_values(by=BY_PP + [SESSION_DATE])
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

    def build_recent_adherence(self, session_df: pd.DataFrame) -> pd.DataFrame:
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

    def build_usage(self, session_df: pd.DataFrame) -> pd.DataFrame:
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
        self,
        session_df: pd.DataFrame,
        patient_df: pd.DataFrame,
        scoring_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Compute the number of unique sessions (usage) per patient–protocol pair
        for the *last completed* patient-aligned week before scoring_date.

        Example:
            clinical_start = 2025-11-11 (Tue)
            scoring_date   = 2025-11-18 (Tue, recommendation day)
            --> window used is [2025-11-11, 2025-11-18)
        """
        session_df = session_df.copy()
        session_df[SESSION_DATE] = pd.to_datetime(
            session_df[SESSION_DATE], errors="coerce"
        )

        anchors = self._last_completed_week_window(patient_df, scoring_date)

        # Attach week window to each session row (by patient)
        df = session_df.merge(
            anchors[[PATIENT_ID, "week_start", "week_end"]],
            on=PATIENT_ID,
            how="inner",
        )

        in_window = (df[SESSION_DATE] >= df["week_start"]) & (
            df[SESSION_DATE] < df["week_end"]
        )
        df = df.loc[in_window]

        usage = (
            df.groupby([PATIENT_ID, PROTOCOL_ID], dropna=False)[SESSION_ID]
            .nunique()
            .reset_index(name=USAGE_WEEK)
            .astype({USAGE_WEEK: "Int64"})
        )
        return usage

    def build_week_since_start(
        self, patient_df: pd.DataFrame, scoring_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Compute whole weeks since each patient's clinical start as of scoring_date.
        Example: started 8 days ago -> week 1.
        """
        df = self._with_weeks_since_start(patient_df, scoring_date)
        return df[[PATIENT_ID, WEEKS_SINCE_START]]

    def build_number_prescriptions(self, session_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the cumulative number of prescriptions for each patient-protocol pair.

        Args:
            session (DataFrame): Input session data.

        Returns:
            pd.DataFrame: DataFrame with a column for total prescribed count.
        """
        df = session_df.copy()
        df[TOTAL_PRESCRIBED] = df.groupby(BY_PP).cumcount() + 1
        return df

    def build_prescription_days(
        self,
        session_df: pd.DataFrame,
        patient_df: pd.DataFrame,
        scoring_date: Timestamp,
    ) -> pd.DataFrame:
        """
        Get the list of prescribed days (weekday indices) for each patient–protocol
        in the LAST completed patient-aligned week before scoring_date.

        Example:
            clinical_start = 2025-11-11 (Tue)
            scoring_date   = 2025-11-18

            --> consider sessions with SESSION_DATE in [2025-11-11, 2025-11-18)
                and aggregate their WEEKDAY_INDEX per patient–protocol.
        """
        session_df = session_df.copy()
        session_df[SESSION_DATE] = pd.to_datetime(
            session_df[SESSION_DATE], errors="coerce"
        )

        anchors = self._last_completed_week_window(patient_df, scoring_date)

        df = session_df.merge(
            anchors[[PATIENT_ID, "week_start", "week_end"]],
            on=PATIENT_ID,
            how="inner",
        )

        in_window = (df[SESSION_DATE] >= df["week_start"]) & (
            df[SESSION_DATE] < df["week_end"]
        )
        df = df.loc[in_window]

        prescribed_days = (
            df.groupby([PATIENT_ID, PROTOCOL_ID])[WEEKDAY_INDEX]
            .agg(lambda x: sorted(x.unique()))
            .rename(DAYS)
            .reset_index()
        )
        return prescribed_days

    def _with_weeks_since_start(
        self, patient_df: pd.DataFrame, scoring_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Return patient_df reduced to [PATIENT_ID, CLINICAL_START, WEEKS_SINCE_START],
        where WEEKS_SINCE_START is the number of completed 7-day periods as of scoring_date.
        """
        df = patient_df[[PATIENT_ID, CLINICAL_START]].copy()
        df[CLINICAL_START] = pd.to_datetime(df[CLINICAL_START], errors="coerce")

        scoring_day = scoring_date.normalize()
        start_day = df[CLINICAL_START].dt.normalize()

        days_since_start = (scoring_day - start_day).dt.days
        df[WEEKS_SINCE_START] = (days_since_start // 7).astype("Int64")

        return df

    def _last_completed_week_window(
        self, patient_df: pd.DataFrame, scoring_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        For each patient with at least 1 completed week, compute the last completed
        week window [week_start, week_end) before scoring_date.

        Returns:
            DataFrame with [PATIENT_ID, CLINICAL_START, WEEKS_SINCE_START, week_start, week_end]
            filtered to patients with WEEKS_SINCE_START > 0.
        """
        df = self._with_weeks_since_start(patient_df, scoring_date)

        # We only care about patients with at least 1 completed week
        mask = df[WEEKS_SINCE_START] > 0
        df = df.loc[mask].copy()

        # last completed week index = WEEKS_SINCE_START - 1
        last_week_idx = (df[WEEKS_SINCE_START] - 1).astype(int)

        start_day = df[CLINICAL_START].dt.normalize()
        df["week_start"] = start_day + pd.to_timedelta(last_week_idx * 7, unit="D")
        df["week_end"] = df["week_start"] + pd.Timedelta(days=7)

        return df
