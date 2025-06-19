# ai_cdss/processing/processor.py
from typing import List, Optional, Dict
from functools import reduce
from dataclasses import dataclass

import pandas as pd
from pandas import Timestamp
from pandera.typing import DataFrame

import numpy as np
from ai_cdss.constants import *
from ai_cdss.models import ScoringSchema, PPFSchema, SessionSchema, TimeseriesSchema, DataUnitSet, DataUnitName
from ai_cdss.processing.features import include_missing_sessions, apply_savgol_filter_groupwise, get_rolling_theilsen_slope
from ai_cdss.processing.utils import safe_merge

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#################################
# ------ Data Processing ------ #
#################################

# ---------------------------------------------------------------------
# Data Processor Class

@dataclass
class ProcessingContext:
    scoring_date: Timestamp = None

class DataProcessor:
    """
    A class for processing data and computing a final weighted score.

    The final score is computed as:

    .. math::

        S = \\alpha \\cdot A + \\beta \\cdot DM + \\gamma \\cdot PPF

    where:

    - :math:`A` is Adherence
    - :math:`DM` is the Difficulty Modulator
    - :math:`PPF` is the Patient Prescription Factor

    Parameters
    ----------
    weights : List[float]
        List of weights :math:`\\alpha`, :math:`\\beta`, :math:`\\gamma` for computing the final score.
    alpha : float
        The smoothing factor for EWMA, controlling how much past values influence the trend.
    """
    def __init__(
        self,
        weights: List[float] = [1,1,1], 
        alpha: float = 0.5,
        context: Optional[ProcessingContext] = None
    ):
        """
        Initialize the data processor with optional weights for scoring.
        """
        self.weights = weights
        self.alpha = alpha
        self.context = context or ProcessingContext()

    def process_data(
        self,
        data: DataUnitSet,
        ) -> DataFrame[ScoringSchema]:
        """
        Process and score patient-protocol combinations using session, timeseries, and PPF data.
        When bootstrapping patient, score is based on PPF only.
        Otherwise a weighted sum of DELTA_DM, ADHERENCE_RECENT and PPF determines the score.

        Parameters
        ----------
        session_data : DataFrame[SessionSchema]
            Session-level data including adherence and scheduling information.
        timeseries_data : DataFrame[TimeseriesSchema]
            Timepoint-level data including DMs and performance metrics.
        ppf_data : DataFrame[PPFSchema]
            Patient-protocol fitness values and contributions.

        Returns
        -------
        DataFrame[ScoringSchema]
            Final scored dataframe with protocol recommendations.
        """
        
        # Load data from DataUnitSet container
        session_unit = data.get(DataUnitName.SESSIONS)
        ppf_unit = data.get(DataUnitName.PPF)
        session_data = session_unit.data
        ppf_data = ppf_unit.data

        # Get scoring date from context (today)
        scoring_date = self._get_scoring_date()
        # Impute sessions that were not performed, assigning a date and NOT_PERFORMED status
        session_data = include_missing_sessions(session_data)
        # Filter sessions in study range
        session_data = session_data[
            (session_data[SESSION_DATE] >= session_data[CLINICAL_START]) &
            (session_data[SESSION_DATE] <= session_data[CLINICAL_END])
        ]
        # Weeks since start
        weeks_since_start_df = self.build_week_since_start(session_data, scoring_date)

        # --- Feature Building ---
        if not session_data.empty:

            # Compute Session Features
            dm_df = self.build_delta_dm(session_data[BY_PPS + [SESSION_DATE, DM_VALUE]].dropna())  # DELTA_DM
            adherence_df = self.build_recent_adherence(session_data)                               # ADHERENCE_RECENT
            usage_df = self.build_usage(session_data)                                              # USAGE
            usage_week_df = self.build_week_usage(session_data, scoring_date=scoring_date)         # USAGE_WEEK
            days_df = self.build_prescription_days(session_data, scoring_date=scoring_date)        # DAYS
            
            # Combine Session Features
            feat_pps_df = reduce(lambda l, r: pd.merge(l, r, on=BY_PP + [SESSION_DATE], how='left'), [adherence_df, dm_df, weeks_since_start_df])
            feat_pp_df  = reduce(lambda l, r: pd.merge(l, r, on=BY_PP, how='left'),  [ppf_data, usage_df, usage_week_df, days_df, feat_pps_df])
            feat_pp_df  = feat_pp_df.sort_values(by=BY_PP + [SESSION_DATE])

            # Store the whole scored DataFrame as a csv
            missing_cols = set(session_data.columns) - set(feat_pp_df.columns)
            log_df = feat_pp_df.merge(
                session_data[BY_PP + [SESSION_DATE] + list(missing_cols)], 
                on=BY_PP + [SESSION_DATE], 
                how='left'
            )
            log_filepath = DEFAULT_LOG_SCORING_FILEPATH.format(scoring_date=scoring_date)
            log_df.to_csv(log_filepath, index=False)
            logger.info(f"Logged complete scoring data at {log_filepath}")
            
            # Aggregate to patient protocol level
            scoring_input = feat_pp_df.groupby(BY_PP).agg("last").reset_index()

            # Fill features
            scoring_input = self._init_metrics(scoring_input)

            # Weekly imputation of DM and ADHERENCE
            scoring_input = self._impute_metrics(scoring_input)
        
        # If we are bootstrapping a study, and no patient prescriptions yet
        else:
            
            # Initialize feature df with expected columns
            scoring_columns = BY_PP + [DELTA_DM, RECENT_ADHERENCE, USAGE, USAGE_WEEK, DAYS]
            feat_agg = pd.DataFrame(columns=scoring_columns)
            
            # Merge for scoring
            scoring_input = reduce(lambda l, r: safe_merge(l, r, on=BY_PP), [ppf_data, feat_agg])
            
            # Add weeks since study start
            scoring_input = scoring_input.merge(weeks_since_start_df, on=PATIENT_ID, how="left")

        # --- Scoring Data ---
        scored_df = self.compute_score(scoring_input)
        scored_df.attrs = ppf_data.attrs
        
        return scored_df[BY_PP + FINAL_METRICS]

    def compute_score(self, data):
        """
        Initializes metrics based on legacy data and computes score

        Parameters
        ----------
        data : pd.DataFrame
            Aggregated session and timeseries data per patient-protocol.

        Returns
        -------
        pd.DataFrame
            Scored DataFrame sorted by patient and protocol.
        """       
        # Compute objective function score alpha*Adherence + beta*DM + gamma*PPF
        score = self._compute_score(data)
        
        # Sort the output dataframe
        score.sort_values(by=BY_PP, inplace=True)

        return score

    def _impute_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Weekly imputation using own patient data on PP given Feature df

        Impute the non-prescribed protocols of last week, based on this week data
            DM: Median imputation using the data from same usage + expected protocols
            Adherence: Median imputation using last week data

        Impute protocols with DAYS []
        """
        return data

    def _init_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing protocol-level metrics with zeros.

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe with potentially missing features.

        Returns
        -------
        pd.DataFrame
            Safe dataframe with no NaNs.
        """
        # When no prescriptions add empty list instead of NaN
        data[DAYS] = data[DAYS].apply(lambda x: [] if x is None or (not isinstance(x, list) and pd.isna(x)) else x)
        # When no sessions performed add a 0
        data[USAGE] = data[USAGE].astype("Int64").fillna(0)
        data[USAGE_WEEK] = data[USAGE_WEEK].astype("Int64").fillna(0)

        # How to initliaze DM and Adherence? -> Later weeks?
        return data
    
    def _get_scoring_date(self):
        return self.context.scoring_date or pd.Timestamp.today().normalize()

    def _compute_ewma(self, df, value_col, group_cols, sufix=""):
        """
        Compute Exponential Weighted Moving Average (EWMA) over grouped time series.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing time-series values.
        value_col : str
            The column for which EWMA should be calculated.
        group_cols : list of str
            Columns defining the group over which to apply the EWMA.

        Returns
        -------
        pd.DataFrame
            DataFrame with EWMA column replacing the original.
        """
        return df.assign(
            **{f"{value_col}{sufix}": df.groupby(by=group_cols)[value_col].transform(lambda x: x.ewm(alpha=self.alpha, adjust=True).mean())}
        )

    def _compute_score(self, scoring: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the final score based on adherence, DM, and PPF values.

        Applies the weighted scoring formula:

        .. math::

            S = \\alpha \\cdot A + \\beta \\cdot DM + \\gamma \\cdot PPF

        Parameters
        ----------
        scoring : pd.DataFrame
            DataFrame containing columns: ADHERENCE, DM_VALUE, PPF.

        Returns
        -------
        pd.DataFrame
            DataFrame with an added SCORE column.
        """
        scoring_df = scoring.copy()
        # weighted nan-compativle sum of factors
        scoring_df[SCORE] = (
            scoring_df[RECENT_ADHERENCE].astype("float64").fillna(0.) * self.weights[0]
            + scoring_df[DELTA_DM].astype("float64").fillna(0.) * self.weights[1]
            + scoring_df[PPF].astype("float64").fillna(0.) * self.weights[2]
        )
        return scoring_df

    def build_delta_dm(self, session: DataFrame[SessionSchema]) -> pd.DataFrame:
        # grouped = session.groupby(BY_PPS).agg({DM_VALUE: "mean"}).reset_index()
        grouped = session.copy().sort_values(by=BY_PP + [SESSION_DATE])

        # Session index per patient protocol
        grouped[SESSION_INDEX] = grouped.groupby(BY_PP).cumcount() + 1

        # Compute smoothing
        grouped[DM_SMOOTH] = grouped.groupby(BY_PP)[DM_VALUE].transform(
            apply_savgol_filter_groupwise, SAVGOL_WINDOW_SIZE, SAVGOL_POLY_ORDER
        )
        
        # Compute slope
        grouped[DELTA_DM] = grouped.groupby(BY_PP)[DM_SMOOTH].transform(
            lambda g: get_rolling_theilsen_slope(
                g,
                grouped.loc[g.index, SESSION_INDEX],
                THEILSON_REGRESSION_WINDOW_SIZE
            )
        ).fillna(0)

        return grouped[BY_PP + [SESSION_DATE, DM_VALUE, DELTA_DM]]

    def build_recent_adherence(self, session_df: DataFrame[SessionSchema]) -> pd.DataFrame:
        """
        This feature builder must return adherence for patient protocols, with the following considerations:
        - Adherence computed as SESSION_TIME / PRESCRIBED_SESSION_TIME
        - Adherence is considered 0 if presribed session is not performed
            - BUT is not considered if no session was performed that day.

        Additionally a recency bias is applied.
        """
        df = session_df.copy()
        # Includes prescribed but not performed sessions
        # df = include_missing_sessions(df)

        # Include same day logic do not penalize
        def day_skip_to_nan(group):
            # Check all STATUS == NOT_PERFORMED
            day_skipped = all(group[STATUS] == SessionStatus.NOT_PERFORMED)
            # EWMA does not use nan values for computation
            if day_skipped:
                group[ADHERENCE] = np.nan
            return group
        
        df = df.groupby(by=[PATIENT_ID, SESSION_DATE], group_keys=False).apply(day_skip_to_nan)

        df = df.sort_values(by=BY_PP + [SESSION_DATE, WEEKDAY_INDEX])
        df['SESSION_INDEX'] = df.groupby(BY_PP).cumcount() + 1

        # For a given patient protocol, take the array of adherences and mean aggregate with recency bias.
        df = self._compute_ewma(df, ADHERENCE, BY_PP, sufix="_RECENT")
        return df[BY_PPS + [SESSION_DATE, STATUS, SESSION_INDEX, ADHERENCE, RECENT_ADHERENCE]]

    def build_usage(self, session_df: DataFrame[SessionSchema]) -> pd.DataFrame:
        """
        This feature builder must return how many times protocols are used in this format:
        PATIENT_ID  PROTOCOL_ID  USAGE
        12          220          2
        12          231          1
        12          233          0

        Protocols that have not been yet prescribed for a patient are not returned.
        """
        df = session_df.copy()
        return (
            df.groupby([PATIENT_ID, PROTOCOL_ID], dropna=False)[SESSION_ID]
            .nunique()
            .reset_index(name=USAGE)
            .astype({USAGE: "Int64"})
        )

    def build_week_usage(self, session_df: DataFrame[SessionSchema], scoring_date: Timestamp = None) -> pd.DataFrame:
        """
        This feature builder must return how many times protocols are used in this week in this format:
        PATIENT_ID  PROTOCOL_ID  USAGE_WEEK
        12          220          2
        12          231          1
        12          233          0

        Protocols that have not been yet prescribed for a patient are not returned.
        """
        df = session_df.copy()

        # Compute current week's Monday 00:00 and Sunday 23:59:59.999999
        def week_range(date):
            week_start = date - pd.Timedelta(days=date.weekday())  # Monday 00:00
            week_start = week_start.normalize()
            week_end = week_start + pd.Timedelta(days=7)  # Next Monday 00:00
            return week_start, week_end

        # Calculate last week's Monday and Sunday
        week_start, week_end = week_range(scoring_date)

        # Filter only sessions in last week
        df = df[(df[SESSION_DATE] >= week_start) & (df[SESSION_DATE] < week_end)]

        # Count unique sessions per patient and protocol
        usage = (
            df.groupby([PATIENT_ID, PROTOCOL_ID], dropna=False)[SESSION_ID]
            .nunique()
            .reset_index(name=USAGE_WEEK)
            .astype({USAGE_WEEK: "Int64"})
        )

        return usage


    def build_week_since_start(self, patient_df: DataFrame[SessionSchema], scoring_date: Timestamp = None) -> pd.DataFrame:
        df = patient_df.copy()

        # Normalize session date and clinical trial start to the beginning of the week (Monday)
        session_week_start = df[SESSION_DATE] - pd.to_timedelta(df[SESSION_DATE].dt.weekday, unit='D')
        trial_week_start = df[CLINICAL_START] - pd.to_timedelta(df[CLINICAL_START].dt.weekday, unit='D')

        # Compute weeks since clinical trial start
        df[WEEKS_SINCE_START] = ((session_week_start - trial_week_start) / pd.Timedelta(weeks=1)).astype("int64")

        return df[[PATIENT_ID, PROTOCOL_ID, SESSION_DATE, WEEKS_SINCE_START]]

    def build_prescription_days(self, session_df: DataFrame[SessionSchema], scoring_date: Timestamp = None) -> pd.DataFrame:
        """
        This feature builder must return active prescriptions signaled as:
        PRESCRIPTION_ENDING_DATE == 2100-01-01 00:00:00

        In the following format, (encoding weekdays from 0-6):
        PATIENT_ID  PROTOCOL_ID DAYS
        12          220         [2]
        12          233         [0]
        """
        # If no scoring date is given, use today's date at 00:00
        week_start = scoring_date - pd.Timedelta(days=scoring_date.weekday())  # Monday 00:00
        week_start = week_start.normalize()
        
        # Filter activities where the prescription is still active in this week.
        # i.e., prescriptions whose ending date is on or after the start of this week
        active_prescriptions = session_df[session_df[PRESCRIPTION_ENDING_DATE] > week_start]
        
        # Group by patient and protocol, collect all unique weekday indices (0–6) where active prescriptions occurred
        prescribed_days = (
            active_prescriptions
            .groupby(BY_PP)[WEEKDAY_INDEX]
            .agg(lambda x: sorted(x.unique()))
            .rename(DAYS)
            .reset_index()
        )
        return prescribed_days

