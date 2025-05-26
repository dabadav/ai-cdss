from typing import List
from dataclasses import dataclass
from functools import reduce


import pandas as pd
import pandera as pa
from pandera.typing import DataFrame
from scipy import signal

from ai_cdss.models import ScoringSchema, PPFSchema, SessionSchema, TimeseriesSchema
from ai_cdss.processing import safe_merge, apply_savgol_filter_groupwise, get_rolling_theilsen_slope
from ai_cdss.constants import (
    BY_PP, BY_PPS, BY_PPST, 
    PROTOCOL_ID, 
    SESSION_ID,
    DM_KEY, PE_KEY,
    DM_VALUE, PE_VALUE,
    DELTA_DM,
    ADHERENCE,
    RECENT_ADHERENCE,
    USAGE,
    DAYS,
    PPF,
    SCORE,
    WEEKDAY_INDEX,
    PRESCRIPTION_ENDING_DATE,
    PRESCRIPTION_ACTIVE,
    FINAL_METRICS, 
    SAVGOL_WINDOW_SIZE,
    SAVGOL_POLY_ORDER,
    THEILSON_REGRESSION_WINDOW_SIZE
)
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Data Processor Class

@dataclass
class FeatureBlock:
    df: pd.DataFrame
    level: List[str]

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
        alpha: float = 0.5
    ):
        """
        Initialize the data processor with optional weights for scoring.
        """
        self.weights = weights
        self.alpha = alpha

    @pa.check_types
    def process_data(
        self, 
        session_data: DataFrame[SessionSchema], 
        timeseries_data: DataFrame[TimeseriesSchema], 
        ppf_data: DataFrame[PPFSchema], 
        init_data: pd.DataFrame
    ) -> DataFrame[ScoringSchema]:
        """
        Process and score patient-protocol combinations using session, timeseries, and PPF data.

        Applies EWMA to adherence and difficulty modulators, merges features,
        and computes final patient-protocol scores.

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

        # --- Log-Level Output ---
        ## PATIENT_ID + PROTOCOL_ID + SESSION_ID + ADHERENCE + RECENT_ADHERENCE + USAGE + DAYS + DM_VALUE + DELTA_DM_VALUE + PE_VALUE + PPF + SCORE
        
        # --- Feature Building ---
        ts_feat_df = self.build_delta_dm(timeseries_data)
        adherence_df = self.build_recent_adherence(session_data)
        usage_df = self.build_usage(session_data)
        days_df = self.build_prescription_days(session_data)

        # --- Combine Session Features ---
        feat_df = reduce(lambda l, r: safe_merge(l, r, on=BY_PPS), [ts_feat_df, adherence_df, usage_df, days_df])
        # Store the whole scored DataFrame as a csv
    
        # --- Aggregation ---
        feat_agg = feat_df.groupby(BY_PP).agg("last").reset_index()

        # --- Merge All for Scoring ---
        scoring_input = reduce(lambda l, r: safe_merge(l, r, on=BY_PP), [
            ppf_data,
            feat_agg
        ])

        # --- Scoring ---
        scored_df = self.compute_score(scoring_input, init_data)
        scored_df.attrs = ppf_data.attrs
        
        return scored_df

    def compute_score(self, data, protocol_metrics):
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
        # Initialize missing values
        data = self._init_metrics(data, protocol_metrics)
        
        # Compute objective function score alpha*Adherence + beta*DM + gamma*PPF
        score = self._compute_score(data)
        
        # Sort the output dataframe
        score.sort_values(by=BY_PP, inplace=True)

        return score

    def _init_metrics(self, data: pd.DataFrame, protocol_metrics: pd.DataFrame) -> pd.DataFrame:
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
        data = data.fillna(0)

        # # Only fill NaNs using map for matching PROTOCOL_IDs
        # data[ADHERENCE] = data[ADHERENCE].fillna(0)
        # # data[ADHERENCE] = data[ADHERENCE].fillna(data[PROTOCOL_ID].map(protocol_metrics[ADHERENCE]))
        # data[DM_VALUE] = data[DM_VALUE].fillna(data[PROTOCOL_ID].map(protocol_metrics["DM_DELTA"]))
        # # data[DM_VALUE] = data[DM_VALUE].fillna(data[PROTOCOL_ID].map(protocol_metrics["DM_DELTA"]))
        # data[PE_VALUE] = data[PE_VALUE].fillna(0)
        # data[DAYS]  = data[DAYS].fillna(0)
        # data[USAGE] = data[USAGE].fillna(0)
        # data = data.fillna(0)
        return data
    
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
        scoring_df[SCORE] = (
            scoring_df[RECENT_ADHERENCE] * self.weights[0]
            + scoring_df[DELTA_DM] * self.weights[1]
            + scoring_df[PPF] * self.weights[2]
        )
        return scoring_df

    def build_delta_dm(self, ts_df: DataFrame[TimeseriesSchema]) -> FeatureBlock:
        grouped = ts_df.groupby(BY_PPS).agg({DM_VALUE: "mean", PE_VALUE: "mean"}).reset_index()

        grouped['SESSION_INDEX'] = grouped.groupby(BY_PP).cumcount() + 1
        grouped['DM_SMOOTH'] = grouped.groupby(BY_PP)[DM_VALUE].transform(
            apply_savgol_filter_groupwise, SAVGOL_WINDOW_SIZE, SAVGOL_POLY_ORDER
        )
        grouped[DELTA_DM] = grouped.groupby(BY_PP, group_keys=False).apply(
            lambda g: get_rolling_theilsen_slope(g['DM_SMOOTH'], g['SESSION_INDEX'], THEILSON_REGRESSION_WINDOW_SIZE)
        ).fillna(0)
        return grouped[BY_PPS + [DM_VALUE, DELTA_DM]]

    def build_recent_adherence(self, session_df: DataFrame[SessionSchema]) -> pd.DataFrame:
        df = session_df.copy()
        df = self._compute_ewma(df, ADHERENCE, BY_PP, sufix="_RECENT")
        return df[BY_PPS + [RECENT_ADHERENCE]]

    def build_usage(self, session_df: DataFrame[SessionSchema]) -> pd.DataFrame:
        df = session_df.copy()
        df[USAGE] = df.groupby(BY_PP)[SESSION_ID].transform("nunique").astype("Int64")
        return df[BY_PPS + [USAGE]]

    def build_prescription_days(self, session_df: DataFrame[SessionSchema]) -> pd.DataFrame:
        prescribed_days = (
            session_df[session_df[PRESCRIPTION_ENDING_DATE] == PRESCRIPTION_ACTIVE]
            .groupby(BY_PP)[WEEKDAY_INDEX]
            .agg(lambda x: sorted(x.unique()))
            .rename(DAYS)
            .reset_index()
        )
        merged = safe_merge(session_df[BY_PPS], prescribed_days, on=BY_PP, how="left")
        return merged[BY_PPS + [DAYS]]