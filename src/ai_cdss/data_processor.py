# data_processor.py
import pandas as pd
import logging
from typing import List
from ai_cdss.models import ScoringSchema, PPFSchema, SessionSchema, TimeseriesSchema
import pandera as pa
from pandera.typing import DataFrame

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    A class for processing patient session data, applying Exponential Weighted 
    Moving Average (EWMA) and computing a final weighted score.

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

    Methods
    -------
    process_data(session_data, timeseries_data, ppf_data)
        Processes session and timeseries data, computes adherence and DM EWMA, and
        calculates final scoring.
        
    compute_ewma(df, value_col, group_cols)
        Computes Exponential Weighted Moving Average (EWMA):

        .. math::
            
            EWMA_t = \\alpha \\cdot X_t + (1 - \\alpha) \\cdot EWMA_{t-1}

    compute_score(scoring)
        Computes the final scoring function.
    """
    def __init__(self, weights: List[float] = [1,1,1], alpha: float = 0.5):
        """
        Initialize the data processor with optional weights for scoring.
        """
        self.weights = weights
        self.alpha = alpha

    @pa.check_types
    def process_data(self, session_data: DataFrame[SessionSchema], timeseries_data: DataFrame[TimeseriesSchema], ppf_data: DataFrame[PPFSchema]) -> DataFrame[ScoringSchema]:
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
        data = self.merge_session_and_timeseries(session_data=session_data, timeseries_data=timeseries_data)
        score = self.compute_patient_protocol_scores(data, ppf_data)
        score.attrs = ppf_data.attrs # Propagate attrs
        return score

    def merge_session_and_timeseries(self, session_data: DataFrame[SessionSchema], timeseries_data: DataFrame[TimeseriesSchema]):
        """
        Merge and align session and timeseries data for each patient and protocol.

        Applies EWMA to adherence, DMs, and performance values, and joins the
        processed timeseries with session-level info.

        Parameters
        ----------
        session_data : DataFrame[SessionSchema]
            The session metadata per patient and protocol.
        timeseries_data : DataFrame[TimeseriesSchema]
            Detailed timepoint observations for difficulty and performance.

        Returns
        -------
        pd.DataFrame
            Merged and preprocessed dataset ready for protocol-level aggregation.
        """
        # Aggregate timeseries by session
        timeseries = (
            timeseries_data
            .pipe(self.aggregate_dms_by_time) # Merge same timepoint DMs
            .sort_values(by=["PATIENT_ID", "SESSION_ID", "SECONDS_FROM_START"]) # Sort df by time
            .pipe(self.compute_ewma, value_col="DM_VALUE", group_cols=["PATIENT_ID", "PROTOCOL_ID"]) # Compute DM EWMA
            .pipe(self.compute_ewma, value_col="PE_VALUE", group_cols=["PATIENT_ID", "PROTOCOL_ID"]) # Compute PE EWMA
        )
        
        # Compute ewma adherence
        session = (
            session_data
            .sort_values(by=["PATIENT_ID", "PROTOCOL_ID", "SESSION_ID"]) # Sort df by protocol, session
            .pipe(self.compute_ewma, value_col="ADHERENCE", group_cols=["PATIENT_ID", "PROTOCOL_ID"]) # Compute Adherence EWMA
        )
        
        # Merge session and timeseries data and compute protocol metrics ADHERENCE and ACTIVE PRESCRIPTIONS
        data = session.merge(timeseries, on=["PATIENT_ID", "PROTOCOL_ID", "SESSION_ID"], how="left")

        return data

    def compute_patient_protocol_scores(self, data, ppf_data):
        """
        Combine protocol-level metrics and PPF values to compute a final score.

        Merges adherence, DM, and PPF features and applies the scoring formula.

        Parameters
        ----------
        data : pd.DataFrame
            Aggregated session and timeseries data per patient-protocol.
        ppf_data : DataFrame[PPFSchema]
            Patient-Protocol Fitness dataframe with global and feature-level scores.

        Returns
        -------
        pd.DataFrame
            Scored DataFrame sorted by patient and protocol.
        """
        # Aggregate metrics per protocol
        data = self.aggregate_metrics_per_protocol(data)
        # Merge session, timeseries, and ppf
        data = ppf_data.merge(data, on=["PATIENT_ID", "PROTOCOL_ID"], how="left")
        # Initialize missing values
        data = self.initialize_missing_metrics(data)
        # Compute objective function score alpha*Adherence + beta*DM + gamma*PPF
        score = self.compute_score(data)
        # Sort the output dataframe
        score.sort_values(by=["PATIENT_ID", "PROTOCOL_ID"], inplace=True)
        return score

    def aggregate_dms_by_time(self, timeseries_data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate multiple difficulty modulator (DM) values at the same timepoint.

        Groups by patient, protocol, session, and time, and computes average DM and PE.

        Parameters
        ----------
        timeseries_data : pd.DataFrame
            Raw timeseries data with DM_KEY, DM_VALUE, PE_KEY, PE_VALUE.

        Returns
        -------
        pd.DataFrame
            Aggregated timeseries data with unique timepoints.
        """
        return (
            timeseries_data
            .sort_values(by=["PATIENT_ID", "PROTOCOL_ID", "SESSION_ID", "SECONDS_FROM_START"]) # Sort df by time
            .groupby(["PATIENT_ID", "PROTOCOL_ID", "SESSION_ID", "GAME_MODE", "SECONDS_FROM_START"])
            .agg({
                "DM_KEY": lambda x: tuple(set(x)),  # Unique parameters at this time
                "DM_VALUE": "mean",               # Average parameter value
                "PE_KEY": "first",                # Assume same performance key per time, take first
                "PE_VALUE": "mean"                # Average performance value (usually only one)
            })
            .reset_index()
        )

    def aggregate_metrics_per_protocol(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate EWMA values and adherence metrics at the protocol level.

        Also computes usage frequency and determines recommended weekdays.

        Parameters
        ----------
        data : pd.DataFrame
            Merged session-timeseries data.

        Returns
        -------
        pd.DataFrame
            Aggregated features per (PATIENT_ID, PROTOCOL_ID).
        """
        return (
            data
            .sort_values(["PATIENT_ID", "PROTOCOL_ID", "SESSION_ID", "SECONDS_FROM_START"])
            .groupby(["PATIENT_ID", "PROTOCOL_ID"])
            .agg(
                ADHERENCE=("ADHERENCE", "last"),
                DM_VALUE=("DM_VALUE", "last"),
                PE_VALUE=("PE_VALUE", "last"),
                USAGE=("SESSION_ID", "count"),
                DAYS=("WEEKDAY_INDEX", lambda x: sorted(x[data.loc[x.index, "PRESCRIPTION_ENDING_DATE"] == "2100-01-01"].unique())),
            )
        )

    def initialize_missing_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
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
        return data
    
    def compute_ewma(self, df, value_col, group_cols):
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
            **{f"{value_col}": df.groupby(group_cols)[value_col].transform(lambda x: x.ewm(alpha=self.alpha, adjust=True).mean())}
        )

    def compute_score(self, scoring: pd.DataFrame) -> pd.DataFrame:
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
        scoring_df['SCORE'] = (
            scoring_df['ADHERENCE'] * self.weights[0]
            + scoring_df['DM_VALUE'] * self.weights[1]
            + scoring_df['PPF'] * self.weights[2]
        )
        return scoring_df
