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
    Processing batch patient session data, including time-series and session data,
    and computing final scoring based on predefined weights.
    """

    def __init__(self, weights: List[float] = [1,1,1], alpha: float = 0.5):
        """
        Initialize the data processor with optional weights for scoring.

        Parameters
        ----------
        weights : list, optional
            Weights for computing the final score. Default is [1, 1, 1].
        """
        self.weights = weights
        self.alpha = alpha

    @pa.check_types
    def process_data(self, session_data: DataFrame[SessionSchema], timeseries_data: DataFrame[TimeseriesSchema], ppf_data: DataFrame[PPFSchema]) -> DataFrame[ScoringSchema]:

        # Aggregate timeseries by session
        timeseries = (
            timeseries_data
            .pipe(self.aggregate_dms_by_time) # Merge same timepoint DMs
            .sort_values(by=["PATIENT_ID", "SESSION_ID", "SECONDS_FROM_START"]) # Sort df by time
            .pipe(self.compute_ewma, value_col="DM_VALUE", group_cols=["PATIENT_ID", "PROTOCOL_ID"]) # Compute DM EWMA
            .pipe(self.compute_ewma, value_col="PE_VALUE", group_cols=["PATIENT_ID", "PROTOCOL_ID"]) # Compute PE EWMA
        )
        
        # Aggregate session by sessino
        session = (
            session_data
            .sort_values(by=["PATIENT_ID", "PROTOCOL_ID", "SESSION_ID"]) # Sort df by protocol, session
            .pipe(self.compute_ewma, value_col="ADHERENCE", group_cols=["PATIENT_ID", "PROTOCOL_ID"]) # Compute Adherence EWMA
        )
        
        # Merge session and timeseries data and compute protocol metrics ADHERENCE and ACTIVE PRESCRIPTIONS
        data = session.merge(timeseries, on=["PATIENT_ID", "PROTOCOL_ID", "SESSION_ID"], how="left")
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
        
        return (
            timeseries_data
            .sort_values(by=["PATIENT_ID", "PROTOCOL_ID", "SESSION_ID", "SECONDS_FROM_START"]) # Sort df by time
            .groupby(["PATIENT_ID", "PROTOCOL_ID", "SESSION_ID", "GAME_MODE", "SECONDS_FROM_START"])
            .agg({
                "DM_KEY": lambda x: list(set(x)),  # Unique parameters at this time
                "DM_VALUE": "mean",               # Average parameter value
                "PE_KEY": "first",                # Assume same performance key per time, take first
                "PE_VALUE": "mean"                # Average performance value (usually only one)
            })
            .reset_index()
        )

    def aggregate_metrics_per_protocol(self, data: pd.DataFrame) -> pd.DataFrame:

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
        data = data.fillna(0)
        return data
    
    def compute_ewma(self, df, value_col, group_cols):
        """Compute Exponential Weighted Moving Average (EWMA) for a given column within each session time-series."""
        return df.assign(
            **{f"{value_col}": df.groupby(group_cols)[value_col].transform(lambda x: x.ewm(alpha=self.alpha, adjust=True).mean())}
        )

    def compute_score(self, scoring: pd.DataFrame) -> pd.DataFrame:
        """Compute the final scoring based on weights."""
        scoring_df = scoring.copy()
        scoring_df['SCORE'] = (
            scoring_df['ADHERENCE'] * self.weights[0]
            + scoring_df['DM_VALUE'] * self.weights[1]
            + scoring_df['PPF'] * self.weights[2]
        )
        return scoring_df
