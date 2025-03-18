# data_processor.py
import pandas as pd
import logging
from typing import List
from ai_cdss.models import ScoringSchema, PPFSchema, BatchSchema
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
    def process_data(self, data: DataFrame[BatchSchema], ppf_data: DataFrame[PPFSchema]) -> DataFrame[ScoringSchema]:

        # Compute EMWA of Adherenece, DM, PE
        data = self.compute_metrics_ewma(data)
        # Compute USAGE, and aggregate metrics by last value
        data = self.aggregate_metrics_per_protocol(data)
        # Merge with precomputed PPF data
        data = ppf_data.merge(data, on=["PATIENT_ID", "PROTOCOL_ID"], how="left")
        # Initialize missing values
        data = self.initialize_missing_metrics(data)
        # Compute objective function score alpha*Adherence + beta*DM + gamma*PPF
        score = self.compute_score(data)

        return score

    def compute_metrics_ewma(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies EWMA smoothing to difficulty modulator and performance estimator metrics.

        :param data: Merged DataFrame containing session and time-series data.
        :return: DataFrame with EWMA-applied values.
        """
        data = data.copy()
        
        data["ADHERNCE"] = (
            data.groupby("SESSION_ID")["ADHERENCE"]
            .transform(lambda x: x.ewm(alpha=self.alpha, adjust=True).mean())
        )
        data["DM_VALUE"] = (
            data.groupby("SESSION_ID")["DM_VALUE"]
            .transform(lambda x: x.ewm(alpha=self.alpha, adjust=True).mean())
        )
        data["PE_VALUE"] = (
            data.groupby("SESSION_ID")["PE_VALUE"]
            .transform(lambda x: x.ewm(alpha=self.alpha, adjust=True).mean())
        )

        return data

    def aggregate_metrics_per_protocol(self, data: pd.DataFrame) -> pd.DataFrame:
        aggregated_data = (
            data.groupby(["PATIENT_ID", "PROTOCOL_ID"])
            .agg(
                ADHERENCE=("ADHERENCE", "last"),
                DM_VALUE=("DM_VALUE", "last"),
                PE_VALUE=("PE_VALUE", "mean"),
                USAGE=("SESSION_ID", "count"),
                DAYS=("WEEKDAY_INDEX", lambda x: sorted(x[data.loc[x.index, "PRESCRIPTION_ENDING_DATE"] == "2100-01-01"].unique())),

            )
            .reset_index()
        )

        return aggregated_data

    def initialize_missing_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.fillna(0)
        return data
    
    def compute_score(self, scoring: pd.DataFrame) -> pd.DataFrame:
        """Compute the final scoring based on weights."""
        scoring_df = scoring.copy()
        scoring_df['SCORE'] = (
            scoring_df['ADHERENCE'] * self.weights[0] +
            scoring_df['DM_VALUE'] * self.weights[1] +
            scoring_df['PPF'] * self.weights[2]
        )
        return scoring_df
