import pandas as pd
from ai_cdss.constants import *


class Scorer:
    def __init__(self, weights=None):
        self.weights = weights or [1, 1, 1]

    def compute_score(self, data: pd.DataFrame) -> pd.DataFrame:
        score = self._compute_score(data)
        score.sort_values(by=BY_PP, inplace=True)
        return score

    def _compute_score(self, scoring: pd.DataFrame) -> pd.DataFrame:
        scoring_df = scoring.copy()
        scoring_df[SCORE] = (
            scoring_df[RECENT_ADHERENCE].astype("float64").fillna(0.0) * self.weights[0]
            + scoring_df[DELTA_DM].astype("float64").fillna(0.0) * self.weights[1]
            + scoring_df[PPF].astype("float64").fillna(0.0) * self.weights[2]
        )
        return scoring_df
