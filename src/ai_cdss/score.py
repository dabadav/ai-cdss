"""Scoring + imputation — one file, two trivial classes.

Both classes were single-method modules in v0.3.1. Folded together
here because they share a contract (operate on the per-(PP) "scoring
input" frame) and are too small to justify separate files.

    Imputer   — fill NaNs in DELTA_DM / RECENT_ADHERENCE etc. so the
                Scorer can compute a SCORE without `NaN` propagation.
    Scorer    — linear combination of (RECENT_ADHERENCE, DELTA_DM, PPF)
                with configurable weights. One row in, one row out;
                no aggregation.

Both classes are stateless aside from constructor configuration.
"""
from __future__ import annotations

import pandas as pd

from ai_cdss.constants import (
    BY_PP,
    DAYS,
    DELTA_DM,
    PATIENT_ID,
    PPF,
    RECENT_ADHERENCE,
    SCORE,
    SESSION_INDEX,
    USAGE,
    USAGE_WEEK,
    WEEKS_SINCE_START,
)


# ─────────────────────────────────────────────────────────────────────
# Imputer

class Imputer:
    """Fill NaNs and seed default values for the scoring input frame.

    Two operations:
      `init_metrics` — coerce dtypes + zero-fill the count columns
                       (USAGE, USAGE_WEEK, SESSION_INDEX,
                       WEEKS_SINCE_START) and default DAYS to an empty
                       list.
      `impute_metrics` — fill NaNs in a target column with a per-patient
                       median (passed in as a separate frame).
    """

    def init_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Coerce count-style columns to Int64 + zero-fill. DAYS gets an
        empty list whenever it's NaN/None."""
        data[DAYS] = data[DAYS].apply(
            lambda x: [] if x is None or (not isinstance(x, list) and pd.isna(x)) else x
        )
        data[USAGE] = data[USAGE].astype("Int64").fillna(0)
        data[USAGE_WEEK] = data[USAGE_WEEK].astype("Int64").fillna(0)
        data[SESSION_INDEX] = data[SESSION_INDEX].astype("Int64").fillna(0)
        data[WEEKS_SINCE_START] = data[WEEKS_SINCE_START].astype("Int64").fillna(0)
        return data

    def impute_metrics(
        self, data: pd.DataFrame, column: str, values: pd.DataFrame,
    ) -> pd.DataFrame:
        """Fill NaNs in `data[column]` with the per-patient value from
        `values[PATIENT_ID, column]`. Left-merges, fills, drops the
        join column."""
        imputed = data.copy()
        merged = imputed.merge(
            values[[PATIENT_ID, column]],
            on=PATIENT_ID, how="left", suffixes=("", "_median"),
        )
        merged[column] = merged[column].fillna(merged[f"{column}_median"])
        merged.drop(columns=[f"{column}_median"], inplace=True)
        return merged


# ─────────────────────────────────────────────────────────────────────
# Scorer

class Scorer:
    """Linear combination scoring: weighted sum of three metric columns.

    `SCORE = w0 * RECENT_ADHERENCE + w1 * DELTA_DM + w2 * PPF`

    Default weights = [1, 1, 1] (equal). NaNs are filled with 0 inside
    the formula so a missing component doesn't drag the score to NaN.
    """

    def __init__(self, weights: list[float] | None = None) -> None:
        self.weights = weights or [1, 1, 1]

    def compute_score(self, data: pd.DataFrame) -> pd.DataFrame:
        scored = data.copy()
        scored[SCORE] = (
            scored[RECENT_ADHERENCE].astype("float64").fillna(0.0) * self.weights[0]
            + scored[DELTA_DM].astype("float64").fillna(0.0) * self.weights[1]
            + scored[PPF].astype("float64").fillna(0.0) * self.weights[2]
        )
        return scored.sort_values(by=BY_PP)
