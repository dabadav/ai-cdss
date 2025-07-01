import pandas as pd
from ai_cdss.constants import (
    DAYS,
    PATIENT_ID,
    SESSION_INDEX,
    USAGE,
    USAGE_WEEK,
    WEEKS_SINCE_START,
)


class Imputer:
    def impute_metrics(
        self, data: pd.DataFrame, column: str, values: pd.DataFrame
    ) -> pd.DataFrame:
        """Impute missing values in the specified column using values from another DataFrame."""
        data_imputed = data.copy()
        merged = data_imputed.merge(
            values[[PATIENT_ID, column]],
            on=PATIENT_ID,
            how="left",
            suffixes=("", "_median"),
        )
        merged[column] = merged[column].fillna(merged[f"{column}_median"])
        merged.drop(columns=[f"{column}_median"], inplace=True)
        return merged

    def init_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Initialize metric columns with default values and correct types."""
        data[DAYS] = data[DAYS].apply(
            lambda x: [] if x is None or (not isinstance(x, list) and pd.isna(x)) else x
        )
        data[USAGE] = data[USAGE].astype("Int64").fillna(0)
        data[USAGE_WEEK] = data[USAGE_WEEK].astype("Int64").fillna(0)
        data[SESSION_INDEX] = data[SESSION_INDEX].astype("Int64").fillna(0)
        data[WEEKS_SINCE_START] = data[WEEKS_SINCE_START].astype("Int64").fillna(0)
        return data
