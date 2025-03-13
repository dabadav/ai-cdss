from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import pandera as pa
import json
import functools
from typing import Dict, List
from pandera.typing import DataFrame
from ai_cdss.models import SessionSchema, SessionProcessedSchema, PatientSchema, ProtocolMatrixSchema
from recsys_interface.data.interface import fetch_rgs_data, fetch_timeseries_data

# ---------------------------------------------------------------------
# PARAMS
# ------
# patient_list: List

#################################
# ------ RGS Data Loader ------ #
#################################

class BaseDataLoader(ABC):
    """Abstract class for data loading strategies."""

    @abstractmethod
    def load_session_data(self) -> DataFrame[SessionSchema]:
        pass

    @abstractmethod
    def load_patient_data(self) -> DataFrame[PatientSchema]:
        pass

    @abstractmethod
    def load_protocol_data(self) -> DataFrame[ProtocolMatrixSchema]:
        pass

    @abstractmethod
    def load_timeseries_data(self) -> pd.DataFrame:
        pass

class DataLoader(BaseDataLoader):
    """Loads data from database and CSV files."""

    def __init__(self, patient_list):
        self.patient_list = patient_list

    @pa.check_types
    def load_session_data(self) -> DataFrame[SessionSchema]:
        # TODO: rgs_mode param?
        data_app = fetch_rgs_data(self.patient_list, rgs_mode="app")
        return data_app
    
    def load_timeseries_data(self):
        # TODO: rgs_mode param?
        dms_app = pd.read_csv("../../data/app_timeseries.csv")
        return dms_app

    def load_patient_data(self) -> DataFrame[PatientSchema]:
        return pd.read_csv("../../data/clinical_scores.csv", index_col=0)

    def load_protocol_data(self) -> DataFrame[ProtocolMatrixSchema]:
        return pd.read_csv("../../data/protocol_attributes.csv", index_col=0)


#################################
# ------ Clinical Scales ------ #
#################################

class MultiKeyDict(object):

    def __init__(self, **kwargs):
        self._keys = {}
        self._data = {}
        for k, v in kwargs.items():
            self[k] = v

    def __getitem__(self, key):
        try:
            return self._data[key]
        except KeyError:
            return self._data[self._keys[key]]

    def __setitem__(self, key, val):
        try:
            self._data[self._keys[key]] = val
        except KeyError:
            if isinstance(key, tuple):
               if not key:
                  raise ValueError('Empty tuple cannot be used as a key')
               key, other_keys = key[0], key[1:]
            else:
               other_keys = []
            self._data[key] = val
            for k in other_keys:
                self._keys[k] = key

    def __repr__(self):
        return f"MultiKeyDict(data={self._data}, keys={self._keys})"
    
    __str__ = __repr__ 

    def add_keys(self, to_key, new_keys):
        if to_key not in self._data:
            to_key = self._keys[to_key]
        for key in new_keys:
            self._keys[key] = to_key

    @classmethod
    def from_dict(cls, dic):
        result = cls()
        for key, val in dic.items():
            result[key] = val
        return result
    
    # --- Serialization Methods ---
    def to_json(self, filepath):
        """Save MultiKeyDict to a JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"data": self._data, "keys": self._keys}, f, indent=4)

    @classmethod
    def from_json(cls, filepath):
        """Load MultiKeyDict from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            obj = json.load(f)
        instance = cls()
        instance._data = obj["data"]
        instance._keys = obj["keys"]
        return instance


#################################
# ------ Data Processing ------ #
#################################

class BaseDataProcessor(ABC):
    """Abstract class for different processing strategies."""

    @abstractmethod
    def process(self, df: pd.DataFrame):
        pass

class DataProcessor:
    """
    Data repository class

    This class will load and process Session data, Timeseries data, Patient data, Protocol data.
    """
    def __init__(self, mapping_dict):
        self.session_batch = None
        self.patient_data = None
        self.protocol_data = None
        self.mapping_dict = mapping_dict

    # def process_session_data(self, session_batch: DataFrame[SessionSchema]) -> DataFrame[SessionProcessedSchema]:
    #     """
    #     Clean and standardize the session DataFrame:
    #     - Enforce expected data types for each column.
    #     - Drop sessions with missing critical values.
    #     - Convert date columns to datetime.date.
    #     - Map weekday names to numeric codes (0=Monday, ..., 6=Sunday).
    #     """
    #     pass

    def process_timeseries_data(self, timeseries_df):
        """
        Process raw time-series data:
        - Ensure numeric types for parameter and performance values.
        - Aggregate entries with the same timestamp (seconds from start) within each session.
        - Compute exponential moving averages (EWMA) of parameter and performance values to smooth fluctuations.
        """
        # Cast parameter and performance values to float
        timeseries_df["PARAMETER_VALUE"] = timeseries_df["PARAMETER_VALUE"].astype(float)
        timeseries_df["PERFORMANCE_VALUE"] = timeseries_df["PERFORMANCE_VALUE"].astype(float)
        # Aggregate by session and time point (unique timestamp within a session), averaging multiple entries if any
        aggregated = timeseries_df.groupby(
            ["PATIENT_ID", "SESSION_ID", "PROTOCOL_ID", "GAME_MODE", "SECONDS_FROM_START"]
        ).agg({
            "PARAMETER_KEY": lambda x: list(set(x)),       # unique parameters at this time
            "PARAMETER_VALUE": "mean",                     # average parameter value
            "PERFORMANCE_KEY": "first",                    # assume same performance key per time, take first
            "PERFORMANCE_VALUE": "mean"                    # average performance value (usually only one)
        }).reset_index()
        # Compute Exponential Weighted Moving Average (EWMA) for values within each session time-series
        # This smooths out the instantaneous fluctuations in DMs and PEs.
        aggregated.sort_values(by=["PATIENT_ID", "SESSION_ID", "SECONDS_FROM_START"], inplace=True)

        
        aggregated["PARAMETER_VALUE_EWMA"] = aggregated.groupby(["PATIENT_ID", "SESSION_ID"])["PARAMETER_VALUE"]\
                                                    .transform(lambda x: x.ewm(alpha=0.5, adjust=False).mean())
        

        aggregated["PERFORMANCE_VALUE_EWMA"] = aggregated.groupby(["PATIENT_ID", "SESSION_ID"])["PERFORMANCE_VALUE"]\
                                                    .transform(lambda x: x.ewm(alpha=0.5, adjust=False).mean())
        
        return aggregated

    def process_patient_data(self, patient_df: DataFrame[PatientSchema], max_subscales) -> DataFrame[PatientSchema]:
        """Process the clinical scores to compute the deficits given maximum value per subscale
        """
        ### TODO: How to handle clinical subscales max scores?? as param
        ### - Store a dict of the max values for all possible subscales envisioned giving flexibility to name
        ### - Retrieve the max value from there given the subscales present in my subset

        deficit_matrix = 1 - (patient_df / pd.Series(max_subscales))
        deficit_matrix.to_csv("deficit_matrix")
        return deficit_matrix

    def expand_session_batch(self, session_batch: pd.DataFrame):
        """
        Augment the session DataFrame with rows for expected sessions that were *not* performed.
        For each prescription (identified by PRESCRIPTION_ID), generate sessions on the scheduled weekday 
        that do not exist in the recorded sessions, marking them as NOT_PERFORMED with zero adherence.
        """
        session_cols = [
            "SESSION_ID", "STARTING_HOUR", "STARTING_TIME_CATEGORY", "REAL_SESSION_DURATION",
            "SESSION_DURATION", "TOTAL_SUCCESS", "TOTAL_ERRORS", "SCORE"
        ]

        missing_sessions = []
        # Group sessions by prescription and find expected dates not present
        for _, group in session_batch.groupby("PRESCRIPTION_ID"):
            
            # Use the first row of each group to get prescription schedule info
            first_row = group.iloc[0]
            start = first_row.PRESCRIPTION_STARTING_DATE
            end = first_row.PRESCRIPTION_ENDING_DATE
            weekday = first_row.WEEKDAY_INDEX

            ###### NA Issue 
            if pd.isna(start) or pd.isna(end) or pd.isna(weekday):
                continue  # skip if any critical info is missing

            expected_dates = self.generate_expected_sessions(start, end, int(weekday))
            performed_dates = set(group["SESSION_DATE"].dropna().unique())
            
            # Any expected date not in performed_dates is a missed session
            for miss_date in expected_dates:

                if miss_date not in performed_dates:

                    new_row = first_row.copy()
                    new_row["SESSION_DATE"] = miss_date
                    new_row["STATUS"] = "NOT_PERFORMED"
                    new_row["ADHERENCE"] = 0.0

                    # Set all session outcome-related columns to NaN (since session didn't occur)
                    for col in session_cols:
                        new_row[col] = np.nan

                    missing_sessions.append(new_row)

        if missing_sessions:
            
            df_missing = pd.DataFrame(missing_sessions)
            sessions_df = pd.concat([session_batch, df_missing], ignore_index=True)
            
            # Sort by prescription and date for chronological order
            sessions_df.sort_values(by=["PRESCRIPTION_ID", "PROTOCOL_ID", "SESSION_DATE"], inplace=True)
            
        return sessions_df

    def map_latent_to_clinical(self, protocol_attributes, agg_func=np.mean):
        """We need to collapse the protocol feature space into the clinical feature space.
        """
        df_clinical = pd.DataFrame(index=protocol_attributes.index)

        # Collapse using agg_func the protocol latent attributes    
        for clinical_scale, features in self.mapping_dict.items():
            df_clinical[clinical_scale] = protocol_attributes[features].apply(agg_func, axis=1)

        df_clinical.index = protocol_attributes["PROTOCOL_ID"]
        self.protocol_data = df_clinical

        return df_clinical
    
    @staticmethod
    def generate_expected_sessions(start_date, end_date, target_weekday):
        """
        Generate all expected session dates between start_date and end_date for the given target weekday.
        If the prescription end date is in the future, use today as the end limit (assuming future sessions are not yet done).
        """
        expected_dates = []
        today = pd.Timestamp.today()
        # If the prescription is still ongoing, cap the end_date at today
        if end_date is None:
            return expected_dates  # no valid end date
        if end_date > today:
            end_date = today
        # Find the first occurrence of the target weekday on or after start_date
        if start_date.weekday() != target_weekday:
            days_until_target = (target_weekday - start_date.weekday()) % 7
            start_date = start_date + pd.Timedelta(days=days_until_target)
        # Generate dates every 7 days (weekly) from the adjusted start_date up to end_date
        current_date = start_date
        while current_date <= end_date:
            expected_dates.append(current_date)
            current_date += pd.Timedelta(days=7)
        return expected_dates

class SessionProcessor:
    def __init__(
        self, 
        expected_dtypes: Dict[str, str], 
        critical_columns: List[str], 
        date_columns: List[str],
        weekday_column: str
    ):
        self.expected_dtypes = expected_dtypes
        self.critical_columns = critical_columns
        self.date_columns = date_columns
        self.weekday_col = weekday_column
        self.weekday_map = {
            "MONDAY": 0, "TUESDAY": 1, "WEDNESDAY": 2,
            "THURSDAY": 3, "FRIDAY": 4, "SATURDAY": 5, "SUNDAY": 6
        }

    # Each method is pure, accepting and returning DataFrames explicitly
    def enforce_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.astype(self.expected_dtypes, errors='raise')

    def drop_missing_critical(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna(subset=self.critical_columns)

    def convert_dates_to_date(self, df: pd.DataFrame, date_cols: List[str]) -> pd.DataFrame:
        return df.assign(**{
            col: pd.to_datetime(df[col], errors='coerce').dt.date for col in date_cols
        })

    def map_weekdays(self, df: pd.DataFrame, weekday_col: str) -> pd.DataFrame:
        return df.assign(**{
            weekday_col: df[weekday_col].map(self.weekday_map).astype("Int64")
        })

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Chains all processing steps together."""
        return (
            df.pipe(self.enforce_dtypes)
              .pipe(self.drop_missing_critical)
              .pipe(self.convert_dates_to_date, self.date_columns)
              .pipe(self.map_weekdays, self.weekday_col)
        )

def enforce_dtypes(df: pd.DataFrame, dtypes: Dict[str, str]) -> pd.DataFrame:
    """Enforce data types explicitly, raise if incompatible."""
    return df.astype(dtypes, errors='raise')

def drop_missing_critical(df: pd.DataFrame, critical_cols: list) -> pd.DataFrame:
    """Drop rows missing critical columns."""
    return df.dropna(subset=critical_cols)

def convert_dates_to_date(df: pd.DataFrame, date_cols: list) -> pd.DataFrame:
    """Convert datetime columns to date."""
    return df.assign(**{
        col: pd.to_datetime(df[col], errors='coerce').dt.date for col in date_cols
    })

def process_session_data(
    self, 
    session_batch: pd.DataFrame
) -> pd.DataFrame:
    expected_dtypes = {
        "PATIENT_ID": "Int64",
        "HOSPITAL_ID": "Int64",
        "PARETIC_SIDE": "string",
        "UPPER_EXTREMITY_TO_TRAIN": "string",
        "HAND_RAISING_CAPACITY": "string",
        "COGNITIVE_FUNCTION_LEVEL": "string",
        "HAS_HEMINEGLIGENCE": "Int64",
        "GENDER": "string",
        "SKIN_COLOR": "string",
        "AGE": "Int64",
        "VIDEOGAME_EXP": "Int64",
        "COMPUTER_EXP": "Int64",
        "COMMENTS": "string",
        "PTN_HEIGHT_CM": "Int64",
        "ARM_SIZE_CM": "Int64",
        "PRESCRIPTION_ID": "Int64",
        "SESSION_ID": "Int64",
        "PROTOCOL_ID": "Int64",
        "PRESCRIPTION_STARTING_DATE": "datetime64[ns]",
        "PRESCRIPTION_ENDING_DATE": "datetime64[ns]",
        "SESSION_DATE": "datetime64[ns]",
        "STARTING_HOUR": "Int64",
        "STARTING_TIME_CATEGORY": "string",
        "STATUS": "string",
        "PROTOCOL_TYPE": "string",
        "AR_MODE": "string",
        "WEEKDAY": "string",
        "REAL_SESSION_DURATION": "Int64",
        "PRESCRIBED_SESSION_DURATION": "Int64",
        "SESSION_DURATION": "Int64",
        "ADHERENCE": "float64",
        "TOTAL_SUCCESS": "Int64",
        "TOTAL_ERRORS": "Int64",
        "SCORE": "float64"
    }

    critical_cols = ["PRESCRIPTION_ID", "SESSION_DURATION"]
    date_cols = ["SESSION_DATE", "PRESCRIPTION_STARTING_DATE", "PRESCRIPTION_ENDING_DATE"]

    return (
        session_batch
        .pipe(enforce_dtypes, expected_dtypes)
        .pipe(drop_missing_critical, critical_cols)
        .pipe(convert_dates_to_date, date_cols)
        .pipe(map_weekdays, "WEEKDAY")
    )

# ---------------------------------------------------------------------
# RETURNS
# -------

