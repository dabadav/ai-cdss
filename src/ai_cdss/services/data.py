from abc import ABC, abstractmethod
import importlib.resources
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import pandera as pa
from pandera.typing import DataFrame
import json
import yaml

from ai_cdss.models import SessionSchema, SessionProcessedSchema, PatientSchema, ProtocolMatrixSchema
from recsys_interface.data.interface import fetch_rgs_data, fetch_timeseries_data
from ai_cdss import config

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
    
    def get(self, key, default=None):
        """Alias for __getitem__ to allow mkd.get('key') syntax."""
        try:
            return self[key]
        except KeyError:
            return default
    
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

    def items(self):
        """Return a view of the primary keys and their values."""
        return self._data.items()

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
        """Save MultiKeyDict to a JSON file in the new format."""
        keys_dict = {}
        for alias, primary_key in self._keys.items():
            if primary_key in keys_dict:
                keys_dict[primary_key].append(alias)
            else:
                keys_dict[primary_key] = [alias]
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"data": self._data, "keys": keys_dict}, f, indent=4)

    @classmethod
    def from_json(cls, filepath):
        """Load MultiKeyDict from a JSON file in the new format."""
        with open(filepath, "r", encoding="utf-8") as f:
            obj = json.load(f)
        instance = cls()
        instance._data = obj["data"]
        instance._keys = {}

        for primary_key, aliases in obj["keys"].items():
            for alias in aliases:
                instance._keys[alias] = primary_key
        
        return instance

    # --- YAML Serialization Methods ---
    def to_yaml(self, filepath):
        """Save MultiKeyDict to a YAML file in the new format."""
        keys_dict = {}
        for alias, primary_key in self._keys.items():
            if primary_key in keys_dict:
                keys_dict[primary_key].append(alias)
            else:
                keys_dict[primary_key] = [alias]

        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump({"data": self._data, "keys": keys_dict}, f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, filepath):
        """Load MultiKeyDict from a YAML file in the new format."""
        with open(filepath, "r", encoding="utf-8") as f:
            obj = yaml.safe_load(f)
        instance = cls()
        instance._data = obj["data"]
        instance._keys = {}

        for primary_key, aliases in obj["keys"].items():
            for alias in aliases:
                instance._keys[alias] = primary_key
        
        return instance

class PatientData:
    """Container for patient data and associated metadata (max values)."""
    def __init__(self, data: DataFrame, max_values: Dict[str, float]):
        self.data = data
        self.max_values = max_values

    @property
    def deficit_matrix(self) -> DataFrame:
        """Compute the deficit matrix using the data and max values."""
        return 1 - (self.data / pd.Series(self.max_values))

def create_patient_schema(max_values: Dict[str, float]) -> pa.DataFrameModel:
    """Dynamically create a Pandera schema based on max values."""
    class PatientSchema(pa.DataFrameModel):
        pass

    # Add columns to the schema with range checks
    for col_name, max_value in max_values.items():
        setattr(
            PatientSchema,
            col_name,
            pa.Column(
                float,
                checks=pa.Check.in_range(0, max_value),
                nullable=False
            )
        )

    return PatientSchema

class PatientDataLoader:
    """Load and validate patient data with associated max values."""
    def __init__(self, max_values: Dict[str, float]):
        self.max_values = max_values
        self.PatientSchema = create_patient_schema(max_values)

    @pa.check_types
    def load_patient_data(self) -> DataFrame[self.PatientSchema]:
        """Load and validate patient data."""
        data = pd.read_csv("../../data/clinical_scores.csv", index_col=0)
        return data

    def get_patient_data(self) -> PatientData:
        """Return a PatientData object with validated data and max values."""
        data = self.load_patient_data()
        return PatientData(data, self.max_values)

#####################################
# ------ Protocol Attributes ------ #
#####################################

class ProtocolData:


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

def expand_session_batch(session_batch: pd.DataFrame):
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

        expected_dates = generate_expected_sessions(start, end, int(weekday))
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

def map_latent_to_clinical(protocol_attributes, mapping_dict, agg_func=np.mean):
    """We need to collapse the protocol feature space into the clinical feature space.
    """
    df_clinical = pd.DataFrame(index=protocol_attributes.index)

    # Collapse using agg_func the protocol latent attributes    
    for clinical_scale, features in mapping_dict.items():
        df_clinical[clinical_scale] = protocol_attributes[features].apply(agg_func, axis=1)

    df_clinical.index = protocol_attributes["PROTOCOL_ID"]

    return df_clinical

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

# ------------------------------
# ------ Clinical Scores

class ClinicalProcessor:
    def __init__(self, scale_yaml_path: Optional[str] = None):
        """Initialize with an optional path to scale.yaml, defaulting to internal package resource."""
        if scale_yaml_path:
            self.scales_path = Path(scale_yaml_path)
        else:
            self.scales_path = importlib.resources.files(config) / "scales.yaml"

        if not self.scales_path.exists():
            raise FileNotFoundError(f"Scale YAML file not found at {self.scales_path}")

        # logger.info(f"Loading subscale max values from: {self.scales_path}")
        self.scales_dict = MultiKeyDict.from_yaml(self.scales_path)

    def process(self, patient_df: pd.DataFrame) -> pd.DataFrame:
        """Compute deficit matrix given patient clinical scores.

        Args:
            patient_df (pd.DataFrame): DataFrame where columns are subscale names.

        Returns:
            pd.DataFrame: A deficit matrix (values normalized between 0 and 1).
        """
        # Retrieve max values using MultiKeyDict
        max_subscales = [self.scales_dict.get(scale, None) for scale in patient_df.columns]

        # Check for missing values
        if None in max_subscales:
            missing_subscales = [scale for scale, max_val in zip(patient_df.columns, max_subscales) if max_val is None]
            # logger.warning(f"Missing max values for subscales: {missing_subscales}")
            raise ValueError(f"Missing max values for subscales: {missing_subscales}")

        # Compute deficit matrix
        deficit_matrix = 1 - (patient_df / pd.Series(max_subscales, index=patient_df.columns))

        # Save (Optional: Remove in production if not needed)
        # deficit_matrix.to_csv("deficit_matrix.csv")
        # logger.info("Deficit matrix computed and saved to 'deficit_matrix.csv'")

        return deficit_matrix

# ------------------------------
# ------ Session Data

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

# ---------------------------------------------------------------------
# RETURNS
# -------
# patient_df
# session_df
# protocol_df
