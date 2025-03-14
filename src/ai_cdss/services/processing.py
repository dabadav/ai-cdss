from abc import ABC, abstractmethod
import importlib.resources
from typing import Dict, List, Optional, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import json
import yaml

from ai_cdss import config

#################################
# ------ Data Processing ------ #
#################################

def expand_session_batch(session: pd.DataFrame):
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
    for _, group in session.groupby("PRESCRIPTION_ID"):
        
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
        sessions_df = pd.concat([session, df_missing], ignore_index=True)
        
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


class BaseDataProcessor(ABC):
    """Abstract class for different processing strategies."""

    @abstractmethod
    def process(self, df: pd.DataFrame):
        pass

# ------------------------------
# ------ Timeseries Sessions

class TimeseriesProcessor(BaseDataProcessor):
    """
    Data repository class

    This class will load and process Session data, Timeseries data, Patient data, Protocol data.
    """
    def __init__(self):
        self.session = None
        self.patient_deficiency = None
        self.protocol_mapped = None

    def process(self, timeseries):
        """
        Process raw time-series data:
        - Ensure numeric types for parameter and performance values.
        - Aggregate entries with the same timestamp (seconds from start) within each session.
        - Compute exponential moving averages (EWMA) of parameter and performance values to smooth fluctuations.
        """
        
        timeseries_df = timeseries.copy()
        
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
        
        # Last value
        aggregated = aggregated.groupby(["PATIENT_ID", "PROTOCOL_ID"])["PARAMETER_VALUE_EWMA"].last().reset_index()
        return aggregated

# ------------------------------
# ------ Clinical Scores

class ClinicalProcessor(BaseDataProcessor):
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
# ------ Protocol Attributes

class ProtocolProcessor(BaseDataProcessor):
    def __init__(self, mapping_yaml_path: Optional[str] = None):
        """Initialize with an optional path to scale.yaml, defaulting to internal package resource."""
        if mapping_yaml_path:
            self.mapping_path = Path(mapping_yaml_path)
        else:
            self.mapping_path = importlib.resources.files(config) / "mapping.yaml"

        if not self.mapping_path.exists():
            raise FileNotFoundError(f"Scale YAML file not found at {self.mapping_path}")

        # logger.info(f"Loading subscale max values from: {self.scales_path}")
        self.mapping = MultiKeyDict.from_yaml(self.mapping_path)

    def process(self, protocol_df: pd.DataFrame, agg_func=np.mean) -> pd.DataFrame:
        """Compute deficit matrix given patient clinical scores.
        """
        # Retrieve max values using MultiKeyDict
        df_clinical = pd.DataFrame(index=protocol_df.index)

        # Collapse using agg_func the protocol latent attributes    
        for clinical_scale, features in self.mapping.items():
            df_clinical[clinical_scale] = protocol_df[features].apply(agg_func, axis=1)

        df_clinical.index = protocol_df["PROTOCOL_ID"]

        return df_clinical

# ------------------------------
# ------ Session Data

class SessionProcessor(BaseDataProcessor):
    def __init__(self):
        pass
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Chains all processing steps together."""
        return (
            df
            .pipe(self.compute_adherence)
            .pipe(self.last_value)
        )

    @staticmethod
    def compute_adherence(session, alpha=0.8):
        """ Compute adherence scores.
        """
        session['ADHERENCE_EWMA'] = (
            session
            .groupby(['PATIENT_ID', 'PROTOCOL_ID'])['ADHERENCE']
            .transform(lambda x: x.ewm(alpha=alpha, adjust=True).mean())
        )
        return session

    @staticmethod
    def last_value(session):
        session_df = session.copy()
        session_last = session_df.groupby(["PATIENT_ID", "PROTOCOL_ID"])["ADHERENCE_EWMA"].last().reset_index()
        return session_last

# ------------------------------
# ------ Utils

def compute_adherence(session, alpha=0.8):
    """ Compute adherence scores.
    """
    session['ADHERENCE_EWMA'] = (
        session
        .groupby(['PATIENT_ID', 'PROTOCOL_ID'])['ADHERENCE']
        .transform(lambda x: x.ewm(alpha=alpha, adjust=True).mean())
    )
    return session

def feature_contributions(df_A, df_B):
    # Convert to numpy
    A = df_A.to_numpy()
    B = df_B.to_numpy()

    # Compute row-wise norms
    A_norms = np.linalg.norm(A, axis=1, keepdims=True)
    B_norms = np.linalg.norm(B, axis=1, keepdims=True)
    
    # Replace zero norms with a small value to avoid NaN (division by zero)
    A_norms[A_norms == 0] = 1e-10
    B_norms[B_norms == 0] = 1e-10

    # Normalize each row to unit vectors
    A_norm = A / A_norms
    B_norm = B / B_norms

    # Compute feature contributions
    contributions = A_norm[:, np.newaxis, :] * B_norm[np.newaxis, :, :]

    return contributions

def compute_ppf(patient_deficiency, protocol_mapped):
    """ Compute the patient-protocol feature matrix (PPF) and feature contributions.
    """
    contributions = feature_contributions(patient_deficiency, protocol_mapped)
    ppf = np.sum(contributions, axis=2)
    ppf = pd.DataFrame(ppf, index=patient_deficiency.index, columns=protocol_mapped.index)
    contributions = pd.DataFrame(contributions.tolist(), index=patient_deficiency.index, columns=protocol_mapped.index)
    
    ppf_long = ppf.stack().reset_index()
    ppf_long.columns = ["PATIENT_ID", "PROTOCOL_ID", "PPF"]

    contrib_long = contributions.stack().reset_index()
    contrib_long.columns = ["PATIENT_ID", "PROTOCOL_ID", "CONTRIB"]

    return ppf_long, contrib_long

def create_multiindex_df(
    patient_ids: Iterable,
    protocol_ids: Iterable,
    index_names: List[str] = ["PATIENT_ID", "PROTOCOL_ID"]
) -> pd.DataFrame:
    return (
        pd.MultiIndex.from_product(
            [patient_ids, protocol_ids],
            names=index_names
        )
        .to_frame(index=False)
        .reset_index(drop=True)
    )

multiindex = lambda id_a, id_b: pd.MultiIndex.from_product([id_a, id_b], names=["PATIENT_ID", "PROTOCOL_ID"])

def compute_usage(session: pd.DataFrame, index: pd.MultiIndex) -> pd.DataFrame:
    """
    Compute the number of sessions per (PATIENT_ID, PROTOCOL_ID).
    
    Parameters:
    - session (pd.DataFrame): DataFrame containing session data with "SESSION_ID".
    - index (pd.MultiIndex): MultiIndex to ensure all patient-protocol combinations exist.
    
    Returns:
    - pd.DataFrame: DataFrame with "USAGE" column.
    """
    return (
        session
        .groupby(["PATIENT_ID", "PROTOCOL_ID"])["SESSION_ID"]
        .agg(USAGE="count")
        .reindex(index, fill_value=0)
        .reset_index()
    )

# For timeseries DM PERFORMANCE
def extract_last_value(df: pd.DataFrame, group_cols: list, value_col: str, new_col_name: str = None) -> pd.DataFrame:
    """
    Groups a DataFrame by specified columns and extracts the last value for a given column.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - group_cols (list): The columns to group by (e.g., ["PATIENT_ID", "PROTOCOL_ID"]).
    - value_col (str): The column to extract the last value from.
    - new_col_name (str, optional): The name of the output column. If None, keeps the original name.
    
    Returns:
    - pd.DataFrame: A DataFrame with group_cols + the last value of value_col.
    """
    new_col_name = new_col_name or value_col  # Use custom name if provided
    return (
        df
        .groupby(group_cols)[value_col]
        .last()
        .reset_index()
        .rename(columns={value_col: new_col_name})
    )

extract_last = lambda df, value_col, new_col: (
    df.groupby(["PATIENT_ID", "PROTOCOL_ID"])[value_col]
    .last()
    .reset_index()
    .rename(columns={value_col: new_col})
)

# JOIN
def merge_data(left, right):
    return pd.merge(left, right, on=["PATIENT_ID", "PROTOCOL_ID"], how="left")

def top_n_protocols(df, group_col="PATIENT_ID", score_col="SCORE", n=10):
    """
    Selects the top N protocols based on SCORE for each patient.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - group_col (str): Column to group by (e.g., 'PATIENT_ID').
    - score_col (str): The column used for ranking (e.g., 'SCORE').
    - n (int): Number of top values to keep per group.

    Returns:
    - pd.DataFrame: DataFrame containing the top N protocols per patient.
    """
    return (
        df.groupby(group_col)
        .apply(lambda x: x.nlargest(n, score_col))
        .reset_index(drop=True)
    )

rank_top_n = lambda df, n: (
    df.groupby("PATIENT_ID")
    .apply(lambda x: x.nlargest(n, "SCORE"))
    .reset_index(drop=True)
)

def schedule(df, days_per_week=7, prescriptions_per_day=5):
    """
    Generates a weekly schedule for each patient by distributing their top recommended protocols across the week.
    Ensures that:
    1. The same protocol is not scheduled twice in a single day.
    2. The total number of prescriptions is exactly `days_per_week * prescriptions_per_day`.
    
    Args:
    df (pd.DataFrame): Long format DataFrame with columns ['PATIENT_ID', 'PROTOCOL_ID'].
    days_per_week (int): Number of days in the schedule (default: 7).
    prescriptions_per_day (int): Number of protocols per day (default: 5).
    
    Returns:
    pd.DataFrame: A DataFrame where each row corresponds to a (PATIENT_ID, PROTOCOL_ID) pair,
                and the 'DAYS' column contains a list of day indexes (1-based) for when the protocol should be played.
    """
    total_prescriptions = days_per_week * prescriptions_per_day
    schedule_dict = {}

    for patient_id, group in df.groupby("PATIENT_ID"):
        protocols = group["PROTOCOL_ID"].tolist()

        # Expand protocol list to ensure at least `total_prescriptions`
        expanded_protocols = (protocols * ((total_prescriptions // len(protocols)) + 1))[:total_prescriptions]

        # Shuffle protocols for distribution across days
        np.random.shuffle(expanded_protocols)

        # Assign protocols to days ensuring no duplicates in a single day
        patient_schedule = {protocol: [] for protocol in protocols}
        day_protocols = [[] for _ in range(days_per_week)]
        
        for i, protocol in enumerate(expanded_protocols):
            day_idx = i % days_per_week
            if protocol not in day_protocols[day_idx]:  # Ensure no duplicate protocol on the same day
                day_protocols[day_idx].append(protocol)
                patient_schedule[protocol].append(day_idx + 1)  # Use 1-based indexing for days

        schedule_dict[patient_id] = patient_schedule

    # Convert to long format DataFrame
    structured_schedule = []
    for patient_id, protocols in schedule_dict.items():
        for protocol_id, days in protocols.items():
            structured_schedule.append({"PATIENT_ID": patient_id, "PROTOCOL_ID": protocol_id, "DAYS": days})

    schedule_df = pd.DataFrame(structured_schedule)
    df["DAYS"] = schedule_df.DAYS
    
    return df

below_mean = lambda x: x < x.mean()

interchange_mask = lambda df: (
    df.groupby('PATIENT_ID')['SCORE']
    .transform(below_mean)
)

def compute_protocol_similarity(protocol_mapped):
    """ Compute protocol similarity.
    """
    import gower

    protocol_attributes = protocol_mapped.copy()
    protocol_ids = protocol_attributes.PROTOCOL_ID
    protocol_attributes.drop(columns="PROTOCOL_ID", inplace=True)

    hot_encoded_cols = protocol_attributes.columns.str.startswith("BODY_PART")
    weights = np.ones(len(protocol_attributes.columns))
    weights[hot_encoded_cols] = weights[hot_encoded_cols] / hot_encoded_cols.sum()
    protocol_attributes = protocol_attributes.astype(float)

    gower_sim_matrix = gower.gower_matrix(protocol_attributes, weight=weights)
    gower_sim_matrix = pd.DataFrame(1- gower_sim_matrix, index=protocol_ids, columns=protocol_ids)
    gower_sim_matrix.columns.name = "PROTOCOL_SIM"

    return gower_sim_matrix

def matrix_to_xy(df, columns=None, reset_index=False):
    bool_index = np.triu(np.ones(df.shape)).astype(bool)
    xy = (
        df.where(bool_index).stack().reset_index()
        if reset_index
        else df.where(bool_index).stack()
    )
    if reset_index:
        xy.columns = columns or ["row", "col", "val"]
    return xy

def find_substitute(patient, protocol, protocol_sim, usage):
    # Exclude the current protocol
    protocols = protocol_sim.columns.drop(protocol)
    
    # Get usage and similarity data for other protocols
    protocol_usage = get_usage(usage, patient)
    usage = protocol_usage[protocols]
    sim = protocol_sim.loc[protocol, protocols]
    
    # Find the minimum usage value
    min_usage = usage.min()
    # Get candidates with the lowest usage
    candidates = usage[usage == min_usage].index
    
    # Among these candidates, select the one with highest similarity
    max_sim = sim[candidates].max()
    final_candidates = sim[sim == max_sim].index
    
    # Return the first candidate (or handle ties)
    return final_candidates[0] if not final_candidates.empty else None

def get_usage(session, patient_id):
    patient_sessions = session[session.PATIENT_ID == patient_id]
    patient_sessions.index = patient_sessions.PROTOCOL_ID
    return patient_sessions.USAGE

def substitute_protocol(row, protocol_sim, usage_df):
    if row["INTERCHANGE"]:
        return find_substitute(
            row["PATIENT_ID"],
            row["PROTOCOL_ID"], 
            protocol_sim, 
            usage_df
        )
    return row["PROTOCOL_ID"]

# ---------------------------------------------------------------------
# RETURNS
# -------
# patient_df
# session_df
# protocol_df
