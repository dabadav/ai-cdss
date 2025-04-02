import importlib.resources
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd

from ai_cdss import config
from ai_cdss.utils import MultiKeyDict

# ------------------------------
# Clinical Scores

class ClinicalSubscales:
    def __init__(self, scale_yaml_path: Optional[str] = None):
        """Initialize with an optional path to scale.yaml, defaulting to internal package resource."""
        # Retrieves max values for clinical subscales from config/scales.yaml
        if scale_yaml_path:
            self.scales_path = Path(scale_yaml_path)
        else:
            self.scales_path = importlib.resources.files(config) / "scales.yaml"
        if not self.scales_path.exists():
            raise FileNotFoundError(f"Scale YAML file not found at {self.scales_path}")

        # Load scales maximum values
        self.scales_dict = MultiKeyDict.from_yaml(self.scales_path)

    def compute_deficit_matrix(self, patient_df: pd.DataFrame) -> pd.DataFrame:
        """Compute deficit matrix given patient clinical scores."""

        # Retrieve max values using MultiKeyDict
        max_subscales = [self.scales_dict.get(scale, None) for scale in patient_df.columns]
        
        # Check for missing subscale values
        if None in max_subscales:
            missing_subscales = [scale for scale, max_val in zip(patient_df.columns, max_subscales) if max_val is None]
            raise ValueError(f"Missing max values for subscales: {missing_subscales}")
        
        # Compute deficit matrix
        deficit_matrix = 1 - (patient_df / pd.Series(max_subscales, index=patient_df.columns))
        return deficit_matrix

# ------------------------------
# Protocol Attributes

class ProtocolToClinicalMapper:
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

    def map_protocol_features(self, protocol_df: pd.DataFrame, agg_func=np.mean) -> pd.DataFrame:
        """Map protocol-level features into clinical scales using a predefined mapping."""
        # Retrieve max values using MultiKeyDict
        df_clinical = pd.DataFrame(index=protocol_df.index)
        # Collapse using agg_func the protocol latent attributes    
        for clinical_scale, features in self.mapping.items():
            df_clinical[clinical_scale] = protocol_df[features].apply(agg_func, axis=1)
        df_clinical.index = protocol_df["PROTOCOL_ID"]
        return df_clinical

# ---------------------------------------------------------------
# Data Utilities

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

    gower_sim_matrix = gower_sim_matrix.stack().reset_index()
    gower_sim_matrix.columns = ["PROTOCOL_A", "PROTOCOL_B", "SIMILARITY"]

    return gower_sim_matrix

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
