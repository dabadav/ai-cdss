# ai_cdss/data_processor.py
from typing import List
from functools import reduce

import pandas as pd
import pandera as pa
from pandera.typing import DataFrame

import os
import logging
from datetime import datetime

import importlib.resources
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import TheilSenRegressor
from scipy.signal import savgol_filter

from ai_cdss import config
from ai_cdss.utils import MultiKeyDict
from ai_cdss.constants import *
from ai_cdss.models import ScoringSchema, PPFSchema, SessionSchema, TimeseriesSchema

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#################################
# ------ Data Processing ------ #
#################################

# ---------------------------------------------------------------------
# Data Processor Class

class DataProcessor:
    """
    A class for processing data and computing a final weighted score.

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
    """
    def __init__(
        self,
        weights: List[float] = [1,1,1], 
        alpha: float = 0.5
    ):
        """
        Initialize the data processor with optional weights for scoring.
        """
        self.weights = weights
        self.alpha = alpha

    # @pa.check_types
    def process_data(
        self, 
        session_data: DataFrame[SessionSchema], 
        timeseries_data: DataFrame[TimeseriesSchema], 
        ppf_data: DataFrame[PPFSchema], 
        init_data: pd.DataFrame
    ) -> DataFrame[ScoringSchema]:
        """
        Process and score patient-protocol combinations using session, timeseries, and PPF data.
        When bootstrapping patient, score is based on PPF only.
        Otherwise a weighted sum of DELTA_DM, ADHERENCE_RECENT and PPF determines the score.

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

        # --- Log-Level Output ---
        ## PATIENT_ID + PROTOCOL_ID + SESSION_ID + ADHERENCE + RECENT_ADHERENCE + USAGE + DAYS + DM_VALUE + DELTA_DM_VALUE + PE_VALUE + PPF + SCORE
        
        # --- Feature Building ---
        if not session_data.empty and not timeseries_data.empty:
            # Compute Session Features
            ts_feat_df = self.build_delta_dm(timeseries_data)           # DELTA_DM
            adherence_df = self.build_recent_adherence(session_data)    # ADHERENCE_RECENT
            usage_df = self.build_usage(session_data)                   # USAGE
            days_df = self.build_prescription_days(session_data)        # DAYS
            
            # Combine Session Features
            feat_df = reduce(lambda l, r: safe_merge(l, r, on=BY_PPS), [ts_feat_df, adherence_df, usage_df, days_df])
            
            # Store the whole scored DataFrame as a csv
            feat_df.to_csv(DEFAULT_LOG_DIR / "scored_features.csv", index=False)   
            
            # Aggregate to patient protocol level
            feat_agg = feat_df.groupby(BY_PP).agg("last").reset_index()
        
        # If we are bootstrapping a study, and no patient prescriptions yet
        else:
            # Initialize feature df with expected columns
            scoring_columns = BY_PP + [DELTA_DM, RECENT_ADHERENCE, USAGE, DAYS]
            feat_agg = pd.DataFrame(columns=scoring_columns)

        # --- Merge All for Scoring ---
        scoring_input = reduce(lambda l, r: safe_merge(l, r, on=BY_PP), [
            ppf_data,
            feat_agg
        ])

        # --- Scoring ---
        scored_df = self.compute_score(scoring_input, init_data)
        scored_df.attrs = ppf_data.attrs
        
        return scored_df[BY_PP + FINAL_METRICS]

    def compute_score(self, data, protocol_metrics):
        """
        Initializes metrics based on legacy data and computes score

        Parameters
        ----------
        data : pd.DataFrame
            Aggregated session and timeseries data per patient-protocol.

        Returns
        -------
        pd.DataFrame
            Scored DataFrame sorted by patient and protocol.
        """
        # Initialize missing values
        data = self._init_metrics(data, protocol_metrics)
        
        # Compute objective function score alpha*Adherence + beta*DM + gamma*PPF
        score = self._compute_score(data)
        
        # Sort the output dataframe
        score.sort_values(by=BY_PP, inplace=True)

        return score

    def _init_metrics(self, data: pd.DataFrame, protocol_metrics: pd.DataFrame) -> pd.DataFrame:
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
        # When no prescriptions add empty list instead of NaN
        data[DAYS] = data[DAYS].apply(lambda x: [] if x is None or (not isinstance(x, list) and pd.isna(x)) else x)
        # When no sessions performed add a 0
        data[USAGE] = data[USAGE].astype("Int64").fillna(0)
        # Cast NaN to None 
        # data.where(pd.notna(data), None)

        # data = data.fillna(None)
        # data = data.fillna(0).infer_objects(copy=False)
        # # Only fill NaNs using map for matching PROTOCOL_IDs
        # data[ADHERENCE] = data[ADHERENCE].fillna(0)
        # # data[ADHERENCE] = data[ADHERENCE].fillna(data[PROTOCOL_ID].map(protocol_metrics[ADHERENCE]))
        # data[DM_VALUE] = data[DM_VALUE].fillna(data[PROTOCOL_ID].map(protocol_metrics["DM_DELTA"]))
        # # data[DM_VALUE] = data[DM_VALUE].fillna(data[PROTOCOL_ID].map(protocol_metrics["DM_DELTA"]))
        # data[PE_VALUE] = data[PE_VALUE].fillna(0)
        # data[DAYS]  = data[DAYS].fillna(0)
        return data
    
    def _compute_ewma(self, df, value_col, group_cols, sufix=""):
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
            **{f"{value_col}{sufix}": df.groupby(by=group_cols)[value_col].transform(lambda x: x.ewm(alpha=self.alpha, adjust=True).mean())}
        )

    def _compute_score(self, scoring: pd.DataFrame) -> pd.DataFrame:
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
        # weighted nan-compativle sum of factors
        scoring_df[SCORE] = (
            scoring_df[RECENT_ADHERENCE].astype("float64").fillna(0.) * self.weights[0]
            + scoring_df[DELTA_DM].astype("float64").fillna(0.) * self.weights[1]
            + scoring_df[PPF].astype("float64").fillna(0.) * self.weights[2]
        )
        return scoring_df

    def build_delta_dm(self, ts_df: DataFrame[TimeseriesSchema]) -> pd.DataFrame:
        grouped = ts_df.groupby(BY_PPS).agg({DM_VALUE: "mean", PE_VALUE: "mean"}).reset_index()

        grouped['SESSION_INDEX'] = grouped.groupby(BY_PP).cumcount() + 1
        grouped['DM_SMOOTH'] = grouped.groupby(BY_PP)[DM_VALUE].transform(
            apply_savgol_filter_groupwise, SAVGOL_WINDOW_SIZE, SAVGOL_POLY_ORDER
        )
        grouped[DELTA_DM] = grouped.groupby(BY_PP, group_keys=False)[['DM_SMOOTH', 'SESSION_INDEX']].apply(
            lambda g: get_rolling_theilsen_slope(g['DM_SMOOTH'], g['SESSION_INDEX'], THEILSON_REGRESSION_WINDOW_SIZE)
        ).fillna(0)

        return grouped[BY_PPS + [DM_VALUE, DELTA_DM]]

    def build_recent_adherence(self, session_df: DataFrame[SessionSchema]) -> pd.DataFrame:
        df = session_df.copy()
        df = self._compute_ewma(df, ADHERENCE, BY_PP, sufix="_RECENT")
        return df[BY_PPS + [RECENT_ADHERENCE]]

    def build_usage(self, session_df: DataFrame[SessionSchema]) -> pd.DataFrame:
        df = session_df.copy()
        df[USAGE] = df.groupby(BY_PP)[SESSION_ID].transform("nunique").astype("Int64")
        return df[BY_PPS + [USAGE]]

    def build_prescription_days(self, session_df: DataFrame[SessionSchema]) -> pd.DataFrame:
        prescribed_days = (
            session_df[session_df[PRESCRIPTION_ENDING_DATE] == PRESCRIPTION_ACTIVE]
            .groupby(BY_PP)[WEEKDAY_INDEX]
            .agg(lambda x: sorted(x.unique()))
            .rename(DAYS)
            .reset_index()
        )
        merged = safe_merge(session_df[BY_PPS], prescribed_days, on=BY_PP, how="left")
        return merged[BY_PPS + [DAYS]]
    
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
# Adherence
# ---------------------------------------------------------------

def expand_session(session: pd.DataFrame):
    """
    For each prescription, generate expected sessions and merge with actual performed sessions.
    Missing sessions are marked as NOT_PERFORMED with appropriate filling.
    """
    # Ensure dates are datetime
    session["SESSION_DATE"] = pd.to_datetime(session["SESSION_DATE"])
    last_session_per_patient = session.groupby("PATIENT_ID")["SESSION_DATE"].max().to_dict()
    
    # Step 1: Generate expected sessions DataFrame
    prescriptions = session.drop_duplicates(subset=["PRESCRIPTION_ID", "PATIENT_ID", "PROTOCOL_ID", "PRESCRIPTION_STARTING_DATE", "PRESCRIPTION_ENDING_DATE", "WEEKDAY_INDEX"])
    expected_rows = []

    for _, row in prescriptions.iterrows():
        start = row["PRESCRIPTION_STARTING_DATE"]
        end = row["PRESCRIPTION_ENDING_DATE"]
        weekday = row["WEEKDAY_INDEX"]

        if pd.isna(start) or pd.isna(end) or pd.isna(weekday):
            continue

        # Cap end date at last actual session of the patient
        last_patient_session = last_session_per_patient.get(row["PATIENT_ID"], pd.Timestamp.today())
        if end > last_patient_session:
            end = last_patient_session

        expected_dates = generate_expected_sessions(start, end, int(weekday))
    
        for date in expected_dates:
            template_row = row.copy()
            template_row["SESSION_DATE"] = date
            template_row["STATUS"] = "NOT_PERFORMED"
            template_row["ADHERENCE"] = 0.0
            template_row["SESSION_DURATION"] = 0
            template_row["REAL_SESSION_DURATION"] = 0
            # Set session outcome-related columns to NaN
            for col in METRIC_COLUMNS:
                template_row[col] = np.nan

            # Append to expected rows
            expected_rows.append(template_row.to_dict())

    expected_df = pd.DataFrame(expected_rows)

    if expected_df.empty:
        return session  # No new data, return original

    # Filter expected_df to remove performed sessions
    performed_index = pd.MultiIndex.from_frame(session[["PRESCRIPTION_ID", "SESSION_DATE"]])
    expected_index = pd.MultiIndex.from_frame(expected_df[["PRESCRIPTION_ID", "SESSION_DATE"]])  
    mask = ~expected_index.isin(performed_index)
    expected_df = expected_df.loc[mask]

    # Merge expected sessions with performed sessions
    merged = pd.concat([session, expected_df], ignore_index=True)
    
    # Propagate last performed session values (forward fill by prescription)
    merged.sort_values(by=["PRESCRIPTION_ID", "SESSION_DATE"], inplace=True)
    fill_columns = METRIC_COLUMNS
    merged[fill_columns] = merged.groupby("PRESCRIPTION_ID")[fill_columns].ffill()
    merged.fillna({'DM_VALUE': 0, 'PE_VALUE': 0}, inplace=True)

    # For missing session-specific outcome columns, set to NaN
    is_missing = merged["STATUS"] == "NOT_PERFORMED"
    for col in SESSION_COLUMNS:
        merged.loc[is_missing, col] = np.nan

    return merged.reset_index(drop=True)

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

# ---------------------------------------------------------------
# Delta DM
# ---------------------------------------------------------------

def apply_savgol_filter_groupwise(series, window_size, polyorder):
    series_len = len(series)
    if series_len < polyorder + 1: return series
    window = min(window_size, series_len)
    if window <= polyorder: window = polyorder + 1
    if window > series_len: return series
    if window % 2 == 0: window -= 1
    if window <= polyorder : return series
    try:
        return savgol_filter(series, window_length=window, polyorder=polyorder)
    except ValueError:
        return series

def get_rolling_theilsen_slope(series_y, series_x, window_size):
    slopes = pd.Series([np.nan] * len(series_y), index=series_y.index)
    if len(series_y) < 2 : return slopes
    regressor = TheilSenRegressor(random_state=42, max_subpopulation=1000)
    for i in range(len(series_y)):
        start_index = max(0, i - window_size + 1)
        window_y = series_y.iloc[start_index : i + 1]
        window_x = series_x.iloc[start_index : i + 1]
        if len(window_y) < 2:
            slopes.iloc[i] = 0.0 if len(window_y) == 1 else np.nan; continue
        if len(window_x.unique()) == 1 and len(window_y.unique()) > 1:
            slopes.iloc[i] = np.nan; continue
        if len(window_y.unique()) == 1:
             slopes.iloc[i] = 0.0; continue
        X_reshaped = window_x.values.reshape(-1, 1)
        try:
            regressor.fit(X_reshaped, window_y.values)
            slopes.iloc[i] = regressor.coef_[0]
        except Exception: slopes.iloc[i] = np.nan
    return slopes

# ---------------------------------------------------------------
# PPF
# ---------------------------------------------------------------

def feature_contributions(df_A, df_B):
    # Convert to numpy   
    A = df_A.to_numpy() # (patients, subscales)
    B = df_B.to_numpy() # (protocols, subscales)

    # Compute row-wise norms
    A_norms = np.linalg.norm(A, axis=1, keepdims=True) # (patients, 1)
    B_norms = np.linalg.norm(B, axis=1, keepdims=True) # (protocols, 1)
    
    # Replace zero norms with a small value to avoid NaN (division by zero)
    A_norms[A_norms == 0] = 1e-10
    B_norms[B_norms == 0] = 1e-10

    # Normalize each row to unit vectors
    A_norm = A / A_norms # (patient, subscales)
    B_norm = B / B_norms # (protocol, subscales)

    # Compute feature contributions
    contributions = A_norm[:, np.newaxis, :] * B_norm[np.newaxis, :, :] # (patient, dim, subscales) * (dim, protocol, subscales)

    return contributions # (patients, protocols, subscales_sim)

def compute_ppf(patient_deficiency, protocol_mapped):
    """ Compute the patient-protocol feature matrix (PPF) and feature contributions.
    """
    contributions = feature_contributions(patient_deficiency, protocol_mapped)
    ppf = np.sum(contributions, axis=2) # (patients, protocols, cosine)
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

# ---------------------------------------------------------------
# Utils
# ---------------------------------------------------------------

def check_prescriptions(session: pd.DataFrame):
    patients_non_prescribed = session[session[PRESCRIPTION_ID].isna()]
    if not patients_non_prescribed.empty:
        return patients_non_prescribed

def check_session(session: pd.DataFrame) -> pd.DataFrame:
    """
    Check for data discrepancies in session DataFrame, export findings to ~/.ai_cdss/logs/,
    log summary, and return cleaned DataFrame.

    Parameters
    ----------
    session : pd.DataFrame
        Session DataFrame to check and clean.

    Returns
    -------
    pd.DataFrame
        Cleaned session DataFrame.
    """
    # Patient registered but no data yet (no prescription)
    patients_no_data = session[session["PRESCRIPTION_ID"].isna()]
    if not patients_no_data.empty:
        patients_no_data[["PATIENT_ID", "PRESCRIPTION_ID", "SESSION_ID"]].to_csv(export_file, index=False)
        logger.warning(f"{len(patients_no_data)} patients found without prescription. Check exported file: {export_file}")
    else:
        logger.info("No patients without prescription found.")

    # Drop these rows
    session = session.drop(patients_no_data.index)

    # Sessions in session table but not in recording table (no adherence)
    patient_session_discrepancy = session[session["ADHERENCE"].isna()]
    if not patient_session_discrepancy.empty:
        export_file = os.path.join(log_dir, f"patient_session_discrepancy_{timestamp}.csv")
        patient_session_discrepancy[["PATIENT_ID", "PRESCRIPTION_ID", "SESSION_ID"]].to_csv(export_file, index=False)
        logger.warning(f"{len(patient_session_discrepancy)} sessions found without adherence. Check exported file: {export_file}")
    else:
        logger.info("No sessions without adherence found.")

    # Drop these rows
    session = session.drop(patient_session_discrepancy.index)

    # Final info
    logger.info(f"Session data cleaned. Final shape: {session.shape}")

    return session

def safe_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on,
    how: str = "left",
    export_dir: str = "~/.ai_cdss/logs/",
    left_name: str = "left",
    right_name: str = "right",
) -> pd.DataFrame:
    """
    Perform a safe merge and independently report unmatched rows from left and right DataFrames.

    Parameters
    ----------
    left : pd.DataFrame
        Left DataFrame.
    right : pd.DataFrame
        Right DataFrame.
    on : str or list
        Column(s) to join on.
    how : str, optional
        Type of merge to be performed. Default is "left".
    export_dir : str, optional
        Directory to export discrepancy reports and logs.
    left_name : str, optional
        Friendly name for the left DataFrame, for logging.
    right_name : str, optional
        Friendly name for the right DataFrame, for logging.
    drop_inconsistencies : bool, optional
        If True, drop inconsistent rows (left-only). Default is False.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame.
    """
    # Prepare export directory
    export_dir = os.path.expanduser(export_dir)
    os.makedirs(export_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(export_dir, "data_check.log")

    # Setup logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Outer merge for discrepancy check
    discrepancy_check = left.merge(right, on=on, how="outer", indicator=True)

    left_only = discrepancy_check[discrepancy_check["_merge"] == "left_only"]
    right_only = discrepancy_check[discrepancy_check["_merge"] == "right_only"]

    # Export and log discrepancies if found
    if not left_only.empty:
        export_file = os.path.join(export_dir, f"{left_name}_only_{timestamp}.csv")
        try:
            left_only[BY_ID + ["SESSION_DURATION", "SCORE", "DM_VALUE", "PE_VALUE"]].to_csv(export_file, index=False)
        except KeyError as e:
            left_only.to_csv(export_file, index=False)
            
        logger.warning(
            f"{len(left_only)} rows found only in '{left_name}' DataFrame "
            f"(see export: {export_file})"
        )

    if not right_only.empty:
        export_file = os.path.join(export_dir, f"{right_name}_only_{timestamp}.csv")
        try:
            right_only[BY_PPS + ["SESSION_DURATION", "SCORE", "DM_VALUE", "PE_VALUE"]].to_csv(export_file, index=False)
        except KeyError as e:
            right_only.to_csv(export_file, index=False)
        logger.warning(
            f"{len(right_only)} rows from '{right_name}' DataFrame did not match '{left_name}' DataFrame "
            f"(see export: {export_file})"
        )

    # Step 3: Actual requested merge
    merged = left.merge(right, on=on, how=how)

    return merged
