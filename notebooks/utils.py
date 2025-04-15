import os
import logging
from datetime import datetime

import pandas as pd

from ai_cdss.constants import BY_ID, BY_PPS

def filter_study_range(group):
    day_0 = group["SESSION_DATE"].min()
    study_range = group["SESSION_DATE"] <= day_0 + pd.Timedelta(days=43)
    return group[study_range]

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
    # Step 0: Setup paths
    log_dir = os.path.expanduser("~/.ai_cdss/logs/")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    log_file = os.path.join(log_dir, "data_check.log")

    # Setup logging (re-setup per function call to ensure correct log file)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)

    # Step 1: Patient registered but no data yet (no prescription)
    print("Patient registered but no data yet.")
    patients_no_data = session[session["PRESCRIPTION_ID"].isna()]
    if not patients_no_data.empty:
        export_file = os.path.join(log_dir, f"patients_no_data_{timestamp}.csv")
        patients_no_data[["PATIENT_ID", "PRESCRIPTION_ID", "SESSION_ID"]].to_csv(export_file, index=False)
        logger.warning(f"{len(patients_no_data)} patients found without prescription. Check exported file: {export_file}")
    else:
        logger.info("No patients without prescription found.")

    # Drop these rows
    session = session.drop(patients_no_data.index)

    # Step 2: Sessions in session table but not in recording table (no adherence)
    print("Sessions in session table but not in recording table")
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

    # Setup logger — clean, no extra clutter
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Step 1: Outer merge for discrepancy check
    discrepancy_check = left.merge(right, on=on, how="outer", indicator=True)

    left_only = discrepancy_check[discrepancy_check["_merge"] == "left_only"]
    right_only = discrepancy_check[discrepancy_check["_merge"] == "right_only"]

    # Step 2: Export and log discrepancies if found

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