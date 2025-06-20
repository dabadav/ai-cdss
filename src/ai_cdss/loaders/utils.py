import json
import logging
import shutil
from pathlib import Path
from typing import List

import pandas as pd
from ai_cdss.constants import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, PPF_PARQUET_FILEPATH

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# - Utility Functions
# ---------------------------------------------------------------------


def _decode_subscales(row, subscales_column="CLINICAL_SCORES", id_column="PATIENT_ID"):
    # Use the last evaluation
    data = json.loads(row[subscales_column])[-1]
    # Keep only nested subscales (not metadata like 'evaluation_date')
    subscales = {k: v for k, v in data.items() if isinstance(v, dict)}
    # Flatten nested subscale dicts
    flat = pd.json_normalize(subscales).iloc[0]
    # Add patient ID
    flat[id_column] = row[id_column]
    return flat


def safe_load_csv(file_path: str = None, default_filename: str = None) -> pd.DataFrame:
    """
    Safely loads a CSV file, either from a given file path or from the default data directory.

    Parameters:
        file_path (str, optional): Full path to the CSV file. If not provided, `default_filename` is used.
        default_filename (str, optional): Name of the file in the default directory.

    Returns:
        pd.DataFrame: Loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be read as a valid CSV.
    """
    file_path = Path(file_path) if file_path else DEFAULT_DATA_DIR / default_filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}. Ensure the correct path is specified."
        )

    try:
        df = pd.read_csv(file_path, index_col=0)

        # If the file was loaded from outside the default directory, save a copy
        default_file_path = DEFAULT_DATA_DIR / file_path.name

        if file_path.parent != DEFAULT_DATA_DIR:
            shutil.copy(file_path, default_file_path)
            print(f"File copied to default directory: {default_file_path}")

        return df

    except Exception as e:
        raise ValueError(f"Error reading {file_path}: {e}")


def _load_patient_subscales(file_path: str = None) -> pd.DataFrame:
    """Load patient clinical subscale scores from a given file or the default directory."""
    return safe_load_csv(file_path, "clinical_scores.csv")


def _load_protocol_attributes(file_path: str = None) -> pd.DataFrame:
    """Load protocol attributes from a given file or the default directory."""
    return safe_load_csv(file_path, "protocol_attributes.csv")


def _load_protocol_similarity(file_path: str = None) -> pd.DataFrame:
    file_path = (
        Path(file_path) if file_path else DEFAULT_OUTPUT_DIR / "protocol_similarity.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            "No protocol similarity file found in ~/.ai_cdss/output. "
            "Expected either protocol_similarity.parquet or protocol_similarity.csv."
        )

    similarity_data = pd.read_csv(file_path, index_col=0)

    logger.debug("Protocol similarity data loaded successfully.")
    return similarity_data


def _load_ppf_data(patient_list: List[int]) -> pd.DataFrame:
    # Define PPF file path
    ppf_path = PPF_PARQUET_FILEPATH

    # Check if file exists
    if not ppf_path.exists():
        raise FileNotFoundError("No PPF file found in ~/.ai_cdss/output.")

    # Load parquet
    ppf_data = pd.read_parquet(path=ppf_path)

    # Filter PPF by patient list
    ppf_data = ppf_data[ppf_data["PATIENT_ID"].isin(patient_list)]

    # Check for missing patients
    missing_patients = set(patient_list) - set(ppf_data["PATIENT_ID"].unique())

    # If no PPF data for a patient
    if missing_patients:
        logger.warning(
            f"PPF missing for {len(missing_patients)} patients: {missing_patients}"
        )
        protocols = set(ppf_data["PROTOCOL_ID"].unique())

        # Generate new rows where each missing patient is assigned every protocol
        missing_combinations = pd.DataFrame(
            [
                {
                    "PATIENT_ID": pid,
                    "PROTOCOL_ID": protocol_id,
                    "PPF": None,
                    "CONTRIB": None,
                }  # Initialize to None
                for pid in missing_patients
                for protocol_id in protocols
            ]
        )

        # Concatenate missing patient data into the existing PPF dataset
        ppf_data = pd.concat([ppf_data, missing_combinations], ignore_index=True)
        ppf_data.attrs["missing_patients"] = list(missing_patients)
        return ppf_data

    return ppf_data
