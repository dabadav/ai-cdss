"""
Utility functions for loading and processing clinical and protocol data.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from ai_cdss.constants import (
    CLINICAL_SCORES,
    CLINICAL_SCORES_CSV,
    CONTRIB,
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    PATIENT_ID,
    PPF,
    PPF_PARQUET_FILEPATH,
    PROTOCOL_ATTRIBUTES_CSV,
    PROTOCOL_ID,
    PROTOCOL_SIMILARITY_CSV,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# - Utility Functions
# ---------------------------------------------------------------------


def decode_subscales(
    row: pd.Series,
    subscales_column: str = CLINICAL_SCORES,
    id_column: str = PATIENT_ID,
) -> pd.Series:
    """
    Decode and flatten the last clinical subscale evaluation for a patient.
    """
    data = json.loads(row[subscales_column])[-1]
    # Keep only nested subscales (not metadata like 'evaluation_date')
    subscales = {k: v for k, v in data.items() if isinstance(v, dict)}
    # Flatten nested subscale dicts
    flat = pd.json_normalize(subscales).iloc[0]
    # Add patient ID
    flat[id_column] = row[id_column]
    return flat


def safe_load_csv(
    file_path: Optional[Union[str, Path]] = None, default_filename: Optional[str] = None
) -> pd.DataFrame:
    """
    Safely loads a CSV file, either from a given file path or from the default data directory.
    Args:
        file_path: Full path to the CSV file (str or Path). If not provided, `default_filename` is used from the default directory.
        default_filename: Name of the file in the default directory.
    Returns:
        pd.DataFrame: Loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be read as a valid CSV.
    """
    if file_path is not None:
        file_path = Path(file_path)
    else:
        if default_filename is None:
            raise ValueError("Either file_path or default_filename must be provided.")
        file_path = DEFAULT_DATA_DIR / default_filename

    if not file_path.exists():
        raise FileNotFoundError(
            "File not found: %s. Ensure the correct path is specified." % file_path
        )

    try:
        df = pd.read_csv(file_path, index_col=0)

        # If the file was loaded from outside the default directory, save a copy
        default_file_path = DEFAULT_DATA_DIR / file_path.name

        if file_path.parent != DEFAULT_DATA_DIR:
            DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
            if default_file_path.exists():
                logger.warning(
                    "Overwriting existing file in default directory: %s",
                    default_file_path,
                )
            shutil.copy(file_path, default_file_path)
            logger.info("File copied to default directory: %s", default_file_path)
        return df
    except Exception as e:
        raise ValueError("Error reading %s: %s" % (file_path, e)) from e


def load_patient_subscales(
    file_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Load patient clinical subscale scores from a given file or the default directory.
    Args:
        file_path: Path or str to the file, or None for default.
    Returns:
        pd.DataFrame
    """
    return safe_load_csv(file_path, CLINICAL_SCORES_CSV)


def load_protocol_attributes(
    file_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Load protocol attributes from a given file or the default directory.
    Args:
        file_path: Path or str to the file, or None for default.
    Returns:
        pd.DataFrame
    """
    return safe_load_csv(file_path, PROTOCOL_ATTRIBUTES_CSV)


def load_protocol_similarity(
    file_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Load protocol similarity data from a given file or the default output directory.
    Args:
        file_path: Path or str to the file, or None for default.
    Returns:
        pd.DataFrame
    """
    if file_path is not None:
        file_path = Path(file_path)
    else:
        file_path = DEFAULT_OUTPUT_DIR / PROTOCOL_SIMILARITY_CSV

    if not file_path.exists():
        raise FileNotFoundError(
            "No protocol similarity file found in ~/.ai_cdss/output. "
            "Expected protocol_similarity.csv."
        )
    similarity_data = pd.read_csv(file_path, index_col=0)
    logger.debug("Protocol similarity data loaded successfully.")
    return similarity_data


def load_ppf_data(patient_list: List[int]) -> pd.DataFrame:
    """
    Load patient-protocol fit (PPF) data for a list of patients. If PPF is missing for any patient, add placeholder rows.
    Args:
        patient_list: List of patient IDs.
    Returns:
        pd.DataFrame
    """
    ppf_path = PPF_PARQUET_FILEPATH
    if not ppf_path.exists():
        raise FileNotFoundError("No PPF file found in ~/.ai_cdss/output.")
    ppf_data = pd.read_parquet(path=ppf_path)
    ppf_data = ppf_data[ppf_data[PATIENT_ID].isin(patient_list)]
    missing_patients = set(patient_list) - set(ppf_data[PATIENT_ID].unique())
    if missing_patients:
        logger.warning(
            "PPF missing for %d patients: %s", len(missing_patients), missing_patients
        )
        protocols = set(ppf_data[PROTOCOL_ID].unique())
        if not protocols:
            raise ValueError(
                "No protocols found in PPF data to assign to missing patients."
            )
        missing_combinations = pd.DataFrame(
            [
                {
                    PATIENT_ID: pid,
                    PROTOCOL_ID: protocol_id,
                    PPF: None,
                    CONTRIB: None,
                }
                for pid in missing_patients
                for protocol_id in protocols
            ]
        )
        ppf_data = pd.concat([ppf_data, missing_combinations], ignore_index=True)
        ppf_data.attrs["missing_patients"] = list(missing_patients)
    return ppf_data
