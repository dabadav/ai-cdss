# ai_cdss/data_loader.py
from abc import ABC, abstractmethod
from typing import List
from pathlib import Path

import pandas as pd
import pandera as pa
from pandera.errors import SchemaError
from pandera.typing import DataFrame

from ai_cdss.models import SessionSchema, TimeseriesSchema, PPFSchema, PCMSchema, safe_check_types
from ai_cdss.evaluation.synthetic import (
    generate_synthetic_session_data,
    generate_synthetic_protocol_similarity,
    generate_synthetic_timeseries_data,
    generate_synthetic_ppf_data,
    generate_synthetic_ids,
    generate_synthetic_protocol_metric
)
from ai_cdss.constants import DEFAULT_DATA_DIR, PPF_PARQUET_FILEPATH
from rgs_interface.data.interface import DatabaseInterface

import shutil

import logging
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Base Data Loader

class DataLoaderBase(ABC):
    @abstractmethod
    def load_session_data(self, patient_list: List[int]) -> DataFrame[SessionSchema]:
        pass

    @abstractmethod
    def load_timeseries_data(self, patient_list: List[int]) -> DataFrame[TimeseriesSchema]:
        pass

    @abstractmethod
    def load_ppf_data(self, patient_list: List[int]) -> DataFrame[PPFSchema]:
        pass

    @abstractmethod
    def load_protocol_similarity(self) -> DataFrame[PCMSchema]:
        pass

    @abstractmethod
    def load_patient_clinical_data(self, patient_list: List[int]) -> pd.DataFrame:
        pass

    @abstractmethod
    def load_patient_subscales(self, patient_list: str = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def load_protocol_attributes(self, file_path: str = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def load_protocol_init(self) -> pd.DataFrame:
        pass

# ---------------------------------------------------------------------
# RGS Data Loader

class DataLoader(DataLoaderBase):
    """
    Loads data from database and CSV files.

    Parameters
    ----------
    rgs_mode : str, optional
        Mode for fetching RGS data. Default is "plus".
    """

    def __init__(self, rgs_mode: str = "plus"):
        """
        Initialize the DataLoader with a list of patient IDs and RGS mode.
        """
        self.interface: DatabaseInterface = DatabaseInterface()
        self.rgs_mode = rgs_mode

    # @safe_check_types(SessionSchema)
    def load_session_data(self, patient_list: List[int]) -> DataFrame[SessionSchema]:
        """
        Load session data from the RGS interface.
        New patients without prescriptions are not included in this table

        Returns
        -------
        DataFrame[SessionSchema]
            Session data for the specified patients.
        """
        try:
            session = self.interface.fetch_rgs_data(patient_list, rgs_mode=self.rgs_mode)
            logger.debug("Session data loaded successfully.")
            return session
        except SchemaError as e:
            logger.error(f"Data validation failed: {e}")
            return pd.DataFrame(columns=SessionSchema.to_schema().columns.keys())
        except Exception as e:
            logger.error(f"Failed to load session data: {e}")
            raise

    # @safe_check_types(TimeseriesSchema)
    def load_timeseries_data(self, patient_list: List[int]) -> DataFrame[TimeseriesSchema]:
        """
        Load timeseries data from the RGS interface.
        New patients without prescriptions are not included in this table

        Returns
        -------
        DataFrame[TimeseriesSchema]
            Timeseries data for the specified patients.
        """
        
        try:
            timeseries = self.interface.fetch_timeseries_data(patient_list, rgs_mode=self.rgs_mode)
            logger.debug(f"Timeseries data loaded successfully.")
            return timeseries
        except SchemaError as e:
            logger.error(f"Data validation failed: {e}")
            return pd.DataFrame(columns=TimeseriesSchema.to_schema().columns.keys())
        except Exception as e:
            logger.error(f"Failed to load timeseries data: {e}")
            raise

    # @safe_check_types(PPFSchema)
    def load_ppf_data(self, patient_list: List[int]) -> DataFrame[PPFSchema]:
        """
        Load PPF (precomputed patient-protocol fit) from internal data.

        Returns
        -------
        DataFrame[PPFSchema]
            PPF data indexed by PROTOCOL_ID.
        """
        try:
            # Define PPF file path
            ppf_path = PPF_PARQUET_FILEPATH

            # Check if file exists
            if ppf_path.exists():
                ppf_data = pd.read_parquet(path = ppf_path)
            else:
                raise FileNotFoundError("No PPF file found in ~/.ai_cdss/output.")

            # Filter PPF by patient list
            ppf_data = ppf_data[ppf_data["PATIENT_ID"].isin(patient_list)]

            # Check for missing patients
            missing_patients = set(patient_list) - set(ppf_data["PATIENT_ID"].unique())

            # If no PPF data for a patient
            if missing_patients:
                logger.warning(f"PPF missing for {len(missing_patients)} patients: {missing_patients}")
                protocols = set(ppf_data["PROTOCOL_ID"].unique())

                # Generate new rows where each missing patient is assigned every protocol
                missing_combinations = pd.DataFrame([
                    {"PATIENT_ID": pid, "PROTOCOL_ID": protocol_id, "PPF": None, "CONTRIB": None}  # Initialize to None
                    for pid in missing_patients
                    for protocol_id in protocols
                ])

                # Concatenate missing patient data into the existing PPF dataset
                ppf_data = pd.concat([ppf_data, missing_combinations], ignore_index=True)
                return ppf_data
            
            logger.debug("PPF data loaded successfully.")
            return ppf_data

        except Exception as e:
            logger.error(f"Failed to load PPF data: {e}")
            raise

    # @safe_check_types(PCMSchema)
    def load_protocol_similarity(self) -> DataFrame[PCMSchema]:
        """
        Load protocol similarity data from internal storage.

        Returns
        -------
        DataFrame[ProtocolSimilaritySchema]
            Protocol similarity data with columns: PROTOCOL_ID_1, PROTOCOL_ID_2, SIMILARITY_SCORE.
        """
        try:
            # Define similarity file paths
            output_dir = Path.home() / ".ai_cdss" / "output"
            parquet_file = output_dir / "protocol_similarity.parquet"
            csv_file = output_dir / "protocol_similarity.csv"
            
            # Check if Parquet file exists
            if parquet_file.exists():
                similarity_data = pd.read_parquet(path = parquet_file).reset_index()
            # Fall back to CSV if Parquet file is not found
            elif csv_file.exists():
                similarity_data = pd.read_csv(csv_file, index_col=0)
            else:
                raise FileNotFoundError(
                    "No protocol similarity file found in ~/.ai_cdss/output. "
                    "Expected either protocol_similarity.parquet or protocol_similarity.csv."
                )
            
            logger.info("Protocol similarity data loaded successfully.")
            return similarity_data

        except Exception as e:
            logger.error(f"Failed to load protocol similarity data: {e}")
            raise

    def load_patient_clinical_data(self, patient_list: List[int]) -> pd.DataFrame:
        """
        Load patient clinical data from the RGS interface.

        Parameters
        ----------
        patient_list : List[int]
            List of patient IDs to fetch clinical data for.

        Returns
        -------
        pd.DataFrame
            DataFrame containing clinical data for the specified patients.
        """
        try:
            clinical_data = self.interface.fetch_clinical_data(patient_list)
            if clinical_data.empty:
                logger.warning("No clinical data found for the specified patients.")
            else:
                logger.info(f"Clinical data loaded for {len(clinical_data)} patients.")
            return clinical_data

        except Exception as e:
            logger.error(f"Failed to load patient clinical data: {e}")
            raise

    def load_patient_subscales(self, patient_list = None):
        return _load_patient_subscales()
    
    def load_protocol_attributes(self, file_path = None):
        return _load_protocol_attributes()

    def load_protocol_init(self) -> pd.DataFrame:
        try:
            output_dir = Path.home() / ".ai_cdss" / "output"
            csv_file = output_dir / "protocol_metrics.csv"

            if csv_file.exists():
                protocol_metrics = pd.read_csv(csv_file, index_col=0)
            else:
                raise FileNotFoundError(
                    "No protocol metrics file found in ~/.ai_cdss/output. "
                    "Expected protocol_metrics.csv."
                )
            logger.info("Protocol initialization data loaded successfully.")
            return protocol_metrics

        except Exception as e:
            logger.error(f"Failed to load protocol metrics data: {e}")
            raise

# ---------------------------------------------------------------------
# Local Data Loader

class DataLoaderLocal(DataLoaderBase):

    def load_session_data(self, patient_list: List[int]) -> DataFrame[SessionSchema]:
        pass

    def load_timeseries_data(self, patient_list: List[int]) -> DataFrame[TimeseriesSchema]:
        pass

    def load_ppf_data(self, patient_list: List[int]) -> DataFrame[PPFSchema]:
        pass

    def load_protocol_similarity(self) -> DataFrame[PCMSchema]:
        pass

    def load_patient_clinical_data(self, patient_list: List[int]) -> pd.DataFrame:
        pass

    def load_patient_subscales(self, patient_list = None):
        return _load_patient_subscales(file_path=None)
    
    def load_protocol_attributes(self, file_path: str = None) -> pd.DataFrame:
        return _load_protocol_attributes(file_path=file_path)

    def load_protocol_init(self) -> pd.DataFrame:
        pass

# ---------------------------------------------------------------------
# Synthetic Data Loader

class DataLoaderMock(DataLoaderBase):

    def __init__(self, num_patients: int = 5, num_protocols: int = 3, num_sessions: int = 10):
        # Generate and store shared IDs
        self.ids = generate_synthetic_ids(
            num_patients=num_patients,
            num_protocols=num_protocols,
            num_sessions=num_sessions,
        )
        self.num_protocols = num_protocols

    @safe_check_types(SessionSchema)
    def load_session_data(self, patient_list: List[int] = []) -> DataFrame[SessionSchema]:
        return generate_synthetic_session_data(shared_ids=self.ids)

    @safe_check_types(TimeseriesSchema)
    def load_timeseries_data(self, patient_list: List[int] = []) -> DataFrame[TimeseriesSchema]:
        return generate_synthetic_timeseries_data(shared_ids=self.ids)

    @pa.check_types
    def load_ppf_data(self, patient_list: List[int] = []) -> DataFrame[PPFSchema]:
        return generate_synthetic_ppf_data(shared_ids=self.ids)
    
    @pa.check_types
    def load_protocol_similarity(self) -> DataFrame[PCMSchema]:
        return generate_synthetic_protocol_similarity(num_protocols=self.num_protocols)

    def load_protocol_init(self) -> pd.DataFrame:
        return generate_synthetic_protocol_metric(num_protocols=self.num_protocols)

    def load_patient_clinical_data(self, patient_list):
        return super().load_patient_clinical_data(patient_list)
    
    def load_patient_subscales(self, patient_list = None):
        return super().load_patient_subscales(patient_list)
    
    def load_protocol_attributes(self, file_path = None):
        return super().load_protocol_attributes(file_path)
    
# ---------------------------------------------------------------------
# - Utility Functions
# ---------------------------------------------------------------------

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
        raise FileNotFoundError(f"File not found: {file_path}. Ensure the correct path is specified.")

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
