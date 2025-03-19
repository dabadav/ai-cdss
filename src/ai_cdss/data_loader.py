import pandas as pd
import pandera as pa
from pandera.errors import SchemaError
from pandera.typing import DataFrame
from ai_cdss.models import SessionSchema, TimeseriesSchema, PPFSchema, PCMSchema
from rgs_interface.data.interface import fetch_rgs_data, fetch_timeseries_data
import logging
from typing import List
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# RGS Data Loader

class DataLoader:
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
        self.rgs_mode = rgs_mode

    @pa.check_types
    def load_session_data(self, patient_list: List[int]) -> DataFrame[SessionSchema]:
        """
        Load session data from the RGS interface.

        Returns
        -------
        DataFrame[SessionSchema]
            Session data for the specified patients.
        """
        try:
            session = fetch_rgs_data(patient_list, rgs_mode=self.rgs_mode)
            logger.info("Session data loaded successfully.")
            return session
        except SchemaError as e:
            logger.error(f"Data validation failed: {e}")
            return pd.DataFrame(columns=SessionSchema.to_schema().columns.keys())
        except Exception as e:
            logger.error(f"Failed to load session data: {e}")
            raise

    @pa.check_types
    def load_timeseries_data(self, patient_list: List[int]) -> DataFrame[TimeseriesSchema]:
        """
        Load timeseries data from the RGS interface.

        Returns
        -------
        DataFrame[TimeseriesSchema]
            Timeseries data for the specified patients.
        """
        
        try:
            timeseries = fetch_timeseries_data(patient_list, rgs_mode=self.rgs_mode)
            return timeseries
        except SchemaError as e:
            logger.error(f"Data validation failed: {e}")
            return pd.DataFrame(columns=TimeseriesSchema.to_schema().columns.keys())
        except Exception as e:
            logger.error(f"Failed to load timeseries data: {e}")
            raise

    @pa.check_types
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
            output_dir = Path.home() / ".ai_cdss" / "output"
            
            # Check if file exists
            if (output_dir / "ppf.parquet").exists():
                ppf_data = pd.read_parquet(output_dir / "ppf.parquet").reset_index()
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
            
            logger.info("PPF data loaded successfully.")
            return ppf_data

        except Exception as e:
            logger.error(f"Failed to load PPF data: {e}")
            raise

    @pa.check_types
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
                similarity_data = pd.read_parquet(parquet_file).reset_index()
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