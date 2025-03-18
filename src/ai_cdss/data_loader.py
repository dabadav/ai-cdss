import pandas as pd
import pandera as pa
from pandera.errors import SchemaError
from pandera.typing import DataFrame
from ai_cdss.models import SessionSchema, TimeseriesSchema, PPFSchema
from rgs_interface.data.interface import fetch_rgs_data, fetch_timeseries_data
import logging
from typing import List, Optional
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
    patient_list : list of int
        List of patient IDs to load data for.
    rgs_mode : str, optional
        Mode for fetching RGS data. Default is "plus".
    """

    def __init__(self, rgs_mode: str = "plus"):
        """
        Initialize the DataLoader with a list of patient IDs and RGS mode.
        """
        self.rgs_mode = rgs_mode

    def load_data(self, patient_list: List[int]) -> pd.DataFrame:
        """
        Load and merge session and timeseries data.

        Returns
        -------
        pd.DataFrame
            Merged data containing session and timeseries information.
        """
        session = self.load_session_data(patient_list)
        timeseries = self.load_timeseries_data(patient_list)

        # Merge dms using mean
        timeseries = (
            timeseries
            .groupby(["PATIENT_ID", "SESSION_ID", "PROTOCOL_ID", "GAME_MODE", "SECONDS_FROM_START"])
            .agg({
                "DM_KEY": lambda x: list(set(x)),  # Unique parameters at this time
                "DM_VALUE": "mean",               # Average parameter value
                "PE_KEY": "first",                # Assume same performance key per time, take first
                "PE_VALUE": "mean"                # Average performance value (usually only one)
            })
            .reset_index()
        )

        data = session.merge(timeseries, on=["PATIENT_ID", "SESSION_ID", "PROTOCOL_ID"])
        return data

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
        Load PPF (precomputed performance factors) from internal data.

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
