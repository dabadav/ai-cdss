import logging
from typing import Any, Callable, List, Optional, Union

import pandas as pd
from ai_cdss.constants import PATIENT_ID
from ai_cdss.models import (
    DataUnit,
    DataUnitName,
    Granularity,
    PPFSchema,
    SessionSchema,
    TimeseriesSchema,
)
from pandera.errors import SchemaError
from rgs_interface.data.interface import DatabaseInterface

from .base import DataLoaderBase
from .utils import (
    _decode_subscales,
    _load_ppf_data,
    _load_protocol_attributes,
    _load_protocol_similarity,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# RGS Data Loader


class DataLoader(DataLoaderBase):
    """
    Loads data from database and CSV files for the Clinical Decision Support System.

    Args:
        rgs_mode (str, optional): Mode for fetching RGS data. Default is "plus".
    """

    def __init__(self, rgs_mode: str = "plus"):
        """
        Initialize the DataLoader with a database interface and RGS mode.

        Args:
            rgs_mode (str, optional): Mode for fetching RGS data. Default is "plus".
        """
        self.interface: DatabaseInterface = DatabaseInterface()
        self.rgs_mode = rgs_mode

    def load_patient_data(
        self, patient_list: List[int]
    ) -> Union[pd.DataFrame, DataUnit]:
        """
        Load all data specific to a patient clinical trial.

        Args:
            patient_list (List[int]): List of patient IDs.

        Returns:
            Union[pd.DataFrame, DataUnit]: Patient data as a DataFrame or DataUnit.
        """
        return self._load_data(
            fetch_fn=self.interface.fetch_clinical_data,
            patient_list=patient_list,
            name=DataUnitName.PATIENT,
            granularity=Granularity.PATIENT_ID,
            wrap_in_dataunit=True,
        )

    def load_session_data(
        self, patient_list: List[int]
    ) -> Union[pd.DataFrame, DataUnit]:
        """
        Load session data from the RGS interface.

        Args:
            patient_list (List[int]): List of patient IDs.

        Returns:
            Union[pd.DataFrame, DataUnit]: Session data as a DataFrame or DataUnit.
        """
        return self._load_data(
            fetch_fn=lambda p: self.interface.fetch_rgs_data(p, rgs_mode=self.rgs_mode),
            patient_list=patient_list,
            name=DataUnitName.SESSIONS,
            granularity=Granularity.BY_PPS,
            schema_cls=SessionSchema,
            wrap_in_dataunit=True,
        )

    def load_timeseries_data(
        self, patient_list: List[int]
    ) -> Union[pd.DataFrame, DataUnit]:
        """
        Load timeseries data for the given patient list.

        Args:
            patient_list (List[int]): List of patient IDs.

        Returns:
            Union[pd.DataFrame, DataUnit]: Timeseries data as a DataFrame or DataUnit.
        """
        return self._load_data(
            fetch_fn=lambda p: self.interface.fetch_dm_data(p, rgs_mode=self.rgs_mode),
            patient_list=patient_list,
            schema_cls=TimeseriesSchema,
        )

    def load_ppf_data(self, patient_list: List[int]) -> Union[pd.DataFrame, DataUnit]:
        """
        Load PPF (precomputed patient-protocol fit) data from internal storage.

        Args:
            patient_list (List[int]): List of patient IDs.

        Returns:
            Union[pd.DataFrame, DataUnit]: PPF data as a DataFrame or DataUnit.
        """
        return self._load_data(
            fetch_fn=lambda p: _load_ppf_data(p),
            patient_list=patient_list,
            name=DataUnitName.PPF,
            granularity=Granularity.BY_PP,
            schema_cls=PPFSchema,
            wrap_in_dataunit=True,
        )

    def load_patient_subscales(
        self, patient_list: List[int]
    ) -> Union[pd.DataFrame, DataUnit]:
        """
        Load and decode patient clinical scale data for the given patient list.

        Args:
            patient_list (List[int]): List of patient IDs.

        Returns:
            Union[pd.DataFrame, DataUnit]: Decoded patient scale data indexed by PATIENT_ID.
        """
        patient_data = self._load_data(
            fetch_fn=self.interface.fetch_clinical_data,
            patient_list=patient_list,
            name=DataUnitName.PATIENT,
            granularity=Granularity.PATIENT_ID,
            wrap_in_dataunit=True,
        )
        decoded = patient_data.data.apply(_decode_subscales, axis=1)
        return decoded.set_index(PATIENT_ID)

    def load_protocol_attributes(self, file_path=None):
        """
        Load protocol attributes from internal storage or a specified file path.

        Args:
            file_path (str, optional): Path to the protocol attributes file.

        Returns:
            pd.DataFrame: Protocol attributes data.
        """
        return _load_protocol_attributes()

    def load_protocol_similarity(self):
        """
        Load protocol similarity data from internal storage.

        Returns:
            pd.DataFrame: Protocol similarity data with columns: PROTOCOL_ID_1, PROTOCOL_ID_2, SIMILARITY_SCORE.
        """
        try:
            data = _load_protocol_similarity()
            logger.debug("Protocol similarity data loaded successfully.")
            return data
        except Exception as e:
            logger.error("Failed to load protocol similarity data: %s", e)
            raise

    def _load_data(
        self,
        fetch_fn: Callable[[List[int]], Any],
        patient_list: List[int],
        name: Optional[DataUnitName] = None,
        granularity: Optional[Granularity] = None,
        schema_cls: Optional[Any] = None,
        wrap_in_dataunit: bool = False,
    ) -> Union[pd.DataFrame, DataUnit]:
        """
        Generalized data loading method with optional schema validation and DataUnit wrapping.

        Args:
            fetch_fn (Callable[[List[int]], Any]): Function to fetch data from the interface.
            patient_list (List[int]): List of patient IDs.
            name (Optional[DataUnitName]): Name of the DataUnit (if wrapping).
            granularity (Optional[Granularity]): Granularity level (if wrapping).
            schema_cls (Optional[Any]): Schema class for validation and fallback.
            wrap_in_dataunit (bool): Whether to return as DataUnit.

        Returns:
            Union[pd.DataFrame, DataUnit]: Loaded data as a DataFrame or DataUnit.
        """
        if wrap_in_dataunit:
            assert name is not None, "Name must not be None"
            assert granularity is not None, "Granularity must not be None"

        try:
            data = fetch_fn(patient_list)
            logger.debug("%s data loaded successfully.", name or fetch_fn.__name__)
            if wrap_in_dataunit:
                metadata = dict(data.attrs) if hasattr(data, "attrs") else {}
                return DataUnit(name, data, granularity, metadata, schema_cls)  # type: ignore[arg-type]
            return data

        except SchemaError as e:
            logger.error("Data validation failed: %s", e)
            if schema_cls:
                empty_df = pd.DataFrame(columns=schema_cls.to_schema().columns.keys())
                if wrap_in_dataunit:
                    return DataUnit(name, empty_df, granularity, {}, schema_cls)  # type: ignore[arg-type]
                else:
                    return empty_df
            raise

        except Exception as e:
            logger.error("Failed to load %s: %s", name or fetch_fn.__name__, e)
            raise RuntimeError(
                f"Failed to load {name or fetch_fn.__name__}: {e}"
            ) from e

    def fetch_and_validate_patients(self, study_ids: List[int]) -> List[int]:
        """
        Fetch patient data for the given study IDs and validate that patients exist.

        Args:
            study_ids (List[int]): List of study cohort identifiers to fetch patients for.

        Returns:
            List[int]: List of patient IDs associated with the provided study IDs.

        Raises:
            ValueError: If no patients are found for the given study IDs.
        """
        patient_data = self.interface.fetch_patients_by_study(study_ids=study_ids)
        if patient_data is None or patient_data.empty:
            # No exception: just log and return []
            logger.warning("No patients found for study IDs %s", study_ids)
            return []
        return patient_data[PATIENT_ID].tolist()
