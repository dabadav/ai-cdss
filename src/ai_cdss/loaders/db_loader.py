import json
import logging
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import pandas as pd
from ai_cdss.models import (
    DataUnit,
    DataUnitName,
    Granularity,
    PPFSchema,
    SessionSchema,
    TimeseriesSchema,
    safe_check_types,
)
from pandera.errors import SchemaError
from pandera.typing import DataFrame
from rgs_interface.data.interface import DatabaseInterface

from .base import DataLoaderBase
from .utils import (
    _decode_subscales,
    _load_patient_subscales,
    _load_ppf_data,
    _load_protocol_attributes,
    _load_protocol_similarity,
)

logger = logging.getLogger(__name__)

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

    def load_patient_data(self, patient_list: List[int]) -> DataUnit:
        """
        Include all data specific to patient clinical trial
        """
        return self._load_data(
            fetch_fn=self.interface.fetch_clinical_data,
            patient_list=patient_list,
            name=DataUnitName.PATIENT,
            granularity=Granularity.PATIENT_ID,
            wrap_in_dataunit=True,
        )

    def load_session_data(self, patient_list: List[int]) -> DataUnit:
        """
        Load session data from the RGS interface.
        """
        return self._load_data(
            fetch_fn=lambda p: self.interface.fetch_rgs_data(p, rgs_mode=self.rgs_mode),
            patient_list=patient_list,
            name=DataUnitName.SESSIONS,
            granularity=Granularity.BY_PPS,
            schema_cls=SessionSchema,
            wrap_in_dataunit=True,
        )

    def load_timeseries_data(self, patient_list: List[int]) -> pd.DataFrame:
        return self._load_data(
            fetch_fn=lambda p: self.interface.fetch_dm_data(p, rgs_mode=self.rgs_mode),
            patient_list=patient_list,
            schema_cls=TimeseriesSchema,
        )

    def load_ppf_data(self, patient_list: List[int]) -> DataUnit:
        """
        Load PPF (precomputed patient-protocol fit) from internal data.

        Returns
        -------
        DataFrame[PPFSchema]
            PPF data indexed by PROTOCOL_ID.
        """
        return self._load_data(
            fetch_fn=lambda p: _load_ppf_data(p),
            patient_list=patient_list,
            name=DataUnitName.PPF,
            granularity=Granularity.BY_PP,
            schema_cls=PPFSchema,
            wrap_in_dataunit=True,
        )

    def load_patient_scales(self, patient_list):
        # Load patient data (filtering by list if provided)
        patient_data = self._load_data(
            fetch_fn=self.interface.fetch_clinical_data,
            patient_list=patient_list,
            name=DataUnitName.PATIENT,
            granularity=Granularity.PATIENT_ID,
            wrap_in_dataunit=True,
        )
        # Decode and reindex
        decoded = patient_data.data.apply(_decode_subscales, axis=1)
        return decoded.set_index("PATIENT_ID")

    def load_patient_subscales(self, patient_list: Optional[List[int]] = None):
        return _load_patient_subscales()

    def load_protocol_attributes(self, file_path=None):
        return _load_protocol_attributes()

    def load_protocol_similarity(self):
        """
        Load protocol similarity data from internal storage.

        Returns
        -------
        DataFrame[ProtocolSimilaritySchema]
            Protocol similarity data with columns: PROTOCOL_ID_1, PROTOCOL_ID_2, SIMILARITY_SCORE.
        """
        try:
            data = _load_protocol_similarity()
            logger.debug("Protocol similarity data loaded successfully.")
            return data
        except Exception as e:
            logger.error(f"Failed to load protocol similarity data: {e}")
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
        Generalized data loading method with optional schema and DataUnit wrapping.

        Args:
            fetch_fn: The function to fetch data from the interface.
            patient_list: List of patient IDs.
            name: Optional name of the DataUnit (if wrapping).
            granularity: Optional granularity level (if wrapping).
            schema_cls: Optional schema class for error-based fallback.
            wrap_in_dataunit: Whether to return as DataUnit.

        Returns:
            DataFrame or DataUnit
        """
        if wrap_in_dataunit:
            assert name is not None, "Name must not be None"
            assert granularity is not None, "Granularity must not be None"

        try:
            data = fetch_fn(patient_list)
            logger.debug(f"{name or fetch_fn.__name__} data loaded successfully.")
            if wrap_in_dataunit:
                metadata = dict(data.attrs) if hasattr(data, "attrs") else {}
                return DataUnit(name, data, granularity, metadata)
            return data

        except SchemaError as e:
            logger.error(f"Data validation failed: {e}")
            if schema_cls:
                empty_df = pd.DataFrame(columns=schema_cls.to_schema().columns.keys())
                return (
                    DataUnit(name, empty_df, granularity)
                    if wrap_in_dataunit
                    else empty_df
                )
            raise

        except Exception as e:
            logger.error(f"Failed to load {name or fetch_fn.__name__}: {e}")
            raise Exception(f"Failed to load {name or fetch_fn.__name__}: {e}")
