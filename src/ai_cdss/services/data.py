from abc import ABC, abstractmethod

import pandas as pd
import pandera as pa
from pandera.typing import DataFrame

from ai_cdss.models import SessionSchema, PatientSchema, ProtocolMatrixSchema
from rgs_interface.data.interface import fetch_rgs_data, fetch_timeseries_data

# ---------------------------------------------------------------------
# PARAMS
# ------
# patient_list: List
#

#################################
# ------ RGS Data Loader ------ #
#################################

class BaseDataLoader(ABC):
    """Abstract class for data loading strategies."""

    @abstractmethod
    def load_session_data(self) -> DataFrame[SessionSchema]:
        pass

    @abstractmethod
    def load_patient_data(self) -> DataFrame[PatientSchema]:
        pass

    @abstractmethod
    def load_protocol_data(self) -> DataFrame[ProtocolMatrixSchema]:
        pass

    @abstractmethod
    def load_timeseries_data(self) -> pd.DataFrame:
        pass

class DataLoader(BaseDataLoader):
    """Loads data from database and CSV files."""

    def __init__(self, patient_list):
        self.patient_list = patient_list

    @pa.check_types
    def load_session_data(self) -> DataFrame[SessionSchema]:
        # TODO: rgs_mode param?
        data_app = fetch_rgs_data(self.patient_list, rgs_mode="app")
        return data_app
    
    def load_timeseries_data(self):
        # TODO: rgs_mode param?
        # dms_app = pd.read_csv("../../data/app_timeseries.csv")
        dms_app = fetch_timeseries_data(self.patient_list, rgs_mode="app")
        return dms_app

    def load_patient_data(self) -> DataFrame[PatientSchema]:
        """Loads and filters patient data by patient_list."""
        df = pd.read_csv("../../data/clinical_scores.csv", index_col=0)
        df = df[df.index.isin(self.patient_list)]

        return df
    
    def load_protocol_data(self) -> DataFrame[ProtocolMatrixSchema]:
        return pd.read_csv("../../data/protocol_attributes.csv", index_col=0)

