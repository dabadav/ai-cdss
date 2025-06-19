from typing import List
import pandas as pd
from pandera.typing import DataFrame

from ai_cdss.models import SessionSchema, TimeseriesSchema, PPFSchema, PCMSchema, safe_check_types
from .base import DataLoaderBase
from .utils import _load_patient_subscales, _load_protocol_attributes

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
