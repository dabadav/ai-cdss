from typing import List
from abc import ABC, abstractmethod

import pandas as pd
from pandera.typing import DataFrame

from ai_cdss.models import SessionSchema, TimeseriesSchema, PPFSchema, PCMSchema

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
    def load_patient_subscales(self, patient_list: str = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def load_protocol_attributes(self, file_path: str = None) -> pd.DataFrame:
        pass
