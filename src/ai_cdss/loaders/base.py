from abc import ABC, abstractmethod
from typing import List, Optional, Union

import pandas as pd
from ai_cdss.models import DataUnit

# ---------------------------------------------------------------------
# Base Data Loader


class DataLoaderBase(ABC):
    @abstractmethod
    def load_session_data(
        self, patient_list: List[int]
    ) -> Union[pd.DataFrame, DataUnit]:
        pass

    @abstractmethod
    def load_timeseries_data(
        self, patient_list: List[int]
    ) -> Union[pd.DataFrame, DataUnit]:
        pass

    @abstractmethod
    def load_ppf_data(self, patient_list: List[int]) -> Union[pd.DataFrame, DataUnit]:
        pass

    @abstractmethod
    def load_protocol_similarity(self) -> Union[pd.DataFrame, DataUnit]:
        pass

    @abstractmethod
    def load_patient_subscales(
        self, patient_list: List[int]
    ) -> Union[pd.DataFrame, DataUnit]:
        pass

    @abstractmethod
    def load_protocol_attributes(self, file_path: Optional[str] = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def fetch_and_validate_patients(
        self, study_ids: Optional[List[int]] = None
    ) -> List[int]:
        pass
