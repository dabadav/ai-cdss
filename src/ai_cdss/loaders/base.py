from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

import pandas as pd
from ai_cdss.models import (
    DataUnit,
    PCMSchema,
    PPFSchema,
    SessionSchema,
    TimeseriesSchema,
)
from pandera.typing import DataFrame

# ---------------------------------------------------------------------
# Base Data Loader


class DataLoaderBase(ABC):
    @abstractmethod
    def load_session_data(self, patient_list: List[int]) -> Union[DataFrame, DataUnit]:
        pass

    @abstractmethod
    def load_timeseries_data(
        self, patient_list: List[int]
    ) -> Union[DataFrame, DataUnit]:
        pass

    @abstractmethod
    def load_ppf_data(self, patient_list: List[int]) -> Union[DataFrame, DataUnit]:
        pass

    @abstractmethod
    def load_protocol_similarity(self) -> Union[DataFrame, DataUnit]:
        pass

    def load_patient_subscales(
        self, patient_list: Optional[Any] = None
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def load_protocol_attributes(self, file_path: Optional[str] = None) -> pd.DataFrame:
        pass
