from typing import List

import pandas as pd
from ai_cdss.evaluation.synthetic import (
    generate_synthetic_ids,
    generate_synthetic_ppf_data,
    generate_synthetic_protocol_metric,
    generate_synthetic_protocol_similarity,
    generate_synthetic_session_data,
    generate_synthetic_timeseries_data,
)
from ai_cdss.models import DataUnit

from .base import DataLoaderBase

# ---------------------------------------------------------------------
# Synthetic Data Loader


class DataLoaderMock(DataLoaderBase):

    def __init__(
        self, num_patients: int = 5, num_protocols: int = 3, num_sessions: int = 10
    ):
        # Generate and store shared IDs
        self.ids = generate_synthetic_ids(
            num_patients=num_patients,
            num_protocols=num_protocols,
            num_sessions=num_sessions,
        )
        self.num_protocols = num_protocols

    def load_session_data(self, patient_list: List[int] = None) -> DataUnit:
        if patient_list is None:
            patient_list = []
        return generate_synthetic_session_data(shared_ids=self.ids)

    def load_timeseries_data(self, patient_list: List[int] = None) -> DataUnit:
        if patient_list is None:
            patient_list = []
        return generate_synthetic_timeseries_data(shared_ids=self.ids)

    def load_ppf_data(self, patient_list: List[int] = None) -> DataUnit:
        if patient_list is None:
            patient_list = []
        return generate_synthetic_ppf_data(shared_ids=self.ids)

    def load_protocol_similarity(self) -> pd.DataFrame:
        return generate_synthetic_protocol_similarity(num_protocols=self.num_protocols)

    def load_protocol_init(self) -> pd.DataFrame:
        return generate_synthetic_protocol_metric(num_protocols=self.num_protocols)

    def load_patient_subscales(self, patient_list: List[int] = None):
        if patient_list is None:
            patient_list = []
        return super().load_patient_subscales(patient_list)

    def load_protocol_attributes(self, file_path=None):
        return super().load_protocol_attributes(file_path)
