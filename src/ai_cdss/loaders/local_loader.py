from typing import List, Optional, Union

import pandas as pd
from ai_cdss.models import DataUnit, DataUnitName, Granularity, PPFSchema, SessionSchema

from .base import DataLoaderBase
from .utils import _load_protocol_attributes

# ---------------------------------------------------------------------
# Local Data Loader


class DataLoaderLocal(DataLoaderBase):
    """Local data loader for patient and protocol data."""

    def __init__(
        self,
        session_file: str,
        ppf_file: str,
        protocol_similarity_file: str,
        patient_subscales_file: str,
    ):
        self.session_file = session_file
        self.ppf_file = ppf_file
        self.protocol_similarity_file = protocol_similarity_file
        self.patient_subscales_file = patient_subscales_file

    def load_session_data(self, patient_list: List[int]) -> DataUnit:
        df = pd.read_csv(self.session_file)
        if patient_list:
            df = df[df["PATIENT_ID"].isin(patient_list)]
        return DataUnit(
            name=DataUnitName.SESSIONS,
            data=df,
            level=Granularity.BY_PPS,
            schema=SessionSchema,
        )

    def load_timeseries_data(self, patient_list: List[int]):
        """Stub for timeseries data loading (not implemented for local loader)."""
        raise NotImplementedError(
            "Timeseries data loading is not implemented for local loader."
        )

    def load_ppf_data(self, patient_list: List[int]) -> DataUnit:
        df = pd.read_csv(self.ppf_file)
        if patient_list:
            df = df[df["PATIENT_ID"].isin(patient_list)]
        return DataUnit(
            name=DataUnitName.PPF,
            data=df,
            level=Granularity.BY_PP,
            schema=PPFSchema,
        )

    def load_protocol_similarity(self):
        df = pd.read_csv(self.protocol_similarity_file)
        return df

    def load_patient_subscales(
        self, patient_list: Optional[List[int]] = None
    ) -> DataUnit:
        df = pd.read_csv(self.patient_subscales_file)
        if patient_list:
            df = df[df["PATIENT_ID"].isin(patient_list)]
        return DataUnit(
            name=DataUnitName.PATIENT,
            data=df,
            level=Granularity.PATIENT_ID,
        )

    def load_protocol_attributes(self, file_path: Optional[str] = None) -> pd.DataFrame:
        return _load_protocol_attributes(file_path=file_path)

    def fetch_and_validate_patients(
        self, study_ids: Optional[List[int]] = None
    ) -> List[int]:
        df = pd.read_csv(self.patient_subscales_file)
        if study_ids is not None:
            df = df[df["STUDY_ID"].isin(study_ids)]
        patient_ids = df["PATIENT_ID"].unique().tolist()
        if not patient_ids:
            raise ValueError("No patients found in the local patient subscales file.")
        return patient_ids
