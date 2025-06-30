"""
Basic usage of CDSS to generate a protocol recommendation.
==========================================================

This example demonstrates how to use the `DataLoader`, `DataProcessor`, and `CDSS`
to load data and generate a 7-day protocol plan for testing/CI purposes.
"""

# %%
import sys

sys.path.append("..")
import pandas as pd
from ai_cdss.cdss import CDSS
from ai_cdss.constants import EWMA_ALPHA
from ai_cdss.loaders.mock_loader import DataLoaderMock
from ai_cdss.models import DataUnit, DataUnitName, DataUnitSet, Granularity
from ai_cdss.processing.processor import DataProcessor
from IPython.display import display

print(__doc__)

PATIENT_LIST = [1, 2, 3]

# Parameters
rgs_mode = "app"
weights = [1, 1, 1]
alpha = EWMA_ALPHA

n = 12
days = 7
protocols_per_day = 5

# Services
loader = DataLoaderMock(num_patients=5, num_protocols=5, num_sessions=10)
processor = DataProcessor()

# Execution
session = loader.load_session_data(patient_list=PATIENT_LIST)
timeseries = loader.load_timeseries_data(patient_list=PATIENT_LIST)
ppf = loader.load_ppf_data(patient_list=PATIENT_LIST)
patient = loader.load_patient_data(patient_list=PATIENT_LIST)
protocol_similarity = loader.load_protocol_similarity()

# Construct DataUnitSet for processing
units = [session, ppf, patient]
data_unit_set = DataUnitSet(units)

scores = processor.process_data(data_unit_set, scoring_date=pd.Timestamp.today())

# CDSS
cdss = CDSS(scoring=scores, n=n, days=days, protocols_per_day=protocols_per_day)

# Results
patient_id = 1
recommendation = cdss.recommend(
    patient_id=patient_id, protocol_similarity=protocol_similarity
)
recommendation.to_csv(f"recommendation_{patient_id}_new.csv", index=False)

with pd.option_context("display.max_columns", None):
    display(recommendation)

# %%
