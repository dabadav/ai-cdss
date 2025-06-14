"""
Basic usage of CDSS to generate a protocol recommendation.
==========================================================

This example demonstrates how to use the `DataLoader`, `DataProcessor`, and `CDSS`
to load data and generate a 7-day protocol plan.
"""

# %%

import sys
sys.path.append("..")
from ai_cdss.cdss import CDSS
from ai_cdss.data_loader import DataLoaderMock
from ai_cdss.data_processor import DataProcessor

import pandas as pd
from IPython.display import display

print(__doc__)

PATIENT_LIST = [
    775,  787,  788
]

# Parameters
rgs_mode = "app"
weights = [1,1,1]
alpha = 0.5

n = 12
days = 7
protocols_per_day = 5

# Services
loader = DataLoaderMock(
    num_patients=5,
    num_protocols=5,
    num_sessions=10
)
processor = DataProcessor(
    weights=weights,
    alpha=alpha
)

# Execution
session = loader.load_session_data(patient_list=PATIENT_LIST)
timeseries = loader.load_timeseries_data(patient_list=PATIENT_LIST)
ppf = loader.load_ppf_data(patient_list=PATIENT_LIST)
init_metrics = loader.load_protocol_init()
protocol_similarity = loader.load_protocol_similarity()

scores = processor.process_data(
    session_data=session, 
    timeseries_data=timeseries, 
    ppf_data=ppf,
    init_data=init_metrics)

# CDSS
cdss = CDSS(
    scoring=scores,
    n=n,
    days=days,
    protocols_per_day=protocols_per_day
)

# Results
# patient_id = PATIENT_LIST[0]
patient_id = 1

recommendation = cdss.recommend(patient_id=patient_id, protocol_similarity=protocol_similarity)
recommendation.to_csv(f"recommendation_{patient_id}_new.csv", index=False)

with pd.option_context('display.max_columns', None):
    display(recommendation)

# %%
