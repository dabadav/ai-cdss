# %%

import pandas as pd
from ai_cdss.loaders import DataLoader
from ai_cdss.processing import DataProcessor
from ai_cdss.processing.clinical import ClinicalSubscales
from ai_cdss.processing.processor import ProcessingContext
from IPython.display import display

loader = DataLoader(rgs_mode="plus")
timestamp = pd.Timestamp("2025-06-26 10:28:06")
processor = DataProcessor(context=ProcessingContext(scoring_date=timestamp))

# %%
patients = loader.interface.fetch_patients_by_study([2])
patient_ids = patients.PATIENT_ID.unique().tolist()

# %%
# subscales_map = ClinicalSubscales()
# scales = loader.load_patient_scales(patient_ids)
# subscales_map.compute_deficit_matrix(scales)

# %%
from ai_cdss.interface import CDSSInterface

cdss_client = CDSSInterface(loader, processor)
cdss_client.recommend_for_study(study_id=[2], days=7, protocols_per_day=5, n=12)

# %%2
from ai_cdss.constants import *

patient = loader.load_patient_data(patient_ids)
display(patient.data)
session = loader.load_session_data(patient_ids)
display(session.data)
ppf = loader.load_ppf_data(patient_ids)
display(ppf.data)


# %%
from ai_cdss.models import DataUnitSet

rgs_data = DataUnitSet([session, ppf])
scoring_input = processor.process_data(rgs_data)

# %%
scoring_input

# %%
from ai_cdss.processing.features import include_missing_sessions

scoring_date = processor._get_scoring_date()
session_data = include_missing_sessions(session.data)
session_data[CLINICAL_START] = session_data[CLINICAL_START].dt.normalize()
session_data[CLINICAL_END] = session_data[CLINICAL_END].dt.normalize()
display(session_data)

# %%
from ai_cdss.constants import CLINICAL_END, CLINICAL_START, SESSION_DATE

# Compute upper bound: min(scoring_date, session_data[CLINICAL_END])
date_upper_bound = session_data[CLINICAL_END].where(
    session_data[CLINICAL_END] < scoring_date, scoring_date
)
# Filter sessions in study range
session_data = session_data[
    (session_data[SESSION_DATE] >= session_data[CLINICAL_START])
    & (session_data[SESSION_DATE] <= date_upper_bound)
]
display(session_data)

# %%
weeks_since_start_df = processor.build_week_since_start(session_data)
display(weeks_since_start_df)

# %%
from ai_cdss.constants import BY_PPS, DM_VALUE

dm_df = processor.build_delta_dm(
    session_data[BY_PPS + [SESSION_DATE, DM_VALUE]].dropna()
)  # DELTA_DM
display(dm_df)

# %%
adherence_df = processor.build_recent_adherence(session_data)  # ADHERENCE_RECENT
display(adherence_df)

# %%
usage_df = processor.build_usage(session_data)
display(usage_df)
# %%
usage_week_df = processor.build_week_usage(
    session_data, scoring_date=processor._get_scoring_date()
)  # USAGE_WEEK
display(usage_week_df)
# %%
days_df = processor.build_prescription_days(
    session_data, scoring_date=processor._get_scoring_date()
)  # DAYS
display(days_df)


# %%
# Merge features
from functools import reduce

import pandas as pd
from ai_cdss.constants import BY_PP

# Combine Session Features
feat_pps_df = reduce(
    lambda l, r: pd.merge(l, r, on=BY_PP + [SESSION_DATE], how="left"),
    [adherence_df, dm_df, weeks_since_start_df],
)
display(feat_pps_df)

# %%
from ai_cdss.constants import DELTA_DM, PATIENT_ID, SESSION_INDEX, TOTAL_PRESCRIBED

feat_pp_df = reduce(
    lambda l, r: pd.merge(l, r, on=BY_PP, how="left"),
    [ppf.data, usage_df, usage_week_df, days_df, feat_pps_df],
)
feat_pp_df = feat_pp_df.sort_values(by=BY_PP + [SESSION_DATE])
display(feat_pp_df)

# %%
scoring_input = feat_pp_df.groupby(BY_PP).agg("last").reset_index()
scoring_input = processor._init_metrics(scoring_input)
display(scoring_input)

# %%
from ai_cdss.processing.utils import get_nth

# Weekly imputation of DM and ADHERENCE
delta_nth = get_nth(feat_pp_df, DELTA_DM, BY_PP, SESSION_INDEX, n=1)
delta_medians = delta_nth.groupby(PATIENT_ID)[DELTA_DM].median().reset_index()
scoring_input = processor._impute_metrics(scoring_input, DELTA_DM, delta_medians)

adherence_last = get_nth(feat_pp_df, RECENT_ADHERENCE, BY_PP, SESSION_INDEX, n=-1)
delta_adherence = (
    adherence_last.groupby(PATIENT_ID)[RECENT_ADHERENCE].median().reset_index()
)
scoring_input = processor._impute_metrics(
    scoring_input, RECENT_ADHERENCE, delta_adherence
)
display(scoring_input)

##### Effect on demographics on type of content visited (aliisa definition)
##### Heatmap of areas that people visit the most
# %%
##### Effect on demographics on type of content visited (aliisa definition)
##### Heatmap of areas that people visit the most
# %%
##### Heatmap of areas that people visit the most
# %%
##### Heatmap of areas that people visit the most
# %%
##### Heatmap of areas that people visit the most
# %%
# %%
# %%
# %%
# %%
# %%
