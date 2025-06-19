# %%

from ai_cdss.loaders import DataLoader
from ai_cdss.processing import DataProcessor
from IPython.display import display

loader = DataLoader(rgs_mode='plus')
processor = DataProcessor()

# %%
from ai_cdss.interface import CDSSInterface

cdss_client = CDSSInterface(loader, processor)
cdss_client.recommend_for_study(
    study_id=[405],
    days=7,
    protocols_per_day=5,
    n=12
)

# %%
patient_ids = [12]
patient = loader.load_patient_data(patient_ids)
display(patient.data)
session = loader.load_session_data(patient_ids)
display(session.data)
ppf = loader.load_ppf_data(patient_ids)
display(ppf.data)

# %%
# processor.process_data()
from ai_cdss.models import DataUnitSet
rgs_data = DataUnitSet([session, ppf])

# %%
from ai_cdss.processing.features import include_missing_sessions
session_data = include_missing_sessions(session.data)
display(session_data)

# %%
from ai_cdss.constants import SESSION_DATE, CLINICAL_END, CLINICAL_START
session_data = session_data[
    (session_data[SESSION_DATE] >= session_data[CLINICAL_START]) &
    (session_data[SESSION_DATE] <= session_data[CLINICAL_END])
]
display(session_data)

# %%
weeks_since_start_df = processor.build_week_since_start(session_data, processor._get_scoring_date())
display(weeks_since_start_df)

# %%
from ai_cdss.constants import BY_PPS, DM_VALUE

dm_df = processor.build_delta_dm(session_data[BY_PPS + [SESSION_DATE, DM_VALUE]].dropna())  # DELTA_DM
display(dm_df)

# %%
adherence_df = processor.build_recent_adherence(session_data)                               # ADHERENCE_RECENT
display(adherence_df)

# %%
usage_df = processor.build_usage(session_data)
display(usage_df)
# %%
usage_week_df = processor.build_week_usage(session_data, scoring_date=processor._get_scoring_date())         # USAGE_WEEK
display(usage_week_df)
# %%
days_df = processor.build_prescription_days(session_data, scoring_date=processor._get_scoring_date())        # DAYS
display(days_df)


# %%
# Merge features
from functools import reduce
import pandas as pd

from ai_cdss.constants import BY_PP

# Combine Session Features
feat_pps_df = reduce(lambda l, r: pd.merge(l, r, on=BY_PP + [SESSION_DATE], how='left'), [adherence_df, dm_df, weeks_since_start_df])
display(feat_pps_df)

# %%
feat_pp_df  = reduce(lambda l, r: pd.merge(l, r, on=BY_PP, how='left'),  [ppf.data, usage_df, usage_week_df, days_df, feat_pps_df])
feat_pp_df  = feat_pp_df.sort_values(by=BY_PP + [SESSION_DATE])
display(feat_pp_df)

# %%
scoring_input = processor.process_data(rgs_data)







##### Effect on demographics on type of content visited (aliisa definition)
##### Heatmap of areas that people visit the most
# %%
