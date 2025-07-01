# %%
from typing import List
import pandas as pd
from pathlib import Path
from ai_cdss.loaders.local_loader import DataLoaderLocal
from ai_cdss.processing.processor import DataProcessor
from ai_cdss.services.data_preparation import RecommendationDataService
from ai_cdss.services.ppf_service import PPFService
from ai_cdss.constants import CLINICAL_START, CLINICAL_END, PRESCRIPTION_STARTING_DATE, PRESCRIPTION_ENDING_DATE
from IPython.display import display
# %%

# Define paths
study = "phrase"
DATA_DIR = Path("data")
STUDY_DIR = DATA_DIR / study

SESSION_FILE = STUDY_DIR / f"session_{study}.csv"
PATIENT_SUBSCALES_FILE = STUDY_DIR / f"subscales_{study}.csv"
MAPPING_FILE = STUDY_DIR / f"mapping_{study}.yaml"
PROTOCOL_SIMILARITY_FILE = DATA_DIR / "protocol_similarity.csv"
PROTOCOL_ATTRIBUTES_FILE = DATA_DIR / "protocol_attributes.csv"

# %%

# PPF_FILE = "data/ppf_data.csv"
# SCALE_FILE = "data/scales.yaml"

# Define the data loader
loader = DataLoaderLocal(
    session_file=SESSION_FILE,
    ppf_file=None,
    protocol_similarity_file=PROTOCOL_SIMILARITY_FILE,
    patient_subscales_file=PATIENT_SUBSCALES_FILE,
    protocol_attributes_file=PROTOCOL_ATTRIBUTES_FILE
)

# Instantiate the data processor
processor = DataProcessor()

# Make sure ppf is computed for the patient
# patient_list = loader.fetch_and_validate_patients([1])
# ppf_service = PPFService(loader, mapping_yaml_path=MAPPING_FILE)
# ppf_contrib = ppf_service.compute_patient_fit(patient_list)
# ppf_service.persist_ppf(ppf_contrib)

# %%


# Prepare the data using Data Service
data_service = RecommendationDataService(loader)

study_id = [1]
patient_list, rgs_data, protocol_similarity = data_service.prepare(study_id)

timestamp = pd.Timestamp("2025-06-26 10:28:06")
scores = processor.process_data(rgs_data, timestamp or pd.Timestamp.today())
# %%
display(scores)


# %%

# Function to generate synthetic timestamps
def generate_weekly_timestamps(df, patient_col='PATIENT_ID', start_col=CLINICAL_START):
    """
    For each patient, generate two timestamps: one in first week, one in second week.

    Args:
        df (pd.DataFrame): The rgs_data dataframe containing patient info.
        patient_col (str): Name of the patient ID column.
        start_col (str): Name of the clinical start date column.

    Returns:
        pd.DataFrame: Dataframe with patient_id and synthetic timestamps.
    """
    timestamps = []
    df = df.copy()
    df = df.drop_duplicates(subset=['PATIENT_ID'])
    for _, row in df.iterrows():

        patient_id = row[patient_col]
        start_date = pd.to_datetime(row[start_col])

        first_week_ts = start_date + pd.Timedelta(weeks=1)   # Mid first week
        second_week_ts = start_date + pd.Timedelta(weeks=2)  # Mid second week

        timestamps.append({'patient_id': patient_id, 'synthetic_timestamp': first_week_ts})
        timestamps.append({'patient_id': patient_id, 'synthetic_timestamp': second_week_ts})

    return pd.DataFrame(timestamps)

# Generate timestamps
timestamps_df = generate_weekly_timestamps(pd.read_csv(SESSION_FILE))

# %%
from ai_cdss.models import DataUnitSet

def prepare_for_patient_list(patient_list: List[int]):
    ppf = loader.load_ppf_data(patient_list)
    missing = ppf.metadata.get("missing_patients", [])
    if missing:
        raise RuntimeError(
            f"PPF data missing for patients: {missing}. Please compute PPF before proceeding."
        )
    session = loader.load_session_data(patient_list)
    protocol_similarity = loader.load_protocol_similarity()
    rgs_data = DataUnitSet([session, ppf])
    return patient_list, rgs_data, protocol_similarity

# %%
# Now process each (patient, timestamp) pair
all_scores = []

for _, row in timestamps_df.iterrows():
    patient_id = row['patient_id']
    ts = row['synthetic_timestamp']

    # Optional: Filter rgs_data for this patient, if your processor expects single-patient input
    patient_list, rgs_data, protocol_similarity = prepare_for_patient_list([patient_id])

    # Process
    score = processor.process_data(rgs_data, ts)
    all_scores.append({'patient_id': patient_id, 'timestamp': ts, 'score': score})

# Optionally convert all_scores to DataFrame
results_df = pd.DataFrame(all_scores)

# Display
display(results_df)
