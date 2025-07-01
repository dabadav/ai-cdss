# %%
import pandas as pd
from ai_cdss.loaders.local_loader import DataLoaderLocal
from ai_cdss.processing.processor import DataProcessor
from ai_cdss.services.data_preparation import RecommendationDataService
from ai_cdss.services.ppf_service import PPFService

# Define paths
SESSION_FILE = "data/session_nest.csv"
PATIENT_SUBSCALES_FILE = "data/patient_subscales.csv"
PROTOCOL_SIMILARITY_FILE = "data/protocol_similarity.csv"
PPF_FILE = "data/ppf_data.csv"
MAPPING_FILE = "data/mapping.yaml"
SCALE_FILE = "data/scales.yaml"

# Define the data loader
loader = DataLoaderLocal(
    session_file=SESSION_FILE,
    ppf_file=PPF_FILE,
    protocol_similarity_file=PROTOCOL_SIMILARITY_FILE,
    patient_subscales_file=PATIENT_SUBSCALES_FILE,
)

# Make sure ppf is computed for the patient
patient_list = loader.fetch_and_validate_patients([1])
ppf_service = PPFService(loader, mapping_yaml_path=MAPPING_FILE)
ppf_service.compute_patient_fit(patient_list)

# Instantiate the data processor
processor = DataProcessor()

# Prepare the data using Data Service
data_service = RecommendationDataService(loader)

study_id = [1]
patient_list, rgs_data, protocol_similarity = data_service.prepare(study_id)

timestamp = pd.Timestamp("2025-06-26 10:28:06")
scores = processor.process_data(rgs_data, timestamp or pd.Timestamp.today())

# %%
