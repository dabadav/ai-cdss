# src/pipeline.py
from ai_cdss.services.data import DataProcessor
from ai_cdss.services.scoring import ScoringComputer
from ai_cdss.services.prescription import PrescriptionRecommender
from ai_cdss.models import DataBatch

class PipelineBase:
    """Base class for the rehabilitation recommendation pipeline."""
    
    def __init__(self, patient_list, clinical_score_path, protocol_csv_path, mapping_dict):
        """
        Initialize the pipeline with patient list and data sources.
        """
        self.patient_list = patient_list
        self.clinical_score_path = clinical_score_path
        self.protocol_csv_path = protocol_csv_path

        # Instantiate pipeline components
        self.data_processor = DataProcessor(patient_list, mapping_dict)
        self.scoring_computer = ScoringComputer()
        self.prescription_recommender = PrescriptionRecommender()
        
        # Initialize internal data containers
        self.sessions = None
        self.timeseries = None
        self.patient_profiles = None
        self.protocol_profiles = None
        
        self.ppf_matrix = None
        self.recommendations = None
        self.contributions = None

        self.scores = None
        self.prescriptions = None

    def run(self):
        """
        Execute the full pipeline sequentially.
        """
        print("Starting data extraction...")
        self.extract_data()
        print("Processing data...")
        self.process_data()
        print("Computing similarity...")
        self.compute_scores()
        print("Generating prescriptions...")
        self.generate_prescriptions()
        print("Pipeline completed successfully!")

    def extract_data(self):
        """Extract session and time-series data for patients. Populate internal data structures."""
        self.sessions = self.data_processor.get_session()
        self.timeseries = self.data_processor.get_timeseries()
        self.patient_profiles = self.data_processor.get_patient(self.clinical_score_path)
        self.protocol_profiles = self.data_processor.get_protocol(self.protocol_csv_path)

    def process_data(self):
        """Process session and time-series data. Clean and validate data"""
        self.sessions = self.data_processor.process_session_batch(self.sessions)
        # self.sessions = self.data_processor.expand_expected_sessions(self.sessions)
        self.timeseries = self.data_processor.process_timeseries_data(self.timeseries)
        self.protocol_profiles = self.data_processor.map_latent_to_clinical(self.protocol_profiles)

    def compute_scores(self):
        """Compute Patient-Protocol Fit (PPF) and protocol similarity."""
        self.ppf_matrix, self.contributions = self.scoring_computer.compute_ppf(self.patient_profiles, self.protocol_profiles)
        self.sessions = self.scoring_computer.compute_adherence(self.sessions)
        self.protocol_similarity = self.scoring_computer.compute_protocol_similarity(self.protocol_profiles)
        self.prescriptions = self.scoring_computer.compute_score(self.sessions, self.timeseries, self.ppf_matrix, self.contributions)

    def generate_prescriptions(self):
        """Generate protocol recommendations."""
        self.recommendations = self.prescription_recommender.recommend_protocols(self.prescriptions)
        self.display_recommendations()

    def update_prescriptions(self):
        raise NotImplementedError

    def display_recommendations(self):
        """Display recommendations in a readable format."""
        for patient_id, rec in self.recommendations.items():
            print(f"Patient {patient_id}: Recommended Protocol {rec['protocols']} (Score: {rec['scores']:.3f})")
            print(f" - Explanation: {rec['explanation']}")
