# src/pipeline.py
from ai_cdss.services.data import DataLoader
from ai_cdss.services.processing import ProtocolProcessor, ClinicalProcessor, TimeseriesProcessor
from ai_cdss.services.scoring import ScoringComputer
from ai_cdss.services.prescription import PrescriptionRecommender

class PipelineBase:
    """Base class for the rehabilitation recommendation pipeline.
    
    Loads data
    Computes Metrics
    Schedules Protocol
    Returns df
    """
    
    def __init__(self, patient_list):
        """
        Initialize the pipeline with patient list and data sources.
        """
        self.patient_list = patient_list
        
        # Instantiate pipeline components
        self.data_loader = DataLoader(patient_list)
        self.scoring_computer = ScoringComputer()
        self.prescription_recommender = PrescriptionRecommender()
        
        self.sessions = None
        self.timeseries = None
        self.patient = None
        self.protocol = None

    def run(self):
        """
        Execute the full pipeline sequentially.
        """
        print("Starting data loading...")
        self.loading_data()
        print("Processing data...")
        self.process_data()
        print("Computing similarity...")
        self.compute_scores()
        print("Generating prescriptions...")
        result = self.generate_prescriptions()
        print("Pipeline completed successfully!")
        return result.to_csv("recommendations.csv")

    def load_data(self):
        """Extract session and time-series data for patients. Populate internal data structures."""
        self.sessions = self.data_loader.load_session_data()
        self.timeseries = self.data_loader.load_timeseries_data()
        self.patient = self.data_loader.load_patient_data()
        self.protocol = self.data_loader.load_protocol_data()

    def process_data(self):
        """Process session and time-series data. Clean and validate data"""
        patient_deficiency = ClinicalProcessor().process(self.patient)
        protocol_mapped    = ProtocolProcessor().process(self.protocol)
        timeseries_processed = TimeseriesProcessor().process(self.timeseries)

        return patient_deficiency, protocol_mapped, timeseries_processed
    
    def compute_scores(self):
        """Compute Patient-Protocol Fit (PPF) and protocol similarity."""
        self.ppf_matrix, self.contributions = self.scoring_computer.compute_ppf(self.patient_profiles, self.protocol_profiles)
        self.sessions = self.scoring_computer.compute_adherence(self.sessions)
        self.protocol_similarity = self.scoring_computer.compute_protocol_similarity(self.protocol_profiles)
        self.prescriptions = self.scoring_computer.compute_score(self.sessions, self.timeseries, self.ppf_matrix, self.contributions)

    def generate_prescriptions(self):
        """Generate protocol recommendations."""
        self.recommendations = self.prescription_recommender.recommend_protocols(self.prescriptions)
        return self.recommendations

    def update_prescriptions(self):
        """
        **TODO**
        
        Load Last week Prescription
        Add swapping implementations from pipeline.ipynba to prescription module
        """
        raise NotImplementedError

    def display_recommendations(self):
        """Display recommendations in a readable format."""
        for patient_id, rec in self.recommendations.items():
            print(f"Patient {patient_id}: Recommended Protocol {rec['protocols']} (Score: {rec['scores']:.3f})")
            print(f" - Explanation: {rec['explanation']}")
