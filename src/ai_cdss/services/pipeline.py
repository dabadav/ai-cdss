# src/pipeline.py
from ai_cdss.services.data import DataLoader
from ai_cdss.services.processing import ProtocolProcessor, ClinicalProcessor, TimeseriesProcessor, SessionProcessor
from ai_cdss.services.scoring import ScoringComputer
from ai_cdss.services.prescription import PrescriptionRecommender
from ai_cdss.services.processing import create_multiindex_df, compute_protocol_similarity, compute_usage, merge_data, compute_ppf, compute_scoring, schedule, rank_top_n, interchange_mask, substitute_protocol
from functools import reduce

class CDSS:
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
        
        # Data sources
        self.session = None
        self.timeseries = None
        self.patient = None
        self.protocol = None

        # Internal data
        self.patient_protocol_index = None
        self.protocol_similarity = None
        self.protocol_usage = None

        # Data output
        self.scoring = None
        self.prescriptions = None

    def _init_scoring(self):
        patient_protocol_table = create_multiindex_df(self.patient.index, self.protocol.PROTOCOL_ID)
        session_processed, timeseries_processed, patient_deficiency, protocol_mapped = self.process_data()
        
        ppf, contrib = compute_ppf(patient_deficiency, protocol_mapped)

        dfs = [session_processed, timeseries_processed, ppf, contrib, self.protocol_usage]
        scoring = reduce(merge_data, [patient_protocol_table] + dfs)
        
        self.scoring = compute_scoring(scoring, weights=[1,1,1])

    def _init_protocol_similarity(self):
        self.protocol_similarity = compute_protocol_similarity(self.protocol)

    def _init_protocol_usage(self):
        self.protocol_usage = compute_usage(self.session, self.patient_protocol_index)

    def _init_prescriptions(self):
        scoring_ranked = rank_top_n(self.scoring, 12)
        self.prescriptions = schedule(scoring_ranked)

    def update_prescriptions(self):
        scoring_df = self.scoring.copy()
        scoring_df["INTERCHANGE"] = interchange_mask(scoring_df)
        scoring_df["NEW_PROTOCOL_ID"] = scoring_df.apply(
            substitute_protocol,
            axis=1,
            args=(self.protocol_sim, self.usage)
        )

    def run(self):
        """
        Execute the full pipeline sequentially.
        """
        print("Starting data loading...")
        self.load_data()
        print("Processing data...")
        self._init_protocol_usage()
        self._init_scoring()
        self._init_prescriptions()

        print("Pipeline completed successfully!")
        return self.prescriptions.to_csv("recommendations.csv")

    def load_data(self):
        """Extract session and time-series data for patients. Populate internal data structures."""
        self.session = self.data_loader.load_session_data()
        self.timeseries = self.data_loader.load_timeseries_data()
        self.patient = self.data_loader.load_patient_data()
        self.protocol = self.data_loader.load_protocol_data()

    def process_data(self):
        """Process session and time-series data. Clean and validate data"""
        session_processed = SessionProcessor().process(self.session)
        timeseries_processed = TimeseriesProcessor().process(self.timeseries)
        patient_deficiency = ClinicalProcessor().process(self.patient)
        protocol_mapped    = ProtocolProcessor().process(self.protocol)

        return session_processed, timeseries_processed, patient_deficiency, protocol_mapped
    
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
