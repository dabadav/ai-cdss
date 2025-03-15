# src/pipeline.py
from ai_cdss.services.data import DataLoader
from ai_cdss.services.processing import ProtocolProcessor, ClinicalProcessor, TimeseriesProcessor, SessionProcessor
from ai_cdss.services.processing import create_multiindex_df, compute_protocol_similarity, compute_usage, merge_data, compute_ppf, compute_scoring, schedule, rank_top_n, interchange_mask, substitute_protocol, multiindex
from functools import reduce

class CDSS:
    """Base class for the rehabilitation recommendation pipeline.
    
    Loads data
    Computes Metrics
    Schedules Protocol
    Returns df
    """
    
    def __init__(self, patient_list, n):
        """
        Initialize the pipeline with patient list and data sources.

        """
        
        # Parameters
        self.patient_list = patient_list
        self.n = n

        # Instantiate pipeline components
        self.data_loader = DataLoader(patient_list)
        
        # Data sources
        self.session = None
        self.timeseries = None
        self.patient = None
        self.protocol = None

        # Internal data
        self.protocol_similarity = None
        self.protocol_usage = None

        # Data output
        self.scoring = None
        self.prescriptions = None

    def run(self):
        """
        Execute the full pipeline sequentially.
        """
        print("Starting data loading...")
        self.load_data()
        print("Processing data...")
        self._init_protocol_similarity()
        self._init_protocol_usage()
        self._init_scoring()
        self._init_prescriptions()

        self.prescriptions.to_csv("recommendations.csv")
        print("Pipeline completed successfully!")
        return self.prescriptions

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
        patient_protocol_index = multiindex(self.patient.index, self.protocol.PROTOCOL_ID)
        self.protocol_usage = compute_usage(self.session, patient_protocol_index)

    def _init_prescriptions(self):
        scoring_ranked = rank_top_n(self.scoring, self.n)
        self.prescriptions = schedule(scoring_ranked)

    def update_prescriptions(self, last_week_prescriptions, current_scoring):
        prescriptions_df = last_week_prescriptions.copy()

        # Update last_week with new score ans usage from current_scoring
        current_scoring = current_scoring[["PATIENT_ID", "PROTOCOL_ID", "SCORE", "USAGE"]].rename(columns={"SCORE": "NEW_SCORE", "USAGE": "NEW_USAGE"})
        
        # Merge last week's prescriptions with the current scoring updates
        prescriptions_df = prescriptions_df.merge(current_scoring, on=["PATIENT_ID", "PROTOCOL_ID"], how="left")
        
        # Apply mask to last week prescriptions based on current_scoring
        prescriptions_df["INTERCHANGE"] = interchange_mask(prescriptions_df)

        prescriptions_df["NEW_PROTOCOL_ID"] = prescriptions_df.apply(
            substitute_protocol,
            axis=1,
            args=(self.protocol_similarity, self.protocol_usage)
        )

        self.prescriptions = prescriptions_df

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
    
    def display_recommendations(self):
        """Display recommendations in a readable format."""
        for patient_id, rec in self.recommendations.items():
            print(f"Patient {patient_id}: Recommended Protocol {rec['protocols']} (Score: {rec['scores']:.3f})")
            print(f" - Explanation: {rec['explanation']}")

