# src/pipeline.py
from pandera.typing import DataFrame
from ai_cdss.models import BatchSchema
# class CDSS:
#     """Base class for the rehabilitation recommendation pipeline.
    
#     Loads data
#     Computes Metrics
#     Schedules Protocol
#     Returns df
#     """
    
#     def __init__(self, patient_list, n):
#         """
#         Initialize the pipeline with patient list and data sources.

#         """
#         # Parameters
#         self.patient_list = patient_list
#         self.n = n

#         # Data sources
#         self.session = None
#         self.timeseries = None

#         # Data output
#         self.scoring = None
#         self.prescriptions = None

#     def recommend(self):
#         """
#         Execute the full pipeline sequentially.
#         """
#         print("Starting data loading...")
#         self._load(self.patient_list)
#         print("Processing data...")
#         self._score()
#         print("Generate prescriptions...")
#         self._prescribe()
#         self.prescriptions.to_csv("recommendations.csv")
#         print("Pipeline completed successfully!")
#         return self.prescriptions

#     def _load(self, patient_list, rgs_mode="app"):
#         """Extract session and time-series data for patients. Update the state of the class."""
#         # Fetch data from db, both session level data, and timeseries of dm and pe
#         self.session = fetch_rgs_data(patient_list, rgs_mode=rgs_mode)
#         self.timeseries = fetch_timeseries_data(patient_list, rgs_mode=rgs_mode)

#     def _score(self):
#         """Process session and time-series data. Clean and validate data"""
#         # Collapse dms of same timepoints and collapse dm, pe of same session by last value of ewma      
#         timeseries_data = collapse_by_session(self.timeseries)
#         # Compute number of sessions per protocol
#         session_data = compute_usage(self.session)
#         # Collapse adherence, dm of same protocol by last value of ewma
#         rgs_data = collapse_by_protocol(session_data.merge(timeseries_data[["SESSION_ID", "DM_VALUE", "PE_VALUE"]], on="SESSION_ID"))
        
#         # Load last ppf data from ai_cdss directory
#         internal_dir = Path.home() / ".ai_cdss" / "output"
#         ppf_data = pd.read_csv(internal_dir / "ppf.csv")
#         ppf_data = ppf_data[ppf_data.PATIENT_ID.isin(self.patient_list)]

#         # Merge all data in scoring df with all patient, protocol pairs
#         rgs_data = ppf_data.merge(rgs_data, on=["PATIENT_ID", "PROTOCOL_ID"], how="left")[["PATIENT_ID", "PROTOCOL_ID", "USAGE", "PPF", "CONTRIB", "ADHERENCE_EWMA", "DM_VALUE_EWMA"]]
        
#         # Initialize non-performed protocols adh, dm, pe
#         rgs_data.fillna(0, inplace=True)

#         # Compute the linear combination of factors
#         rgs_data = compute_scoring(rgs_data, weights=[1,1,1])

#         # Update class state
#         self.scoring = rgs_data

    # def _prescribe(self):

    #     # Get top n protocols per patient by score
    #     scoring_ranked = rank_top_n(self.scoring, self.n)

    #     # Apply scheduling
    #     self.prescriptions = schedule(scoring_ranked)

    # def update_prescriptions(self, last_week_prescriptions, current_scoring):

    #     # Load protocol similarity from ai_cdss last file
    #     internal_dir = Path.home() / ".ai_cdss" / "output"
    #     protocol_similarity = pd.read_csv(internal_dir / "protocol_fcm.csv", index_col=0)
    #     protocol_similarity.columns = protocol_similarity.columns.astype(int)

    #     # Last prescriptions
    #     prescriptions_df = last_week_prescriptions.copy()
        
    #     # Update last_week prescriptions with new score ans usage from current_scoring
    #     current_scoring = current_scoring[["PATIENT_ID", "PROTOCOL_ID", "SCORE", "USAGE"]].rename(columns={"SCORE": "NEW_SCORE", "USAGE": "NEW_USAGE"})
    #     prescriptions_df = prescriptions_df.merge(current_scoring, on=["PATIENT_ID", "PROTOCOL_ID"], how="left")

    #     # Apply mask to last week prescriptions based on current_scoring
    #     prescriptions_df["INTERCHANGE"] = interchange_mask(prescriptions_df)
        
    #     # For all protocols to interchange find a substitue based on fcm matrix
    #     prescriptions_df["NEW_PROTOCOL_ID"] = prescriptions_df.apply(
    #         substitute_protocol,
    #         axis=1,
    #         args=(protocol_similarity, current_scoring)
    #     )

    #     # Update prescriptions
    #     self.prescriptions = prescriptions_df

class CDSS:
    def __init__(self, batch_data: DataFrame[BatchSchema]):
        self.batch_data = batch_data

    def get_prescriptions(self, patient_id: int):
        raise NotImplementedError
    
    def schedule_protocols(self, patient_id: int):
        raise NotImplementedError
