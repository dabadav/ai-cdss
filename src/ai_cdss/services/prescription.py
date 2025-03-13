# src/prescription.py
import pandas as pd
import numpy as np
import pandera as pa
from pandera.typing import DataFrame
from ai_cdss.models import PrescriptionSchema

class PrescriptionRecommender:
    """
    PrescriptionRecommender class for recommending rehabilitation protocols to patients.
    """
    def __init__(self):
        pass

    def rank_protocols(self, scores: pd.DataFrame, n=5) -> pd.DataFrame:
        """ Rank protocols based on scores.

        Returns a DataFrame with top N protocols for each patient with explanations
        """
        protocols_ranked = (
            scores.groupby('PATIENT_ID')
            .apply(lambda x: x.nlargest(n, 'Score'))
            .reset_index(drop=True)
        )
        return protocols_ranked

    @pa.check_types
    def recommend_protocols(self, scores: pd.DataFrame, n=5) -> DataFrame[PrescriptionSchema]:
        """
        Recommend protocols for each patient based on the PPF similarity matrix.
        - ppf_matrix: DataFrame of cosine similarities (patients x protocols).
        - contributions_matrix: optional array of feature contributions for explanation.
        - top_n: how many top protocols to recommend (default 1 for a single best protocol).
        Returns a dict of recommendations, where each key is a patient ID and the value is a dict with details:
        { "protocols": [list of protocol IDs],
            "scores": [corresponding similarity scores],
            "factors": [list of top contributing factors per protocol],
            "explanation": "text explanation of recommendation" }
        """

        # Rank
        rank_df = self.rank_protocols(scores, n)
        # Schedule
        recommendations = self.schedule(rank_df, days_per_week=7, prescriptions_per_day=5)

        return recommendations
    
    def schedule(self, df, days_per_week=7, prescriptions_per_day=5):
        """
        Generates a weekly schedule for each patient by distributing their top recommended protocols across the week.
        Ensures that:
        1. The same protocol is not scheduled twice in a single day.
        2. The total number of prescriptions is exactly `days_per_week * prescriptions_per_day`.
        
        Args:
        df (pd.DataFrame): Long format DataFrame with columns ['PATIENT_ID', 'PROTOCOL_ID'].
        days_per_week (int): Number of days in the schedule (default: 7).
        prescriptions_per_day (int): Number of protocols per day (default: 5).
        
        Returns:
        pd.DataFrame: A DataFrame where each row corresponds to a (PATIENT_ID, PROTOCOL_ID) pair,
                    and the 'DAYS' column contains a list of day indexes (1-based) for when the protocol should be played.
        """
        total_prescriptions = days_per_week * prescriptions_per_day
        schedule_dict = {}

        for patient_id, group in df.groupby("PATIENT_ID"):
            protocols = group["PROTOCOL_ID"].tolist()

            # Expand protocol list to ensure at least `total_prescriptions`
            expanded_protocols = (protocols * ((total_prescriptions // len(protocols)) + 1))[:total_prescriptions]

            # Shuffle protocols for distribution across days
            np.random.shuffle(expanded_protocols)

            # Assign protocols to days ensuring no duplicates in a single day
            patient_schedule = {protocol: [] for protocol in protocols}
            day_protocols = [[] for _ in range(days_per_week)]
            
            for i, protocol in enumerate(expanded_protocols):
                day_idx = i % days_per_week
                if protocol not in day_protocols[day_idx]:  # Ensure no duplicate protocol on the same day
                    day_protocols[day_idx].append(protocol)
                    patient_schedule[protocol].append(day_idx + 1)  # Use 1-based indexing for days

            schedule_dict[patient_id] = patient_schedule

        # Convert to long format DataFrame
        structured_schedule = []
        for patient_id, protocols in schedule_dict.items():
            for protocol_id, days in protocols.items():
                structured_schedule.append({"PATIENT_ID": patient_id, "PROTOCOL_ID": protocol_id, "DAYS": days})
    
        schedule_df = pd.DataFrame(structured_schedule)
        df["DAYS"] = schedule_df.DAYS
        
        return df
            
