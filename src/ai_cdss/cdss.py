# src/pipeline.py
import math
from typing import Dict, List

from pandera.typing import DataFrame
from ai_cdss.models import ScoringSchema

class CDSS:
    """Clinical Decision Support System Class"""
    def __init__(self, scoring: DataFrame[ScoringSchema], n: int = 12, days: int = 7, protocols_per_day: int = 5):
        self.scoring = scoring
        self.n = n
        self.days = days
        self.protocols_per_day = protocols_per_day

    def recommend(self, patient_id: int, protocol_similarity) -> Dict[str, str]:
        """
        Recommend prescriptions for a patient.

        Args:
            patient_id (int): The ID of the patient.
            protocol_similarity: A DataFrame containing protocol similarity scores.

        Returns:
            Dict[str, str]: A dictionary mapping original protocol IDs to recommended protocol IDs.
        """
        recommendations = {}

        # Get current prescriptions for the patient
        prescriptions = self.get_prescriptions(patient_id)

        if not prescriptions.empty:
            # If the patient has prescriptions, decide which ones to swap
            protocols_to_swap = self.decide_prescription_swap(patient_id)
            for protocol_id in protocols_to_swap:
                substitute = self.get_substitute(patient_id, protocol_id, protocol_similarity)
                if substitute:
                    recommendations[protocol_id] = substitute
        else:

            # If the patient has no prescriptions, recommend the top N protocols
            top_protocols = self.get_top_protocols(patient_id, self.n)
            recommendations = self.schedule_protocols(top_protocols)

        return recommendations

    def schedule_protocols(self, protocols: List[int]):
        """
        Distribute protocols across days, ensuring no day exceeds the allowed number of protocols.

        Args:
            protocols (List[str]): The list of protocols to distribute.
            days (int): The number of days to distribute the protocols.
            protocols_per_day (int): The maximum number of protocols per day.

        Returns:
            Dict[int, List[str]]: A dictionary mapping days to the list of protocols scheduled for that day.
        """

        schedule = {day: [] for day in range(1, self.days + 1)}  # Days are 1-indexed
        total_slots = self.days * self.protocols_per_day

        # Repeat protocols as needed to fill the total slots
        repeated_protocols = (protocols * math.ceil(total_slots / len(protocols)))[:total_slots]

        # Distribute protocols evenly across days
        for i, protocol in enumerate(repeated_protocols):
            day = (i % self.days) + 1  # Distribute protocols in a round-robin fashion
            
            if protocol not in schedule[day]:
                schedule[day].append(protocol)

        return schedule

    def decide_prescription_swap(self, patient_id: int) -> List[int]:
        """
        Decide whether to swap a prescription based on its score and marginal value.

        Args:
            prescription (Dict): A dictionary containing prescription details.

        Returns:
            Optional[str]: The ID of the substitute protocol if a swap is needed, or None if no swap is needed.
        """
        prescriptions = self.get_prescriptions(patient_id)
        return prescriptions[prescriptions['SCORE'].transform(lambda x: x < x.mean())].PROTOCOL_ID.values

    def get_substitute(self, patient_id: int, protocol_id: int, protocol_similarity, protocol_excluded: List[int] = None):
        # Get protocol usage for the given patient and protocol
        usage = (
            self.scoring[self.scoring["PATIENT_ID"] == patient_id]
            .set_index("PROTOCOL_ID")["USAGE"]   
        )     
        # Get protocol similarities
        similarities = protocol_similarity[
            (protocol_similarity["PROTOCOL_A"] == protocol_id)
        ]
        
        # Drop rows where PROTOCOL_B is the same as PROTOCOL_A (self-similarity)
        similarities = similarities[similarities["PROTOCOL_A"] != similarities["PROTOCOL_B"]]
        
        # Exclude protocols in the `protocol_excluded` list from similarities
        if protocol_excluded:
            similarities = similarities[~similarities["PROTOCOL_B"].isin(protocol_excluded)]
        
        # Find the minimum usage value
        min_usage = usage.min()
        
        # Get candidates with the lowest usage
        candidates = usage[usage == min_usage].index
        
        # Among these candidates, select the one with the highest similarity
        candidate_similarities = similarities[similarities["PROTOCOL_B"].isin(candidates)]
        
        # Find the maximum similarity among candidates
        if not candidate_similarities.empty:
            max_sim = candidate_similarities["SIMILARITY"].max()
            
            final_candidates = candidate_similarities[
                candidate_similarities["SIMILARITY"] == max_sim
            ]["PROTOCOL_B"]
            
            # Return the first candidate (or handle ties)
            return final_candidates.iloc[0] if not final_candidates.empty else None
        
        else:
            return None

    def get_top_protocols(self, patient_id: int) -> List[int]:
        """
        Select the top N protocols for a patient based on their scores.

        Args:
            patient_id (int): The ID of the patient.
            top_n (int): The number of top protocols to select. Default is 12.

        Returns:
            List[str]: A list of top protocol IDs.
        """
        patient_data = self.scoring[self.scoring["PATIENT_ID"] == patient_id]
        top_protocols = patient_data.nlargest(self.n, "SCORE")["PROTOCOL_ID"].tolist()
        return top_protocols
    
    def get_prescriptions(self, patient_id: int):
        """
        Retrieve the current prescriptions for a patient.

        Args:
            patient_id (int): The ID of the patient.

        Returns:
            List[Dict]: A list of prescriptions, where each prescription is a dictionary
                        containing protocol details.
        """
        patient_data = self.scoring[self.scoring["PATIENT_ID"] == patient_id]
        prescriptions = patient_data[patient_data["DAYS"].apply(lambda x: isinstance(x, list) and len(x) > 0)]
        
        return prescriptions
