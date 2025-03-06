# src/prescription.py
import pandas as pd

class PrescriptionRecommender:
    """
    PrescriptionRecommender class for recommending rehabilitation protocols to patients.
    """
    def __init__(self):
        pass


    def rank_protocols(self, scores, n=10):
        """ Rank protocols based on scores.

        Returns a DataFrame with top N protocols for each patient with explanations
        """
        top_n_protocols = scores.groupby('PATIENT_ID').apply(lambda x: x.nlargest(n, 'Score')).reset_index(drop=True)
        return top_n_protocols

    def recommend_protocols(self, ppf_matrix, contributions_matrix=None, top_n=1):
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
        recommendations = {}
        for patient_id in ppf_matrix.index:

            # Get the top N protocols for this patient
            top_protocols = ppf_matrix.loc[patient_id].nlargest(top_n)
            proto_ids = list(top_protocols.index)
            scores = list(top_protocols.values)
            # If contributions are provided, identify top contributing factor for each protocol
            factors = []
            if contributions_matrix is not None:
                # contributions_matrix is NumPy array: [patients x protocols x features]
                pat_index = list(ppf_matrix.index).index(patient_id)
                for proto_id in proto_ids:
                    proto_index = list(ppf_matrix.columns).index(proto_id)
                    # Get contributions for this patient-protocol pair across features
                    contribs = contributions_matrix[pat_index, proto_index, :]
                    # Find feature with maximum contribution
                    max_idx = contribs.argmax()
                    top_feature = ppf_matrix.columns[max_idx] if max_idx < contribs.shape[0] else None
                    factors.append(top_feature)
            else:
                factors = [None] * len(proto_ids)

            # Simple text explanation (can be elaborated)
            if contributions_matrix is not None and factors[0]:
                explanation = f"Recommended protocol {proto_ids[0]} for patient {patient_id} because of high alignment in '{factors[0]}' aspect."
            else:
                explanation = f"Protocol {proto_ids[0]} is the closest match for patient {patient_id}'s profile."
            
            recommendations[patient_id] = {
                "protocols": proto_ids if top_n > 1 else proto_ids[0],
                "scores": scores if top_n > 1 else scores[0],
                "factors": factors,
                "explanation": explanation
            }
        
        return recommendations
    
    def schedule(self):
        raise NotImplementedError
