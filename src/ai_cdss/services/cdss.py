
#################################
# --------- Recommend --------- #
#################################

class CDSS:

    def __init__(self, scoring, protocol_similarity):
        self.scoring = scoring
        self.protocol_similarity = protocol_similarity

    def data_merge(self, patient_deficiency, protocol_mapped, session_processed, timeseries_processed):
        self.patient_deficiency = patient_deficiency
        self.protocol_mapped = protocol_mapped
        
