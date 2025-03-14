import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# PARAMS
# ------
# patient_deficit_df
# session_df
# protocol_df

class ScoringComputer:
    """
    Combining patient data and protocol data
    
    Computation (independent) on 
        session data
        protocol data

    Aggregating a score dependent on:
        session data
        timeseries data
        ppf
        contributions
    """
    def __init__(self):
        self.ppf = None
        self.contributions = None
        self.protocol_similarity = None
        self.patient_protocol_score = None

    def compute_ppf(self, patient_data, protocol_data):
        """ Compute the patient-protocol feature matrix (PPF) and feature contributions.
        """
        contributions = self.feature_contributions(patient_data, protocol_data)
        ppf = np.sum(contributions, axis=2)

        ppf = pd.DataFrame(ppf, index=patient_data.index, columns=protocol_data.index)
        contributions = pd.DataFrame(contributions.tolist(), index=patient_data.index, columns=protocol_data.index)
        
        self.ppf = ppf
        self.contributions = contributions

        return ppf, contributions
    
    def compute_adherence(self, session_batch, alpha=0.8):
        """ Compute adherence scores.
        """
        session_batch['ADHERENCE_EWMA'] = (
            session_batch.groupby(['PATIENT_ID', 'PROTOCOL_ID'])['ADHERENCE']
            .transform(lambda x: x.ewm(alpha=alpha, adjust=True).mean())
        )
        return session_batch
    
    def compute_protocol_similarity(self, protocol_data):
        """ Compute protocol similarity.
        """
        import gower

        protocol_attributes = protocol_data
        protocol_ids = protocol_attributes.index
        # protocol_attributes.drop(columns="PROTOCOL_ID", inplace=True)
        hot_encoded_cols = protocol_attributes.columns.str.startswith("BODY_PART")
        weights = np.ones(len(protocol_attributes.columns))
        weights[hot_encoded_cols] = weights[hot_encoded_cols] / hot_encoded_cols.sum()
        protocol_attributes = protocol_attributes.astype(float)

        gower_sim_matrix = gower.gower_matrix(protocol_attributes, weight=weights)
        gower_sim_matrix = pd.DataFrame(1- gower_sim_matrix, index=protocol_ids, columns=protocol_ids)

        return gower_sim_matrix

    def compute_score(self, session_batch, timeseries_date, ppf, contributions, weights = [1, 1, 1]):
        """ Compute scores for each patient based on the PPF similarity matrix.
        """
        # Aggregate dfs by last
        dms_agg = timeseries_date.groupby(['PATIENT_ID', 'PROTOCOL_ID']).last().reset_index()
        data_agg = session_batch.groupby(['PATIENT_ID', 'PROTOCOL_ID']).last().reset_index()

        # Merge dataframes on PATIENT_ID and PROTOCOL_ID
        data_all = data_agg[['PATIENT_ID', 'PROTOCOL_ID', 'ADHERENCE_EWMA']].merge(
            dms_agg[['PATIENT_ID', 'PROTOCOL_ID', 'PARAMETER_VALUE_EWMA', 'PERFORMANCE_VALUE_EWMA']],
            on=['PATIENT_ID', 'PROTOCOL_ID'],
            how='inner'
        )

        # To long format
        ppf_stacked = ppf.stack().reset_index()
        ppf_stacked.columns = ['PATIENT_ID', 'PROTOCOL_ID', 'PPF']
        
        contributions_stacked = contributions.stack().reset_index()
        contributions_stacked.columns = ['PATIENT_ID', 'PROTOCOL_ID', 'CONTRIBUTION']
        
        data_all = ppf_stacked.merge(data_all, left_on=['PATIENT_ID', 'PROTOCOL_ID'], right_on=['PATIENT_ID', 'PROTOCOL_ID'], how='left')
        data_all = data_all.merge(contributions_stacked, left_on=['PATIENT_ID', 'PROTOCOL_ID'], right_on=['PATIENT_ID', 'PROTOCOL_ID'], how='left')

        # Fill Non Played Protocols
        data_all.fillna(value={"ADHERENCE_EWMA": 1, "PARAMETER_VALUE_EWMA": 0, "PERFORMANCE_VALUE_EWMA": 0}, inplace=True)
        data_all.sort_values(by=['PATIENT_ID', 'PROTOCOL_ID'], inplace=True)

        # Compute the weighted combination
        data_all['Score'] = (
            data_all['ADHERENCE_EWMA'] * weights[0] +
            data_all['PARAMETER_VALUE_EWMA'] * weights[1] +
            data_all['PPF'] * weights[2]
        )

        self.patient_protocol_score = data_all

        return data_all

    @staticmethod 
    def feature_contributions(df_A, df_B):
        # Convert to numpy
        A = df_A.to_numpy()
        B = df_B.to_numpy()

        # Compute row-wise norms
        A_norms = np.linalg.norm(A, axis=1, keepdims=True)
        B_norms = np.linalg.norm(B, axis=1, keepdims=True)
        
        # Replace zero norms with a small value to avoid NaN (division by zero)
        A_norms[A_norms == 0] = 1e-10
        B_norms[B_norms == 0] = 1e-10

        # Normalize each row to unit vectors
        A_norm = A / A_norms
        B_norm = B / B_norms

        # Compute feature contributions
        contributions = A_norm[:, np.newaxis, :] * B_norm[np.newaxis, :, :]

        return contributions

# ---------------------------------------------------------------------
# PPF

def feature_contributions(df_A, df_B):
    # Convert to numpy
    A = df_A.to_numpy()
    B = df_B.to_numpy()

    # Compute row-wise norms
    A_norms = np.linalg.norm(A, axis=1, keepdims=True)
    B_norms = np.linalg.norm(B, axis=1, keepdims=True)
    
    # Replace zero norms with a small value to avoid NaN (division by zero)
    A_norms[A_norms == 0] = 1e-10
    B_norms[B_norms == 0] = 1e-10

    # Normalize each row to unit vectors
    A_norm = A / A_norms
    B_norm = B / B_norms

    # Compute feature contributions
    contributions = A_norm[:, np.newaxis, :] * B_norm[np.newaxis, :, :]

    return contributions

def compute_ppf(patient_data, protocol_data):
    """ Compute the patient-protocol feature matrix (PPF) and feature contributions.
    """
    contributions = feature_contributions(patient_data, protocol_data)
    ppf = np.sum(contributions, axis=2)
    ppf = pd.DataFrame(ppf, index=patient_data.index, columns=protocol_data.index)
    contributions = pd.DataFrame(contributions.tolist(), index=patient_data.index, columns=protocol_data.index)
    
    return ppf, contributions

# ADD
#     Contributions      , PPF
# --> Series[List[float]], Series[float]

# ---------------------------------------------------------------------
# Adherence

def compute_adherence(session_batch, alpha=0.8):
    """ Compute adherence scores.
    """
    session_batch['ADHERENCE_EWMA'] = (
        session_batch.groupby(['PATIENT_ID', 'PROTOCOL_ID'])['ADHERENCE']
        .transform(lambda x: x.ewm(alpha=alpha, adjust=True).mean())
    )
    return session_batch

## Population
### PPF
### Adherence
### DeltaDM
        
def compute_protocol_similarity(protocol_data):
    """ Compute protocol similarity.
    """
    import gower

    protocol_attributes = protocol_data.copy()
    protocol_ids = protocol_attributes.PROTOCOL_ID
    protocol_attributes.drop(columns="PROTOCOL_ID", inplace=True)

    hot_encoded_cols = protocol_attributes.columns.str.startswith("BODY_PART")
    weights = np.ones(len(protocol_attributes.columns))
    weights[hot_encoded_cols] = weights[hot_encoded_cols] / hot_encoded_cols.sum()
    protocol_attributes = protocol_attributes.astype(float)

    gower_sim_matrix = gower.gower_matrix(protocol_attributes, weight=weights)
    gower_sim_matrix = pd.DataFrame(1- gower_sim_matrix, index=protocol_ids, columns=protocol_ids)

    return gower_sim_matrix
    
# ---------------------------------------------------------------------
# RETURNS
# ------
# scoring_df: DataFrame[ScoringSchema]