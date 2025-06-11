from ai_cdss.data_processor import DataProcessor
from ai_cdss.data_loader import DataLoaderMock, DataLoader
from ai_cdss.cdss import CDSS
from ai_cdss.constants import RECENT_ADHERENCE, DELTA_DM, PPF, PATIENT_ID, PROTOCOL_ID, SCORE, DAYS, USAGE
from rgs_interface.data.schemas import RecsysMetricsRow
from IPython.display import display
import pandas as pd
import numpy as np
import datetime
import uuid

def test_scoring_function_nan_values():
    """
    Test if compute score function can handle nan values, used when bootstrapping a patient.
    """
    sample_scoring_df = pd.DataFrame({
        PATIENT_ID: [101, 102, 103, 104, 105],
        PROTOCOL_ID: [301, 302, 303, 304, 305],
        RECENT_ADHERENCE: [0.8, np.nan, np.nan, 0.6, 0.9],
        DELTA_DM: [10, 5, np.nan, 20, 12],
        PPF: [0.5, 0.3, 0.4, 0.3, 0.7],
        DAYS: [[1,2],np.nan,[1],[3],[]],
        USAGE: [1,np.nan,3,4,1]
    })
    
    processor = DataProcessor()
    scored_df = processor.compute_score(sample_scoring_df, None)

    print("Initial sample_scoring_df with NaNs:")
    display(sample_scoring_df)

    print("\nScored DataFrame after NaN handling:")
    display(scored_df)

    expected_scores = pd.Series([11.3, 5.3, 0.4, 20.9, 13.6], name=SCORE)
    pd.testing.assert_series_equal(scored_df[SCORE], expected_scores, check_exact=False, rtol=1e-6)

def test_fetching_prescribed_but_not_performed_patient():
    """
    Get and verify output of DataLoader when fetching a patient with prescribed but not performed sessions in week 0.

    global_dev - `clinical_trials`

    study 404
    study_name TEST_PRESCRIBED_NOT_PERFORMED
    patient 11

    # TODO: Update rgs_interface to filter session prior to study_start
    """
    loader = DataLoader()
    session = loader.load_session_data(patient_list=[11]) # -> Empty
    timeseries = loader.load_timeseries_data(patient_list=[11]) # -> Empty
    ppf = loader.load_ppf_data(patient_list=[11]) # PPF

    display(session)
    display(timeseries)
    display(ppf)

    assert session.empty, "Expected session DataFrame to be empty"
    assert timeseries.empty, "Expected timeseries DataFrame to be empty"
    assert not ppf.empty, "Expected ppf DataFrame to be non-empty"

def test_scoring_prescribed_but_not_performed():
    # Simular `SessionSchema`
    session_df = pd.DataFrame(columns = [
        'PATIENT_ID',
        'PRESCRIPTION_ID',
        'SESSION_ID',
        'PROTOCOL_ID',
        'PRESCRIPTION_STARTING_DATE',
        'PRESCRIPTION_ENDING_DATE',
        'SESSION_DATE',
        'WEEKDAY_INDEX',
        'STATUS',
        'REAL_SESSION_DURATION',
        'PRESCRIBED_SESSION_DURATION',
        'SESSION_DURATION',
        'ADHERENCE',
        'TOTAL_SUCCESS',
        'TOTAL_ERRORS',
        'GAME_SCORE'
    ])

    # Simular `TimeseriesSchema` (sin datos)
    timeseries_df = pd.DataFrame(columns=[
        "PATIENT_ID", "SESSION_ID", "PROTOCOL_ID", 
        "GAME_MODE", "SECONDS_FROM_START", 
        "DM_KEY", "DM_VALUE", "PE_KEY", "PE_VALUE"
    ])
    
    # Simular `PPFSchema`
    ppf_df = pd.DataFrame([{
        'PATIENT_ID': 1,
        'PROTOCOL_ID': 100,
        'PPF': 0.5,
        'CONTRIB': [0.1, 0.2, 0.0, 0.0]
    }])

    processor = DataProcessor()
    scoring_df = processor.process_data(session_data=session_df, timeseries_data=timeseries_df, ppf_data=ppf_df, init_data=None)

    # Verificar métricas
    with pd.option_context('display.max_columns', None):
        display(scoring_df)

    # Verificar métricas
    assert scoring_df['PPF'].iloc[0] == 0.5
    assert isinstance(scoring_df['CONTRIB'].iloc[0], list)
    assert pd.isna(scoring_df['ADHERENCE_RECENT'].iloc[0])
    assert pd.isna(scoring_df['DELTA_DM'].iloc[0])
    assert scoring_df['USAGE'].iloc[0] == 0
    assert scoring_df['SCORE'].iloc[0] == 0.5

def test_recommendation_prescribed_but_not_performed():
    """
    Expected behavior should be to repeat the same prescriptions
    - Mechanism (resend cdss.get_prescriptions) --> Modify SQL query, Modify the session empty expected behavior
    - Mechanism (duplicate prescriptions_staging) --> ??
    """
    scoring_df = pd.DataFrame([{
        'PATIENT_ID': 1,
        'PROTOCOL_ID': 100,
        'PPF': 0.5,
        'CONTRIB': [0.1, 0.2, 0.0, 0.0],
        'ADHERENCE_RECENT': float('nan'),
        'DELTA_DM': float('nan'),
        'USAGE': 0,
        'DAYS': [],
        'SCORE': 0.5
    }])

    loader = DataLoader()
    protocol_similarity = loader.load_protocol_similarity()

    cdss = CDSS(scoring_df)
    prescriptions = cdss.get_prescriptions(1)
    with pd.option_context('display.max_columns', None):
        display(prescriptions)
    
    recommendations = cdss.recommend(1, protocol_similarity=protocol_similarity)
    with pd.option_context('display.max_columns', None):
        display(recommendations)

    # Check that recommendations repeat the same protocol
    assert not recommendations.empty, "Expected recommendations to be returned"
    assert all(recommendations['PROTOCOL_ID'] == 100), "Expected protocol 100 to be repeated"
    assert all(recommendations['PATIENT_ID'] == 1), "Expected recommendations for patient 1"

def test_nan_recsys_metrics_db_insert():
    """
    Test the case where patient is bootstrapped so SCORE depends on PPF only
    And we are inserting to recsys_metrics table
    """
    # Simular `SessionSchema`
    session_df = pd.DataFrame(columns = [
        'PATIENT_ID',
        'PRESCRIPTION_ID',
        'SESSION_ID',
        'PROTOCOL_ID',
        'PRESCRIPTION_STARTING_DATE',
        'PRESCRIPTION_ENDING_DATE',
        'SESSION_DATE',
        'WEEKDAY_INDEX',
        'STATUS',
        'REAL_SESSION_DURATION',
        'PRESCRIBED_SESSION_DURATION',
        'SESSION_DURATION',
        'ADHERENCE',
        'TOTAL_SUCCESS',
        'TOTAL_ERRORS',
        'GAME_SCORE'
    ])

    # Simular `TimeseriesSchema` (sin datos)
    timeseries_df = pd.DataFrame(columns=[
        "PATIENT_ID", "SESSION_ID", "PROTOCOL_ID", 
        "GAME_MODE", "SECONDS_FROM_START", 
        "DM_KEY", "DM_VALUE", "PE_KEY", "PE_VALUE"
    ])
    
    # Simular `PPFSchema`
    ppf_df = pd.DataFrame([{
        'PATIENT_ID': 3,
        'PROTOCOL_ID': 100,
        'PPF': 0.5,
        'CONTRIB': [0.1, 0.2, 0.0, 0.0]
    }])

    processor = DataProcessor()
    scoring_df = processor.process_data(session_data=session_df, timeseries_data=timeseries_df, ppf_data=ppf_df, init_data=None)

    cdss = CDSS(scoring_df)
    prescriptions = cdss.get_prescriptions(3)

    loader = DataLoader()
    protocol_similarity = loader.load_protocol_similarity()

    recommendations = cdss.recommend(3, protocol_similarity=protocol_similarity)

    metrics_df = pd.melt(
        recommendations,
        id_vars=["PATIENT_ID", "PROTOCOL_ID"],
        value_vars=["DELTA_DM", "ADHERENCE_RECENT", "PPF"],
        var_name="KEY",
        value_name="VALUE"
    )
    metrics_df = metrics_df.where(pd.notna(metrics_df), None)

    display(metrics_df)
    
    datetime_now = datetime.datetime.now()
    unique_id = uuid.uuid4()

    for _, row in metrics_df.iterrows():
        recsys_metric_row = RecsysMetricsRow.from_row(
                row, 
                recommendation_id=unique_id,
                metric_date=datetime_now,
            )
        # loader.interface.add_recsys_metric_entry(
        #     recsys_metric_row
        # )
