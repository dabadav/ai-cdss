
from ai_cdss.data_loader import DataLoaderMock

def test_mock_loader():
    loader = DataLoaderMock(
        num_patients=2,
        num_protocols=3,
        num_sessions=2
    )

    session_df = loader.load_session_data()
    timeseries_df = loader.load_timeseries_data()
    ppf_df = loader.load_ppf_data(patient_list=[])
    protocol_metrics = loader.load_protocol_init()  

    print("=== Timeseries ===")
    print(timeseries_df.shape)
    print(timeseries_df.head())

    print("=== Session ===")
    print(session_df.shape)
    print(session_df.head())

    print("=== PPF ===")
    print(ppf_df.shape)
    print(ppf_df.head())

    print("=== Protocol metrics ===")
    print(protocol_metrics.shape)
    print(protocol_metrics.head())

    # Assert DataFrames are not empty
    assert not timeseries_df.empty, "Timeseries DataFrame is empty"
    assert not session_df.empty, "Session DataFrame is empty"
    assert not ppf_df.empty, "PPF DataFrame is empty"
    assert not protocol_metrics.empty, "Protocol metrics DataFrame is empty"

    # Check all dataframes have same unique PATIENT_IDs
    ts_patients = set(timeseries_df["PATIENT_ID"].unique())
    ses_patients = set(session_df["PATIENT_ID"].unique())
    ppf_patients = set(ppf_df["PATIENT_ID"].unique())

    assert ts_patients == ses_patients == ppf_patients, \
        "Mismatch in unique PATIENT_IDs across DataFrames"

    # Check all dataframes have same unique PATIENT_IDs
    ts_protocols = set(timeseries_df["PROTOCOL_ID"].unique())
    ses_protocols = set(session_df["PROTOCOL_ID"].unique())
    ppf_protocols = set(ppf_df["PROTOCOL_ID"].unique())
    metrics_protocols = set(protocol_metrics["PROTOCOL_ID"].unique())

    assert ts_protocols == ses_protocols == ppf_protocols == metrics_protocols, \
        "Mismatch in unique PROTOCOL_IDs across DataFrames"

    # Check all dataframes have same unique PATIENT_IDs
    ts_sessions = set(timeseries_df["SESSION_ID"].unique())
    ses_sessions = set(session_df["SESSION_ID"].unique())
    
    assert ts_sessions == ses_sessions, \
        "Mismatch in unique SESSION_IDs across DataFrames"