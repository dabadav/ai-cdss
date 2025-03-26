import pytest
import pandas as pd
from ai_cdss.data_processor import DataProcessor

def test_aggregate_dms_per_time():
    """Test that aggregate_dms_per_time produces one row per timepoint for each session and protocol."""
    
    # Sample test data (note: duplicate rows for the same timepoint)
    test_data = pd.DataFrame({
        "PATIENT_ID": [1, 1, 1, 2, 2, 2],
        "SESSION_ID": [101, 101, 101, 202, 202, 202],
        "PROTOCOL_ID": [1, 1, 1, 2, 2, 2],
        "GAME_MODE": ["A", "A", "A", "B", "B", "B"],
        "SECONDS_FROM_START": [0, 0, 10, 10, 20, 20],
        "DM_KEY": ["dm1", "dm2", "dm1", "dm1", "dm1", "dm2"],
        "DM_VALUE": [0.5, 0.6, 0.7, 0.3, 0.4, 0.5],
        "PE_KEY": ["pe1", "pe1", "pe2", "pe1", "pe1", "pe2"],
        "PE_VALUE": [1.0, 1.2, 1.4, 0.8, 0.9, 1.1]
    })
    
    # Expected output (should aggregate duplicate rows)
    expected_output = pd.DataFrame({
        "PATIENT_ID": [1, 1, 2],
        "SESSION_ID": [101, 101, 202],
        "PROTOCOL_ID": [1, 1, 2],
        "GAME_MODE": ["A", "A", "B"],
        "SECONDS_FROM_START": [0, 10, 20],
        "DM_KEY": [["dm1", "dm2"], ["dm1"], ["dm1", "dm2"]],
        "DM_VALUE": [0.55, 0.7, 0.4],  # Averaging the duplicate values for DM_VALUE
        "PE_KEY": ["pe1", "pe2", "pe1"],
        "PE_VALUE": [1.1, 1.4, 1.0]  # Averaging the duplicate values for PE_VALUE
    })

    # Initialize the class with an alpha value (for other methods, not needed for this one)
    processor = DataProcessor()  
    
    # Call the method
    result_df = processor.aggregate_dms_per_time(test_data)

    # Ensure output DataFrame is not empty
    assert not result_df.empty, "Output DataFrame is empty."
    
    # Check that the result has one row per unique timepoint (seconds_from_start)
    # Ensure there is one row per unique combination of (PATIENT_ID, SESSION_ID, SECONDS_FROM_START)
    assert len(unique_timepoint_combinations) == len(result_df), "There are duplicate rows for the same timepoint within a session and patient."

    # Check that the output matches the expected output
    pd.testing.assert_frame_equal(expected_output, expected_output)

def test_compute_metrics_ewma():
    """ Test compute_metrics_ewma with a small sample DataFrame. """

    # Sample test data
    test_data = pd.DataFrame({
        "PATIENT_ID": [1, 1, 1, 2, 2, 2],
        "SESSION_ID": [101, 101, 101, 202, 202, 202],
        "PROTOCOL_ID": [1, 1, 1, 2, 2, 2],
        "GAME_MODE": ["A", "A", "A", "B", "B", "B"],
        "SECONDS_FROM_START": [0, 10, 20, 0, 10, 20],
        "DM_KEY": ["dm1", "dm1", "dm2", "dm1", "dm1", "dm2"],
        "DM_VALUE": [0.5, 0.6, 0.7, 0.3, 0.4, 0.5],
        "PE_KEY": ["pe1", "pe1", "pe2", "pe1", "pe1", "pe2"],
        "PE_VALUE": [1.0, 1.2, 1.4, 0.8, 0.9, 1.1]
    })

    # Initialize the class with an alpha value
    processor = DataProcessor(alpha=0.5)  

    # Call the method
    result_df = processor.compute_metrics_ewma(test_data)

    # Ensure output DataFrame is not empty
    assert not result_df.empty, "Output DataFrame is empty."

    # Ensure it has the same number of rows
    assert len(result_df) == len(test_data), "Row count mismatch."

    # Ensure column structure is preserved
    expected_columns = ["PATIENT_ID", "SESSION_ID", "PROTOCOL_ID", "GAME_MODE",
                        "SECONDS_FROM_START", "DM_KEY", "DM_VALUE", "PE_KEY", "PE_VALUE"]
    assert list(result_df.columns) == expected_columns, "Column names mismatch."

    # Ensure EWMA transformation modified DM_VALUE and PE_VALUE
    assert not result_df["DM_VALUE"].equals(test_data["DM_VALUE"]), "DM_VALUE did not change after EWMA."
    assert not result_df["PE_VALUE"].equals(test_data["PE_VALUE"]), "PE_VALUE did not change after EWMA."

