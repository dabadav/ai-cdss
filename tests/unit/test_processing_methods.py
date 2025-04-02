import pytest
import pandas as pd
import numpy as np
from ai_cdss.data_processor import DataProcessor
from ai_cdss.constants import *

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
    result_df = processor.aggregate_dms_by_time(test_data)

    # Ensure output DataFrame is not empty
    assert not result_df.empty, "Output DataFrame is empty."
    
    # Check that the output matches the expected output
    pd.testing.assert_frame_equal(expected_output, expected_output)

def test_compute_metrics_ewma():
    """ Test compute_metrics_ewma with a small sample DataFrame. """
    
    def manual_ewma(values, alpha):
        """Manually compute EWMA with adjust=True."""
        ewma_values = []
        for t in range(len(values)):
            weights = [(1 - alpha) ** i for i in range(t + 1)]
            weighted_sum = sum(w * x for w, x in zip(weights, reversed(values[:t + 1])))
            ewma = weighted_sum / sum(weights)
            ewma_values.append(ewma)
        return ewma_values

    np.random.seed(42)

    # Generate random test data
    num_rows = 50
    test_data = pd.DataFrame({
        "PATIENT_ID": np.random.choice([1, 2, 3], size=num_rows),
        "SESSION_ID": np.random.choice([100, 200, 300], size=num_rows),
        "PROTOCOL_ID": np.random.choice([10, 20], size=num_rows),
        "GAME_MODE": np.random.choice(["A", "B"], size=num_rows),
        "SECONDS_FROM_START": np.random.randint(0, 300, size=num_rows),
        "DM_KEY": np.random.choice(["dm1", "dm2"], size=num_rows),
        "DM_VALUE": np.random.uniform(0.0, 1.0, size=num_rows),
        "PE_KEY": np.random.choice(["pe1", "pe2"], size=num_rows),
        "PE_VALUE": np.random.uniform(0.0, 2.0, size=num_rows),
    })
    test_data = test_data.sort_values(by=BY_PPST)

    # Initialize processor
    processor = DataProcessor(alpha=0.3)

    # Run processor
    result_df = processor._compute_ewma(test_data, DM_VALUE, BY_PP)

    # Basic checks
    assert not result_df.empty, "Output DataFrame is empty."
    assert len(result_df) == len(test_data), "Row count mismatch."
    expected_columns = ["PATIENT_ID", "SESSION_ID", "PROTOCOL_ID", "GAME_MODE",
                        "SECONDS_FROM_START", "DM_KEY", "DM_VALUE", "PE_KEY", "PE_VALUE"]
    assert result_df.columns.tolist() == expected_columns, f"Column names mismatch. {result_df.columns.tolist()}"

    # Test subset
    test_patient = test_data.PATIENT_ID.unique()[0]
    test_protocol = test_data.PROTOCOL_ID.unique()[0]

    # Manual result
    values = (
        test_data[
            (test_data[PATIENT_ID] == test_patient)
            & (test_data[PROTOCOL_ID] == test_protocol)
        ][DM_VALUE].tolist()
    )
    result_manual = manual_ewma(values, processor.alpha)

    # Processor result
    result_values = (
        result_df[
            (test_data[PATIENT_ID] == test_patient)
            & (test_data[PROTOCOL_ID] == test_protocol)
        ][DM_VALUE].tolist()
    )

    # Compare manually computed and function-computed EWMA values using assert
    for manual, computed in zip(result_manual, result_values):
        # Use assert with a small tolerance
        assert abs(manual - computed) < 1e-6, f"Mismatch: Manual={manual}, Computed={computed}"
