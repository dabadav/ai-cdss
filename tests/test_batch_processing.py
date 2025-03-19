import pytest
import pandas as pd
import numpy as np

@pytest.mark.parametrize(
    "synthetic_session_data",
    [
        # Test case with 3 patients, 2 protocols, 5 sessions per patient
        {"num_patients": 3, "num_protocols": 2, "num_sessions": 5, "columns_with_nulls": []},
    ],
    indirect=True  # This ensures parameters are passed to the fixture
)
def test_session_data_integrity(synthetic_session_data):
    """Test that generated session data conforms to expected structure."""
    session_df, session_ids = synthetic_session_data  # Unpack the fixture return

    # Ensure data is not empty
    assert len(session_df) > 0, "Session data should not be empty."

    # Check required columns exist
    required_columns = [
        "PATIENT_ID", "SESSION_ID", "PROTOCOL_ID", "AGE", "STATUS",
        "SESSION_DATE", "REAL_SESSION_DURATION", "ADHERENCE"
    ]
    for col in required_columns:
        assert col in session_df.columns, f"Missing column: {col}"

    # Ensure adherence is within the expected range (0 to 1)
    assert session_df["ADHERENCE"].between(0, 1).all(), "Adherence values should be between 0 and 1."

    print(f"Successfully validated {len(session_df)} session records.")
