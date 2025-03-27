# tests/conftest.py
import pytest
from ai_cdss.evaluation.synthetic import generate_synthetic_data, generate_synthetic_ids

# -- Global Fixtures

@pytest.fixture
def synthetic_data_factory():
    """
    Fixture factory that generates both synthetic session and timeseries data using consistent shared IDs.

    Returns a function that can be called with custom parameters to generate:
    - A validated session DataFrame
    - A validated timeseries DataFrame

    Parameters:
    ----------
    num_patients : int, optional
        Number of unique patients to generate data for (default is 3).
    num_protocols : int, optional
        Number of unique protocols per patient (default is 2).
    num_sessions : int, optional
        Number of sessions per patient-protocol combination (default is 3).
    timepoints : int, optional
        Number of timepoints per session for the timeseries data (default is 10).
    null_cols_session : list of str, optional
        List of column names in session data that should contain null values (default is None).
    null_cols_timeseries : list of str, optional
        List of column names in timeseries data that should contain null values (default is None).
    test_discrepancies : bool, optional
        If True, introduces mismatches in SESSION_ID and PATIENT_ID in the timeseries data (default is False).

    Returns:
    -------
    Callable[[], Tuple[pd.DataFrame, pd.DataFrame]]
        A function that returns two pandas DataFrames: (session_data, timeseries_data),
        both validated against their respective schema and sharing consistent IDs.
    """
    def _factory(
        num_patients=3,
        num_protocols=2,
        num_sessions=3,
        timepoints=10,
        null_cols_session=None,
        null_cols_timeseries=None,
        test_discrepancies=False
    ):
        return generate_synthetic_data(
            num_patients=num_patients,
            num_protocols=num_protocols,
            num_sessions=num_sessions,
            timepoints=timepoints,
            null_cols_session=null_cols_session,
            null_cols_timeseries=null_cols_timeseries,
            test_discrepancies=test_discrepancies
        )
    
    return _factory

@pytest.fixture
def shared_synthetic_ids_factory():
    def _factory(num_patients=3, num_protocols=2, num_sessions=3):
        return generate_synthetic_ids(
            num_patients=num_patients,
            num_protocols=num_protocols,
            num_sessions=num_sessions
        )
    return _factory
