# tests/conftest.py
import pytest
from tests.fixtures.synthetic_data import synthetic_session_data, synthetic_timeseries_data

@pytest.fixture
def global_session_data(synthetic_session_data):
    return synthetic_session_data

@pytest.fixture
def global_timeseries_data(synthetic_timeseries_data):
    return synthetic_timeseries_data
