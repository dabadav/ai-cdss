# tests/test_pipeline.py
import pytest
import pandas as pd
from ai_cdss.cdss import CDSS

def test_pipeline_recommend():
    """Test that pipeline.recommend() returns a DataFrame with the correct number of rows."""
    pipeline = CDSS([775], n=3)  # Single patient, expecting 3 recommendations
    recommendations = pipeline.recommend()

    # Ensure the result is a DataFrame
    assert isinstance(recommendations, pd.DataFrame), "recommend() should return a DataFrame"
    
    # Ensure it has exactly `n` rows
    assert len(recommendations) == 3, f"Expected 3 recommendations, got {len(recommendations)}"

def test_pipeline_recommend_no_sessions():
    """Test that pipeline.recommend() returns a DataFrame with the correct number of rows."""
    pipeline = CDSS([103], n=3)  # Single patient, expecting 3 recommendations
    recommendations = pipeline.recommend()

    # Ensure the result is a DataFrame
    assert isinstance(recommendations, pd.DataFrame), "recommend() should return a DataFrame"
    
    # Ensure it has exactly `n` rows
    assert len(recommendations) == 3, f"Expected 3 recommendations, got {len(recommendations)}"

def test_pipeline_update_prescriptions():
    """Test that pipeline.update_prescriptions() modifies prescriptions correctly."""
    cdss = CDSS([775], n=3)

    # Step 1: Get initial prescriptions
    initial_recommendations = cdss.recommend()
    
    # Ensure it's a DataFrame
    assert isinstance(initial_recommendations, pd.DataFrame), "recommend() should return a DataFrame"
    assert not initial_recommendations.empty, "Initial recommendations should not be empty"

    # Step 2: Update prescriptions
    cdss.update_prescriptions(cdss.prescriptions, cdss.scoring)
    
    # Step 3: Get new prescriptions
    updated_recommendations = cdss.recommend()

    # Ensure `update_prescriptions()` changed something
    assert not updated_recommendations.equals(initial_recommendations), "Prescriptions should be updated"
    
    # Ensure it remains a DataFrame and keeps `n` rows
    assert isinstance(updated_recommendations, pd.DataFrame), "Updated recommendations should still be a DataFrame"
    assert len(updated_recommendations) == 3, f"Expected 3 rows after update, got {len(updated_recommendations)}"

