import numpy as np
from collections import Counter

def test_cdss_pipeline_behavior(synthetic_data_factory):
    """
    Full pipeline behavior test:
    - Generates synthetic session + timeseries data
    - Processes scores
    - Runs CDSS recommendation
    - Asserts expected structure and output
    """
    from ai_cdss.data_processor import DataProcessor
    from ai_cdss.data_loader import DataLoader
    from ai_cdss.cdss import CDSS

    # Generate consistent synthetic data
    session_df, timeseries_df, ppf_df = synthetic_data_factory(
        num_patients=2,
        num_sessions=2,
        timepoints=5,
        null_cols_session=[],
        null_cols_timeseries=[]
    )

    patient_ids = list(session_df.PATIENT_ID.unique())

    # Process data
    processor = DataProcessor(weights=[1, 1, 1], alpha=0.5)
    scores = processor.process_data(
        session_data=session_df,
        timeseries_data=timeseries_df,
        ppf_data=ppf_df
    )

    # Load protocol similarity
    loader = DataLoader(rgs_mode="app")
    protocol_similarity = loader.load_protocol_similarity()

    # Run CDSS
    cdss = CDSS(scoring=scores, n=5, days=7, protocols_per_day=2)

    # Check recommendations
    for pid in patient_ids:
        rec = cdss.recommend(patient_id=pid, protocol_similarity=protocol_similarity)

        collapsed_days_list = list(rec['DAYS'].explode())
        # Count how many times each day appears
        day_counts = Counter(collapsed_days_list)
        # All days where prescribed
        assert set(collapsed_days_list) == set(range(7))
        
        # Check if prescriptions per day is 2
        for day in range(7):
            assert day_counts[day] == 2, f"Day {day} has {day_counts[day]} prescriptions (expected 2)"
