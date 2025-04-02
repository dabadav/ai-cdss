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
    from ai_cdss.data_loader import DataLoader, DataLoaderMock
    from ai_cdss.cdss import CDSS

    # Generate consistent synthetic data
    loader = DataLoaderMock(
        num_patients=2,
        num_protocols=3,
        num_sessions=2
    )

    session_df = loader.load_session_data()
    timeseries_df = loader.load_timeseries_data()
    ppf_df = loader.load_ppf_data(patient_list=[])
    protocol_metrics = loader.load_protocol_init()  

    patient_ids = list(session_df.PATIENT_ID.unique())

    # Process data
    processor = DataProcessor(weights=[1, 1, 1], alpha=0.5)
    scores = processor.process_data(
        session_data=session_df,
        timeseries_data=timeseries_df,
        ppf_data=ppf_df,
        init_data=protocol_metrics
    )

    # Load protocol similarity
    loader = DataLoader(rgs_mode="app")
    protocol_similarity = loader.load_protocol_similarity()

    # Run CDSS
    cdss = CDSS(scoring=scores, n=5, days=7, protocols_per_day=2)

    # Check recommendations
    for pid in patient_ids:
        rec = cdss.recommend(patient_id=pid, protocol_similarity=protocol_similarity)

        # Implement tests on rec
        assert not rec.empty, "Recommendations are empty"