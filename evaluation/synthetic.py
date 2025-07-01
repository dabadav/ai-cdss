def generate_synthetic_patient_data(
    shared_ids, base_start="2024-03-01", base_end="2024-04-30"
):
    """
    Generate synthetic patient-level data including clinical trial start and end dates.
    shared_ids: list of (patient_id, protocol_id, session_id) tuples
    Returns a DataFrame with columns: PATIENT_ID, CLINICAL_START, CLINICAL_END.
    """
    import pandas as pd

    base_start = pd.to_datetime(base_start)
    base_end = pd.to_datetime(base_end)
    patient_ids = sorted(set(pid for pid, _, _ in shared_ids))
    data = []
    for i, patient_id in enumerate(patient_ids):
        # Optionally stagger start/end dates by patient_id for variety
        start = base_start + pd.Timedelta(days=i * 2)
        end = base_end + pd.Timedelta(days=i * 2)
        data.append(
            {
                PATIENT_ID: patient_id,
                CLINICAL_START: start,
                CLINICAL_END: end,
            }
        )
    return pd.DataFrame(data)
