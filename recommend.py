from ai_cdss.cdss import CDSS
from ai_cdss.data_loader import DataLoader
from ai_cdss.data_processor import DataProcessor

def main():
    
    PATIENT_LIST = [
        775,  787,  788, 1123, 1169, 1170, 1171, 1172, 1173, 1983, 2110, 2195,
        2955, 2956, 2957, 2958, 2959, 2960, 2961, 2962, 2963, 3081, 3229, 3318, 3432
    ]

    # Parameters
    rgs_mode = "app"
    weights = [1,1,1]
    alpha = 0.5

    n = 12
    days = 7
    protocols_per_day = 5

    # Services
    loader = DataLoader(
        rgs_mode = rgs_mode
    )
    processor = DataProcessor(
        weights=weights,
        alpha=alpha
    )

    # Execution
    session = loader.load_session_data(patient_list=PATIENT_LIST)
    timeseries = loader.load_timeseries_data(patient_list=PATIENT_LIST)
    ppf = loader.load_ppf_data(patient_list=PATIENT_LIST)
    protocol_similarity = loader.load_protocol_similarity()

    scores = processor.process_data(session_data=session, timeseries_data=timeseries, ppf_data=ppf)
    
    # CDSS
    cdss = CDSS(
        scoring=scores,
        n=n,
        days=days,
        protocols_per_day=protocols_per_day
    )
    
    # Results
    for patient in PATIENT_LIST:
        cdss.recommend(patient_id=patient, protocol_similarity=protocol_similarity)

if __name__ == "__main__":
    main()