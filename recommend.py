from ai_cdss.cdss import CDSS
from ai_cdss.data_loader import DataLoader
from ai_cdss.data_processor import DataProcessor

def main():
    
    PATIENT_LIST = [
        775,  787,  788, 1123, 1169, 1170, 1171, 1172, 1173, 1983, 2110, 2195,
        2955, 2956, 2957, 2958, 2959, 2960, 2961, 2962, 2963, 3081, 3229, 3318, 3432
    ]

    loader = DataLoader(
        rgs_mode = "app"
    )
    processor = DataProcessor(
        weights=[1,1,1],
        alpha=0.5
    )

    data = loader.load_data(PATIENT_LIST)
    ppf = loader.load_ppf_data(PATIENT_LIST)
    scoring = processor.process(data, ppf)

    pipeline = CDSS(
        PATIENT_LIST,
        n=5
    )
    
    pipeline.recommend()

if __name__ == "__main__":
    main()