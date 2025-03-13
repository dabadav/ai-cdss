from ai_cdss.services.pipeline import PipelineBase


def main():
    
    PATIENT_LIST = [
        775,  787,  788, 1123, 1169, 1170, 1171, 1172, 1173, 1983, 2110, 2195,
        2955, 2956, 2957, 2958, 2959, 2960, 2961, 2962, 2963, 3081, 3229, 3318, 3432
    ]

    pipeline = PipelineBase(
        PATIENT_LIST
    )
    pipeline.run()

if __name__ == "__main__":
    main()