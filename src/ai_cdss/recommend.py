from ai_cdss.services.pipeline import PipelineBase


def main():
    
    PATIENT_LIST = [
        775,  787,  788, 1123, 1169, 1170, 1171, 1172, 1173, 1983, 2110, 2195,
        2955, 2956, 2957, 2958, 2959, 2960, 2961, 2962, 2963, 3081, 3229, 3318, 3432
    ]

    latent_to_clinical_mapping_nest = {
        # Functional Independence
        "BARTHEL": ["DAILY_LIVING_ACTIVITY"],  # Barthel Index measures independence in ADLs.

        # Motor Function (Spasticity & Strength)
        "ASH_PROXIMAL": ["BODY_PART_ARM", "BODY_PART_SHOULDER", "COORDINATION"],  # Ashworth scale for proximal limb spasticity.
        "MA_DISTAL": ["BODY_PART_FINGER", "BODY_PART_WRIST", "GRASPING", "PINCHING"],  # Motor Assessment for distal motor function.

        # Fatigue & Pain
        "FATIGUE": ["DIFFICULTY_COGNITIVE", "DIFFICULTY_MOTOR", "PROCESSING_SPEED", "ATTENTION"],  # Fatigue relates to cognitive/motor difficulty.
        "VAS": ["DIFFICULTY_COGNITIVE", "DIFFICULTY_MOTOR"],  # Visual Analog Scale (VAS) for perceived effort.

        # Fugl-Meyer Subscales (Motor Control & Coordination)
        "FM_A": ["BODY_PART_ARM", "BODY_PART_SHOULDER", "RANGE_OF_MOTION_H", "RANGE_OF_MOTION_V"],  # Upper Limb Motor
        "FM_B": ["BODY_PART_WRIST", "PRONATION_SUPINATION", "RANGE_OF_MOTION_H"],  # Wrist Motor
        "FM_C": ["BODY_PART_FINGER", "GRASPING", "PINCHING"],  # Hand Motor
        "FM_D": ["COORDINATION", "RANGE_OF_MOTION_H", "RANGE_OF_MOTION_V"],  # Coordination & Speed
        "FM_TOTAL": ["BODY_PART_ARM", "BODY_PART_WRIST", "BODY_PART_FINGER", "COORDINATION"],  # Full Upper Limb Score

        # Activity & Movement Quality
        "ACT_AU": ["BODY_PART_TRUNK"],  # Activity Autonomy linked to balance.
        "ACT_QOM": ["COORDINATION"],  # Quality of Movement related to balance & coordination.
    }

    pipeline = PipelineBase(
        PATIENT_LIST, 
        clinical_score_path="../../data/clinical_scores.csv", 
        protocol_csv_path="../../data/protocol_attributes.csv",
        mapping_dict=latent_to_clinical_mapping_nest
    )
    pipeline.run()

if __name__ == "__main__":
    main()