from ai_cdss.processing import ClinicalSubscales, ProtocolToClinicalMapper, compute_ppf, merge_data, compute_protocol_similarity
from pathlib import Path
import pandas as pd

def main():
    
    # Get platform-appropriate application data directory
    output_dir = Path.home() / ".ai_cdss" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    patient = pd.read_csv("../data/clinical_scores.csv", index_col=0)
    protocol = pd.read_csv("../data/protocol_attributes.csv", index_col=0)

    patient_deficiency = ClinicalSubscales().compute_deficit_matrix(patient)
    protocol_mapped    = ProtocolToClinicalMapper().map_protocol_features(protocol)

    ppf, contrib = compute_ppf(patient_deficiency, protocol_mapped)
    ppf_contrib = merge_data(ppf, contrib)
    ppf_contrib.set_index('PATIENT_ID', inplace=True)

    # Save to CSV in versioned output directory
    output_path = output_dir / "ppf.parquet"
    ppf_contrib.to_parquet(output_path)

    protocol_similarity = compute_protocol_similarity(protocol)
    protocol_similarity.to_csv(output_dir / "protocol_fcm.csv")

    print(f"Results saved to: {output_path.absolute()}")

if __name__ == "__main__":

    PATIENT_LIST = [
        775,  787,  788, 1123, 1169, 1170, 1171, 1172, 1173, 1983, 2110, 2195,
        2955, 2956, 2957, 2958, 2959, 2960, 2961, 2962, 2963, 3081, 3229, 3318, 3432
    ]

    main(PATIENT_LIST)