from ai_cdss.services.data import DataLoader
from ai_cdss.services.processing import ClinicalProcessor, ProtocolProcessor, compute_ppf, merge_data, compute_protocol_similarity
from pathlib import Path

def main(patient_list):
    # Get platform-appropriate application data directory
    output_dir = Path.home() / ".ai_cdss" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_loader = DataLoader(patient_list)

    patient = data_loader.load_patient_data(path="../data/clinical_scores.csv")
    protocol = data_loader.load_protocol_data(path="../data/protocol_attributes.csv")

    patient_deficiency = ClinicalProcessor().process(patient)
    protocol_mapped    = ProtocolProcessor().process(protocol)

    ppf, contrib = compute_ppf(patient_deficiency, protocol_mapped)
    ppf_contrib = merge_data(ppf, contrib)

    ppf_contrib.set_index('PATIENT_ID', inplace=True)

    # Save to CSV in versioned output directory
    output_path = output_dir / "ppf.csv"
    ppf_contrib.to_csv(output_path)

    protocol_similarity = compute_protocol_similarity(protocol)
    protocol_similarity.to_csv(output_dir / "protocol_fcm.csv")

    print(f"Results saved to: {output_path.absolute()}")


if __name__ == "__main__":

    PATIENT_LIST = [
        775,  787,  788, 1123, 1169, 1170, 1171, 1172, 1173, 1983, 2110, 2195,
        2955, 2956, 2957, 2958, 2959, 2960, 2961, 2962, 2963, 3081, 3229, 3318, 3432
    ]

    main(PATIENT_LIST)