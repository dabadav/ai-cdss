"""
Script to run PPF and Protocol Similarity computation.
It runs for all patients existing in the clinical scores file.

**Requirements:**

The following input files must be located in `~/.ai_cdss/data/`:

- `clinical_scores.csv`  
  Patient clinical subscales (clinical baseline scores for patients)

- `protocol_attributes.csv`  
  Protocol attributes (matrix of protocols and the domains they target for rehabilitation)

**Outputs:**

Results are saved to `~/.ai_cdss/output/`:

- `ppf.parquet`  
  PPF Matrix in long format (`PATIENT_ID`, `PROTOCOL_ID`, `PPF`, `CONTRIB`)  
  Follows schema: :class:`ai_cdss.models.PPFSchema`

- `protocol_similarity.csv`  
  Protocol similarity matrix in long format (`PROTOCOL_A`, `PROTOCOL_B`, `SIMILARITY`)  
  Follows schema: :class:`ai_cdss.models.PCMSchema`

**How to run:**

Run this script from the command line:

.. code-block:: bash

    python -m ai_cdss.ppf
"""
# %%
from pathlib import Path
import pandas as pd
from ai_cdss.data_loader import DataLoaderLocal
from ai_cdss.data_processor import ClinicalSubscales, ProtocolToClinicalMapper, compute_ppf, compute_protocol_similarity

def main():

    # Get platform-appropriate application data directory
    output_dir = Path.home() / ".ai_cdss" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Loader 
    loader = DataLoaderLocal()
    patient = loader.load_patient_subscales()
    protocol = loader.load_protocol_attributes()

    # patient = load_patient_subscales()
    # protocol = load_protocol_attributes()

    patient_deficiency = ClinicalSubscales().compute_deficit_matrix(patient)
    protocol_mapped    = ProtocolToClinicalMapper().map_protocol_features(protocol)

    ppf, contrib = compute_ppf(patient_deficiency, protocol_mapped)
    ppf_contrib = pd.merge(ppf, contrib, on=["PATIENT_ID", "PROTOCOL_ID"], how="left")
    # Save Contrib Subscales as metadata
    ppf_contrib.attrs = {"SUBSCALES": list(protocol_mapped.columns)}

    # Save to CSV in versioned output directory
    output_path = output_dir / "ppf.parquet"
    ppf_contrib.to_parquet(output_path, index=False)

    print(ppf_contrib)
    print(f"Results saved to: {output_path.absolute()}")

if __name__ == "__main__":

    main()
# %%
