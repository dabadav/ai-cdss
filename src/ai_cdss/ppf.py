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

from pathlib import Path
import pandas as pd
from ai_cdss.processing import ClinicalSubscales, ProtocolToClinicalMapper, compute_ppf, compute_protocol_similarity
import shutil

# Default data directory
DEFAULT_DATA_DIR = Path.home() / ".ai_cdss" / "data"
DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)

def safe_load_csv(file_path: str = None, default_filename: str = None) -> pd.DataFrame:
    """
    Safely loads a CSV file, either from a given file path or from the default data directory.

    Parameters:
        file_path (str, optional): Full path to the CSV file. If not provided, `default_filename` is used.
        default_filename (str, optional): Name of the file in the default directory.

    Returns:
        pd.DataFrame: Loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be read as a valid CSV.
    """
    file_path = Path(file_path) if file_path else DEFAULT_DATA_DIR / default_filename

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}. Ensure the correct path is specified.")

    try:
        df = pd.read_csv(file_path, index_col=0)
        
        # If the file was loaded from outside the default directory, save a copy
        default_file_path = DEFAULT_DATA_DIR / file_path.name
        
        if file_path.parent != DEFAULT_DATA_DIR:
            shutil.copy(file_path, default_file_path)
            print(f"File copied to default directory: {default_file_path}")

        return df
    
    except Exception as e:
        raise ValueError(f"Error reading {file_path}: {e}")

def load_patient_subscales(file_path: str = None) -> pd.DataFrame:
    """Load patient clinical subscale scores from a given file or the default directory."""
    return safe_load_csv(file_path, "clinical_scores.csv")

def load_protocol_attributes(file_path: str = None) -> pd.DataFrame:
    """Load protocol attributes from a given file or the default directory."""
    return safe_load_csv(file_path, "protocol_attributes.csv")

def main():
    
    # Get platform-appropriate application data directory
    output_dir = Path.home() / ".ai_cdss" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    patient = load_patient_subscales()
    protocol = load_protocol_attributes()

    patient_deficiency = ClinicalSubscales().compute_deficit_matrix(patient)
    protocol_mapped    = ProtocolToClinicalMapper().map_protocol_features(protocol)

    ppf, contrib = compute_ppf(patient_deficiency, protocol_mapped)
    ppf_contrib = pd.merge(ppf, contrib, on=["PATIENT_ID", "PROTOCOL_ID"], how="left")
    ppf_contrib.set_index('PATIENT_ID', inplace=True)

    # Save Contrib Subscales as metadata
    ppf_contrib.attrs = {"SUBSCALES": list(protocol_mapped.columns)}

    # Save to CSV in versioned output directory
    output_path = output_dir / "ppf.parquet"
    ppf_contrib.to_parquet(output_path)

    protocol_similarity = compute_protocol_similarity(protocol)
    protocol_similarity.to_csv(output_dir / "protocol_similarity.csv")

    print(f"Results saved to: {output_path.absolute()}")

if __name__ == "__main__":

    main()