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
import numpy as np
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

def feature_contributions(df_A, df_B):
    # Convert to numpy   
    A = df_A.to_numpy() # (patients, subscales)
    B = df_B.to_numpy() # (protocols, subscales)

    # Compute row-wise norms
    A_norms = np.linalg.norm(A, axis=1, keepdims=True) # (patients, 1)
    B_norms = np.linalg.norm(B, axis=1, keepdims=True) # (protocols, 1)
    
    # Replace zero norms with a small value to avoid NaN (division by zero)
    A_norms[A_norms == 0] = 1e-10
    B_norms[B_norms == 0] = 1e-10

    # Normalize each row to unit vectors
    A_norm = A / A_norms # (patient, subscales)
    B_norm = B / B_norms # (protocol, subscales)

    # Compute feature contributions
    contributions = A_norm[:, np.newaxis, :] * B_norm[np.newaxis, :, :] # (patient, dim, subscales) * (dim, protocol, subscales)

    return contributions # (patients, protocols, subscales_sim)

def compute_ppf(patient_deficiency, protocol_mapped):
    """ Compute the patient-protocol feature matrix (PPF) and feature contributions.
    """
    contributions = feature_contributions(patient_deficiency, protocol_mapped)
    ppf = np.sum(contributions, axis=2) # (patients, protocols, cosine)
    ppf = pd.DataFrame(ppf, index=patient_deficiency.index, columns=protocol_mapped.index)
    contributions = pd.DataFrame(contributions.tolist(), index=patient_deficiency.index, columns=protocol_mapped.index)
    
    ppf_long = ppf.stack().reset_index()
    ppf_long.columns = ["PATIENT_ID", "PROTOCOL_ID", "PPF"]

    contrib_long = contributions.stack().reset_index()
    contrib_long.columns = ["PATIENT_ID", "PROTOCOL_ID", "CONTRIB"]

    return ppf_long, contrib_long

def compute_protocol_similarity(protocol_mapped):
    """ Compute protocol similarity.
    """
    import gower

    protocol_attributes = protocol_mapped.copy()
    protocol_ids = protocol_attributes.PROTOCOL_ID
    protocol_attributes.drop(columns="PROTOCOL_ID", inplace=True)

    hot_encoded_cols = protocol_attributes.columns.str.startswith("BODY_PART")
    weights = np.ones(len(protocol_attributes.columns))
    weights[hot_encoded_cols] = weights[hot_encoded_cols] / hot_encoded_cols.sum()
    protocol_attributes = protocol_attributes.astype(float)

    gower_sim_matrix = gower.gower_matrix(protocol_attributes, weight=weights)
    gower_sim_matrix = pd.DataFrame(1- gower_sim_matrix, index=protocol_ids, columns=protocol_ids)
    gower_sim_matrix.columns.name = "PROTOCOL_SIM"

    gower_sim_matrix = gower_sim_matrix.stack().reset_index()
    gower_sim_matrix.columns = ["PROTOCOL_A", "PROTOCOL_B", "SIMILARITY"]

    return gower_sim_matrix


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