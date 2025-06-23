import importlib.resources
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from ai_cdss import config
from ai_cdss.constants import *
from ai_cdss.utils import MultiKeyDict

# ------------------------------
# Clinical Scores


class ClinicalSubscales:
    def __init__(self, scale_yaml_path: Optional[str] = None):
        """Initialize with an optional path to scale.yaml, defaulting to internal package resource."""
        # Retrieves max values for clinical subscales from config/scales.yaml
        if scale_yaml_path:
            self.scales_path = Path(scale_yaml_path)
        else:
            self.scales_path = importlib.resources.files(config) / "scales.yaml"
        if not self.scales_path.exists():
            raise FileNotFoundError(f"Scale YAML file not found at {self.scales_path}")

        # Load scales maximum values
        self.scales_dict = MultiKeyDict.from_yaml(self.scales_path)

    def compute_deficit_matrix(self, patient_df: pd.DataFrame) -> pd.DataFrame:
        """Compute deficit matrix given patient clinical scores."""

        # Retrieve max values using MultiKeyDict
        max_subscales = [
            self.scales_dict.get(scale, None) for scale in patient_df.columns
        ]

        # Check for missing subscale values
        if None in max_subscales:
            missing_subscales = [
                scale
                for scale, max_val in zip(patient_df.columns, max_subscales)
                if max_val is None
            ]
            raise ValueError(f"Missing max values for subscales: {missing_subscales}")

        # Compute deficit matrix
        deficit_matrix = 1 - (
            patient_df / pd.Series(max_subscales, index=patient_df.columns)
        )
        deficit_matrix.rename(self.scales_dict._keys, axis=1, inplace=True)
        return deficit_matrix


# ------------------------------
# Protocol Attributes


class ProtocolToClinicalMapper:
    def __init__(self, mapping_yaml_path: Optional[str] = None):
        """Initialize with an optional path to scale.yaml, defaulting to internal package resource."""
        if mapping_yaml_path:
            self.mapping_path = Path(mapping_yaml_path)
        else:
            self.mapping_path = importlib.resources.files(config) / "mapping.yaml"
        if not self.mapping_path.exists():
            raise FileNotFoundError(f"Scale YAML file not found at {self.mapping_path}")
        # logger.info(f"Loading subscale max values from: {self.scales_path}")
        self.mapping = MultiKeyDict.from_yaml(self.mapping_path)

    def map_protocol_features(
        self, protocol_df: pd.DataFrame, agg_func=np.mean
    ) -> pd.DataFrame:
        """Map protocol-level features into clinical scales using a predefined mapping."""
        # Retrieve max values using MultiKeyDict
        df_clinical = pd.DataFrame(index=protocol_df.index)
        # Collapse using agg_func the protocol latent attributes
        for clinical_scale, features in self.mapping.items():
            df_clinical[clinical_scale] = protocol_df[features].apply(agg_func, axis=1)
        df_clinical.index = protocol_df[PROTOCOL_ID]
        return df_clinical
