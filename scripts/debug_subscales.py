# %%
from ai_cdss.loaders import DataLoader

loader = DataLoader()
patient_data = loader.load_patient_data([775, 2195])

# %%
import json

import pandas as pd

patient_data = loader.load_patient_data([775, 2195])


def decode_subscales(row, subscales_column="CLINICAL_SCORES", id_column="PATIENT_ID"):
    # Get last evaluation
    data = json.loads(row[subscales_column])[-1]
    # Keep only keys that are subscales (not metadata)
    subscales = {k: v for k, v in data.items() if isinstance(v, dict)}
    # Flatten subscale dictionaries
    flat = pd.json_normalize(subscales).iloc[0]
    # Add patient ID for indexing
    flat[id_column] = row[id_column]
    return flat


decoded = patient_data.data.apply(decode_subscales, axis=1)
decoded = decoded.set_index("PATIENT_ID")
# Now lets convert all subscales to columns with the value as row for each patient

# %%
from ai_cdss.loaders.utils import _load_patient_subscales

_load_patient_subscales()


# %%
from ai_cdss.loaders import DataLoader
from ai_cdss.processing import DataProcessor

loader = DataLoader()
loader.load_patient_subscales([775])

# %%

from ai_cdss.interface import CDSSInterface
from ai_cdss.loaders import DataLoader
from ai_cdss.processing import DataProcessor

loader = DataLoader()
processor = DataProcessor()

cdss_client = CDSSInterface(loader, processor)

cdss_client.compute_patient_fit([12])

# %%
