# %%
import pandas as pd
from typing import List
import pandera as pa

# Assuming `ScoringSchema` is already defined as in your code
from ai_cdss.models import ScoringSchema
from ai_cdss.data_loader import DataLoader

# %%
loader = DataLoader()
scores = loader.load_patient_clinical_data(patient_list=[204])
# %%

import json
decoded = json.loads(scores.loc[0, "CLINICAL_SCORES"])[-1]
# Input a list of patient IDs to retrieve clinical scores

# Fetches all the rows in db table clinical_trials
# For each row decode the JSON string in the "CLINICAL_SCORES" column
# Each JSON string is a list of dictionaries, where each dictionary contains:
# - "CLINICAL_TRIALS_ID": The ID of the protocol
# - "PATIENT_ID": The adherence score for the protocol
# - "STUDY_ID": The PPF score for the protocol
# - "START_DATE": The date the protocol was started
# - "END_DATE": The date the protocol was ended
# - "CLINICAL_SCORES": A JSON string containing the clinical scores for the protocol

# Clinical scores field consists of the following
"""
[
    {
        'evaluation_date': '2024-09-10',
        'condition': 'pre',
        'MoCA': {
            'Visuospatial/Executive': 4,
            'Naming': 2,
            'Attention': 3,
            'Language': 1,
            'Abstraction': 1,
            'Delayed Recall': 5,
            'Orientation': 1
        },
        'Fugl-Meyer': {
            'FM_A': 14, 
            'FM_B': 17, 
            'FM_C': 5, 
            'FM_D': 0
        }
    }
]
"""
# This format should be generalized for:
# - Multiple undefined clinical assessments per patient
#     - Defined subscales per assessments
# - 
"""
[
    {
        'evalutation': '2024-09-10',
        'condition': 'pre',
        'assessments': [
            {
                'name': 'MoCA',
                'code': '123542-S',
                'subscales': {
                    'Visuospatial/Executive': 4,
                    'Naming': 2,
                    'Attention': 3,
                    'Language': 1,
                    'Abstraction': 1,
                    'Delayed Recall': 5,
                    'Orientation': 1
                }
            },
            {
                'name': 'Fugl-Meyer',
                'code': '123542-S',
                'subscales': {
                    'FM_A': 14, 
                    'FM_B': 17, 
                    'FM_C': 5, 
                    'FM_D': 0
                }
            }
        ]
    }
]

Indexing should be done by patient ID, so that we can easily retrieve the last clinical score for each patient.
[clinical_assessment.subscales for clinical_assessment in json[-1]['assessments']]
"""


# Output should be:
# - Last clinical score for each patient
# - 


# %%
from datetime import date
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, ValidationError, RootModel # Import RootModel

# A general model for the subscales, allowing any string keys to integer values.
class Subscales(BaseModel):
    # This setup allows for any key-value pairs within 'subscales'
    # without explicitly defining every single subscale name.
    # Assumes values are integers, as per your example.
    # For Pydantic V2, you'd typically use 'model_extra' if the schema is truly dynamic and you want to capture it.
    # If the subscales are truly just a dictionary, the __root__ approach with a generic type is still valid.
    # However, if you explicitly want to model it as a dictionary of key:value,
    # and don't expect *other* fields, you can simply define it as a dict.
    # Let's adjust this to a simple dict type hint for direct parsing into dict.
    # Or, if you need validation for specific subscales, define them explicitly.
    # For truly dynamic subscales, your original structure of { "key": value } implies a Dict[str, int].
    pass # If you just want to pass it as a dict, you don't need a model for it.
         # Instead, use Dict[str, int] directly in AssessmentDetail.

# Re-evaluating Subscales:
# If subscales are truly just a dictionary like {"key": value},
# you can use Dict[str, int] directly in AssessmentDetail.
# If you want to validate individual subscale names like "Visuospatial/Executive",
# then you need separate BaseModel for each AssessmentType (MoCA, Fugl-Meyer).
# Given your prompt "defined subscales per assessments", let's assume those are known.
# So, we'll revert to defining specific models for MoCA/Fugl-Meyer subscales
# for explicit validation, and allow for 'other' dynamic assessments.

# MoCA Assessment Subscales
class MoCASubscales(BaseModel):
    Visuospatial_Executive: int = Field(..., alias="Visuospatial/Executive")
    Naming: int
    Attention: int
    Language: int
    Abstraction: int
    Delayed_Recall: int = Field(..., alias="Delayed Recall")
    Orientation: int

# Fugl-Meyer Assessment Subscales
class FuglMeyerSubscales(BaseModel):
    FM_A: int
    FM_B: int
    FM_C: int
    FM_D: int

# A generic assessment detail for *any* assessment, using 'Field' for dynamic 'subscales' content.
# This approach handles known specific assessments (MoCA, Fugl-Meyer)
# AND allows for other *unknown* assessments to be captured generically.
class AssessmentDetail(BaseModel):
    name: str  # e.g., "MoCA", "Fugl-Meyer", "NIHSS"
    code: str
    
    # This field will capture the 'subscales' dictionary.
    # Its type depends on whether the 'name' is known.
    # For simplicity, we can capture it as a generic Dict[str, Any] initially.
    subscales: Dict[str, Any]

    # You can add logic later to process 'subscales' based on 'name'
    # e.g., if self.name == "MoCA", then self.subscales should validate against MoCASubscales.

# Model for a single patient evaluation
class Evaluation(BaseModel):
    evaluation_date: date
    condition: str
    assessments: List[AssessmentDetail]

    # Pydantic V2 equivalent of 'Config.allow_extra = True'
    # This allows other unexpected keys at the Evaluation level, if desired.
    # If your schema is strict and only expects evaluation_date, condition, and assessments,
    # you can remove this.
    model_config = {'extra': 'ignore'} # Or 'allow', 'forbid'


# Root model for the entire JSON structure (a list of evaluations)
class ClinicalData(RootModel[List[Evaluation]]):
    # No __root__ field needed inside the class body for RootModel
    pass

# --- Example Usage (same json_data_format2 as before) ---
json_data_format2 = """
[
    {
        "evaluation_date": "2024-09-10",
        "condition": "pre",
        "assessments": [
            {
                "name": "MoCA",
                "code": "123542-S",
                "subscales": {
                    "Visuospatial/Executive": 4,
                    "Naming": 2,
                    "Attention": 3,
                    "Language": 1,
                    "Abstraction": 1,
                    "Delayed Recall": 5,
                    "Orientation": 1
                }
            },
            {
                "name": "Fugl-Meyer",
                "code": "123542-S",
                "subscales": {
                    "FM_A": 14,
                    "FM_B": 17,
                    "FM_C": 5,
                    "FM_D": 0
                }
            },
            {
                "name": "NIHSS",
                "code": "555-XYZ",
                "subscales": {
                    "LevelOfConsciousness": 0,
                    "BestGaze": 1,
                    "Visual": 0
                }
            }
        ]
    },
    {
        "evaluation_date": "2024-11-01",
        "condition": "post",
        "assessments": [
            {
                "name": "MoCA",
                "code": "123542-S",
                "subscales": {
                    "Visuospatial/Executive": 5,
                    "Naming": 3,
                    "Attention": 5,
                    "Language": 2,
                    "Abstraction": 1,
                    "Delayed Recall": 6,
                    "Orientation": 6
                }
            }
        ]
    }
]
"""
import json

try:
    # Parse the JSON string into a Python object
    json_dict = json.loads(json_data_format2)
    clinical_data_parsed = ClinicalData.model_validate(json_dict) # Use model_validate_json for V2

    # Accessing data is now via .root attribute for RootModel
    first_evaluation = clinical_data_parsed.root[0]
    print(f"Evaluation Date: {first_evaluation.evaluation_date}, Condition: {first_evaluation.condition}")

    # Iterate through the assessments in the first evaluation
    for assessment in first_evaluation.assessments:
        print(f"  Assessment Name: {assessment.name}, Code: {assessment.code}")
        # Accessing subscales directly as a dictionary
        print(f"    Subscales: {assessment.subscales}")
        print(f"    Attention score (if MoCA): {assessment.subscales.get('Attention')}")


except ValidationError as e:
    print(f"Validation Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")# %%

# %%
"""
[
    {
        "evaluation_date": "2024-09-10T09:30:00Z", // Full date-time with timezone
        "condition": "pre",
        "results": [
            {
                "name": "MoCA",
                "total_score": 25,
                "subscales": {
                    "Visuospatial/Executive": 4,
                    "Naming": 2,
                    "Attention": 3,
                    "Language": 1,
                    "Abstraction": 1,
                    "Delayed Recall": 5,
                    "Orientation": 1
                }
            },
            {
                "name": "Fugl-Meyer",
                "total_score": 44,
                "subscales": {
                    "FM_A": 14,
                    "FM_B": 17,
                    "FM_C": 5,
                    "FM_D": 0
                }
            }
        ]
    }
]
"""