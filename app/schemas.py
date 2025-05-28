# schemas.py
from pydantic import BaseModel, RootModel, Field
from typing import List, Optional, Dict
from enum import Enum

class RGSMode(str, Enum):
    app = "app"
    plus = "plus"

# class RecommendationRequest(BaseModel):
#     patient_list: List[int] = Field(..., example=[775])
#     # rgs_mode: Optional[str] = Field(None, example="app")
#     weights: Optional[List[int]] = Field(None, example=[1, 1, 1])
#     alpha: Optional[float] = Field(None, example=0.5)
#     n: Optional[int] = Field(None, example=12)
#     days: Optional[int] = Field(None, example=7)
#     protocols_per_day: Optional[int] = Field(None, example=5)

class RecommendationRequest(BaseModel):
    # input
    study_id: List[int] = Field(..., example=[12])
    
    # optional params
    weights: Optional[List[int]] = Field(None, example=[1, 1, 1])
    alpha: Optional[float] = Field(None, example=0.5)
    n: Optional[int] = Field(None, example=12) # Diversity
    days: Optional[int] = Field(None, example=7) # Num days
    protocols_per_day: Optional[int] = Field(None, example=5) # Intensity

class RecommendationOut(BaseModel):
    patient_id: int = Field(alias="PATIENT_ID")
    protocol_id: int = Field(alias="PROTOCOL_ID")
    ppf: float = Field(alias="PPF")
    adherence: float = Field(alias="ADHERENCE")
    dm_value: float = Field(alias="DM_VALUE")
    pe_value: float = Field(alias="PE_VALUE")
    usage: int = Field(alias="USAGE")
    contrib: List[float] = Field(alias="CONTRIB")
    days: List[int] = Field(alias="DAYS")
    score: float = Field(alias="SCORE")
    explanation: List[str] = Field(alias="EXPLANATION")

class RecommendationsResponse(RootModel):
    root: Dict[int, List[RecommendationOut]]

class RecsysMetricsRow(BaseModel):
    dm_value: float = Field(alias="DM_VALUE")
    adherence: float = Field(alias="ADHERENCE")
    ppf: float = Field(alias="PPF")
    contrib: List[float] = Field(alias="CONTRIB")

class PrescriptionStagingRow(BaseModel):
    patient_id: int = Field(alias="PATIENT_ID")
    protocol_id: int = Field(alias="PROTOCOL_ID")
    day: int = Field(alias="DAY")
