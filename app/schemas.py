# schemas.py
from pydantic import BaseModel, RootModel, Field
from typing import List, Optional, Dict

class RecommendationRequest(BaseModel):
    patient_list: List[int]
    rgs_mode: Optional[str] = None
    weights: Optional[List[int]] = None
    alpha: Optional[float] = None
    n: Optional[int] = None
    days: Optional[int] = None
    protocols_per_day: Optional[int] = None

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
