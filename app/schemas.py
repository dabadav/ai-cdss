# schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class RGSMode(str, Enum):
    app = "app"
    plus = "plus"

class RecommendationRequest(BaseModel):
    # input
    study_id: List[int] = Field(..., example=[12])
    
    # optional params
    weights: Optional[List[int]] = Field(None, example=[1, 1, 1])
    alpha: Optional[float] = Field(None, example=0.5)
    n: Optional[int] = Field(None, example=12) # Diversity
    days: Optional[int] = Field(None, example=7) # Num days
    protocols_per_day: Optional[int] = Field(None, example=5) # Intensity
