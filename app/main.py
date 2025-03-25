from fastapi import FastAPI, Depends
from typing import List
from .config import Settings
from .dependencies import get_settings
from .schemas import RecommendationRequest, RecommendationsResponse, RecommendationOut
from ai_cdss.cdss import CDSS
from ai_cdss.data_loader import DataLoader
from ai_cdss.data_processor import DataProcessor
from ai_cdss.ppf import load_patient_subscales

app = FastAPI()

def get_top_contributing_features(values: List[float], keys: List[str], top_n: int = 3) -> List[str]:
    if len(values) != len(keys):
        raise ValueError("Length of values and keys must match.")
    return [k for k, v in sorted(zip(keys, values), key=lambda x: x[1], reverse=True)[:top_n]]

@app.post("/recommend", response_model=RecommendationsResponse)
def recommend(
    request: RecommendationRequest,
    settings: Settings = Depends(get_settings)
):
    rgs_mode = request.rgs_mode or settings.RGS_MODE
    weights = request.weights or settings.WEIGHTS
    alpha = request.alpha if request.alpha is not None else settings.ALPHA
    n = request.n or settings.N
    days = request.days or settings.DAYS
    protocols_per_day = request.protocols_per_day or settings.PROTOCOLS_PER_DAY

    loader = DataLoader(rgs_mode=rgs_mode)
    processor = DataProcessor(weights=weights, alpha=alpha)

    session = loader.load_session_data(patient_list=request.patient_list)
    timeseries = loader.load_timeseries_data(patient_list=request.patient_list)
    ppf = loader.load_ppf_data(patient_list=request.patient_list)
    protocol_similarity = loader.load_protocol_similarity()
    scores = processor.process_data(session, timeseries, ppf)
    protocol_attributes = load_patient_subscales()
    attribute_keys = list(protocol_attributes.columns)
    
    cdss = CDSS(scoring=scores, n=n, days=days, protocols_per_day=protocols_per_day)

    return RecommendationsResponse(
        root={
            patient: [
                RecommendationOut(
                    **row,
                    top_features=get_top_contributing_features(row["CONTRIB"], attribute_keys)
                )
                for row in cdss.recommend(patient, protocol_similarity).to_dict(orient="records")
            ]
            for patient in request.patient_list
        }
    )


