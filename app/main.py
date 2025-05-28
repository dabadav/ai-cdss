from fastapi import FastAPI, Depends
from typing import List
from config import Settings
from dependencies import get_settings
from schemas import RecommendationRequest, RecommendationsResponse, RecommendationOut, RGSMode
from ai_cdss.cdss import CDSS
from ai_cdss.data_loader import DataLoader
from ai_cdss.data_processor import DataProcessor

app = FastAPI(
    title="AI-CDSS API",
    description="Clinical Decision Support System (CDSS) for personalized rehabilitation protocol recommendations.",
    version="1.0.0",
    contact={
        "name": "Eodyne Systems",
        "email": "contact@eodyne.com"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    }
)

def get_top_contributing_features(values: List[float], keys: List[str], top_n: int = 3) -> List[str]:
    if len(values) != len(keys):
        raise ValueError("Length of values and keys must match.")
    return [k for k, v in sorted(zip(keys, values), key=lambda x: x[1], reverse=True)[:top_n]]

@app.post(
    "/recommend/{rgs_mode}", 
    response_model=RecommendationsResponse,
    summary="Get personalized rehabilitation recommendations",
    description="""
    Generate a list of protocol recommendations for each patient in the request.
    Recommendations are based on patient profiles, time series data, and computed protocol suitability.
    Each recommendation includes a computed PPF score, adherence values, usage history, 
    and an explanation field identifying the top contributing clinical subscales.
    """,
    tags=["Recommendations"]
    )
def recommend(
    request: RecommendationRequest,
    rgs_mode: RGSMode = RGSMode.app,
    settings: Settings = Depends(get_settings),
):
    # params
    weights = request.weights or settings.WEIGHTS
    alpha = request.alpha if request.alpha is not None else settings.ALPHA
    n = request.n or settings.N
    days = request.days or settings.DAYS
    protocols_per_day = request.protocols_per_day or settings.PROTOCOLS_PER_DAY

    # loading / processing code
    loader = DataLoader(rgs_mode=rgs_mode.value)
    processor = DataProcessor(weights=weights, alpha=alpha)

    # study_id -> patient_list
    patient_list = None # retrieve patient_list from patient request.study_id or refactor loader class

    # ** LOADING ERROR HANDLING ** #
    session = loader.load_session_data(patient_list=patient_list)
    timeseries = loader.load_timeseries_data(patient_list=patient_list)
    ppf = loader.load_ppf_data(patient_list=patient_list)
    protocol_similarity = loader.load_protocol_similarity()
    
    # ** PROCESSING ERROR HANDLING ** #
    scores = processor.process_data(session, timeseries, ppf, None) # SessionSchema, TimeseriesSchema, PPFSchema -> ScoringSchema
    
    # business logic
    # ** BUSINESS LOGIC ERROR HANDLING ** #
    cdss = CDSS(scoring=scores, n=n, days=days, protocols_per_day=protocols_per_day)

    # for patient in patient_list:
    #     for row in cdss.recommend(patient, protocol_similarity).to_dict(orient="records")
    #         EXPLANATION=get_top_contributing_features(row["CONTRIB"], scores.attrs.get("SUBSCALES"))
    #         explode days
    #         cast to recsys metrics and prescription_staging
    ######### write operations

    # return interface
    # return RecommendationsResponse(
    #     root={
    #         patient: [
    #             RecommendationOut(
    #                 **row,
    #                 EXPLANATION=get_top_contributing_features(row["CONTRIB"], scores.attrs.get("SUBSCALES"))
    #             )
    #             for row in cdss.recommend(patient, protocol_similarity).to_dict(orient="records")
    #         ]
    #         for patient in request.patient_list
    #     }
    # )


###
### Chron -> patient info / study -> fetch_data -> process data -> write data
###