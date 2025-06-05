from fastapi import FastAPI, Depends
from contextlib import asynccontextmanager
from typing import List
from config import Settings
from dependencies import get_settings
from schemas import RecommendationRequest, RecommendationOut, RGSMode


import sys
sys.path.append("../src")

from ai_cdss.cdss import CDSS
from ai_cdss.data_loader import DataLoader, DataLoaderMock
from ai_cdss.data_processor import DataProcessor

from rgs_interface.data.interface import DatabaseInterface
from rgs_interface.data.schemas import PrescriptionStagingRow, RecsysMetricsRow

import uuid
import pandas as pd

import logging

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI-CDSS API",
    description="Clinical Decision Support System (CDSS) for personalized rehabilitation protocol recommendations.",
    version="1.0.0",
    # lifespan=lifespan,
    contact={
        "name": "Eodyne Systems",
        "email": "contact@eodyne.com"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)

@app.post(
    "/recommend/{rgs_mode}",
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
    rgs_mode: RGSMode = RGSMode.plus,
    settings: Settings = Depends(get_settings),
):
    # params
    weights = request.weights or settings.WEIGHTS
    alpha = request.alpha if request.alpha is not None else settings.ALPHA
    n = request.n or settings.N
    days = request.days or settings.DAYS
    protocols_per_day = request.protocols_per_day or settings.PROTOCOLS_PER_DAY

    # class instances -> move to lifespan
    loader = DataLoader(rgs_mode=rgs_mode.value)
    processor = DataProcessor(weights=weights, alpha=alpha)

    # study_id -> patient_list
    patient_list = loader.interface.fetch_patients_by_study(study_ids = request.study_id).PATIENT_ID.tolist()
    logger.debug(f"Fetched {len(patient_list)} patients for study ID {request.study_id}")

    # ** LOADING DATA ** #
    session = loader.load_session_data(patient_list=patient_list)
    timeseries = loader.load_timeseries_data(patient_list=patient_list)
    ppf = loader.load_ppf_data(patient_list=patient_list)
    protocol_similarity = loader.load_protocol_similarity()
    
    # ** PROCESSING DATA ** #
    scores = processor.process_data(session, timeseries, ppf, None) # SessionSchema, TimeseriesSchema, PPFSchema -> ScoringSchema

    # ** BUSINESS LOGIC ** #
    cdss = CDSS(scoring=scores, n=n, days=days, protocols_per_day=protocols_per_day)
    unique_id = uuid.uuid4()

    for patient in session.PATIENT_ID.unique():

        recommendations = cdss.recommend(patient, protocol_similarity)
        prescription_df = (
            recommendations
            .explode("DAYS")
            .rename(columns={"DAYS": "WEEKDAY"})
        )
        metrics_df = pd.melt(
            recommendations,
            id_vars=["PATIENT_ID", "PROTOCOL_ID"],
            value_vars=["DELTA_DM", "ADHERENCE_RECENT", "PPF"],
            var_name="KEY",
            value_name="VALUE"
        )

        # ** DB WRITING ** #
        for _, row in prescription_df.iterrows():
            loader.interface.add_prescription_staging_entry(
                PrescriptionStagingRow.from_row(
                    row, 
                    recommendation_id=unique_id,
                    study_id=request.study_id
                )
            )

        for _, row in metrics_df.iterrows():
            loader.interface.add_recsys_metrics_entry(
                RecsysMetricsRow.from_row(
                    row, 
                    recommendation_id=unique_id,
                    study_id=request.study_id
                )
            )

###
### Chron -> patient info / study -> fetch_data -> process data -> write data
###