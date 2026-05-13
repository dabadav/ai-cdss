import datetime
from datetime import timedelta
import logging
import time
import uuid
from typing import Any, Dict, List, Optional
import json

import pandas as pd
from ai_cdss.recommend import CDSS
from ai_cdss.constants import (
    BY_PP,
    CLINICAL_START,
    DAYS,
    DELTA_DM,
    PATIENT_ID,
    PPF,
    RECENT_ADHERENCE,
    SCORE,
    SESSION_INDEX,
    USAGE,
    USAGE_WEEK,
    WEEKS_SINCE_START,
    DEFAULT_DEBUG_DIR,
    N,
    N_DAYS,
    PROTOCOLS_PER_DAY,
    DEFAULT_LOG_DIR
)
from ai_cdss.loader import DataLoader
from ai_cdss.pipeline import DataPipeline
from ai_cdss.service import (
    PPFService,
    ProtocolSimilarityService,
    RecommendationDataService,
)
from ai_cdss.interface.debug import DebugReport
from ai_cdss.utils import _json_default
from rgs_interface.data.schemas import PrescriptionStagingRow, RecsysMetricsRow

logger = logging.getLogger(__name__)


class CDSSInterface:
    """
    Main orchestrator for generating clinical decision support recommendations.
    Coordinates data preparation, processing, and persistence for study cohorts.
    """

    def __init__(
        self,
        loader: DataLoader,
        pipeline: Optional[DataPipeline] = None,
        data_service: Optional[RecommendationDataService] = None,
        ppf_service: Optional[PPFService] = None,
        debug: bool = False,
    ):
        self.loader = loader
        self.pipeline = pipeline or DataPipeline()
        self.ppf_service = ppf_service or PPFService(loader)
        self.data_service = data_service or RecommendationDataService(loader)
        self.protocol_similarity_service = ProtocolSimilarityService(loader)
        self.debug = debug
        if self.debug:
            self.debug_service = DebugReport(DEFAULT_DEBUG_DIR)

    def recommend_for_patients(
        self,
        patient_ids: List[int],
        n: int = N,
        days: int = N_DAYS,
        protocols_per_day: int = PROTOCOLS_PER_DAY,
        scoring_date: Optional[pd.Timestamp] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Run recommendations for one **or many** patients.
        Returns the same structure as recommend_for_study, with 'per_patient' detailing each patient's result.

        ``force=False`` (default) skips any patient who already has rows in
        ``prescription_staging`` for the week we would be recommending for
        (regardless of STATUS). Set ``force=True`` to rerun anyway.
        """
        return self._recommend_for_patients_core(
            patient_ids,
            n=n,
            days=days,
            protocols_per_day=protocols_per_day,
            scoring_date=scoring_date,
            force=force,
            context={
                "patient_id": patient_ids,
                "message": f"Recommendations generated for patients {patient_ids}"
            },
        )

    def recommend_for_study(
        self,
        study_id: List[int],
        n: int,
        days: int,
        protocols_per_day: int,
        scoring_date: Optional[pd.Timestamp] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Cohort/study run. See ``recommend_for_patients`` for ``force``."""
        # Keep validation where it belongs
        patient_ids = self.loader.fetch_and_validate_patients(study_ids=study_id)
        return self._recommend_for_patients_core(
            patient_ids,
            n=n,
            days=days,
            protocols_per_day=protocols_per_day,
            scoring_date=scoring_date,
            force=force,
            context={"study_id": study_id, "message": f"Recommendations generated for study {study_id}"},
        )

    def _recommend_for_patients_core(
        self,
        patient_ids: List[int],
        *,
        n: int,
        days: int,
        protocols_per_day: int,
        scoring_date: Optional[pd.Timestamp],
        context: Dict[str, Any],
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Internal core that runs the full pipeline for a given set of patient_ids.
        `context` can include study_id or other metadata to echo back in the response.
        """
        logger.info("Starting recommendation generation for patients: %s", patient_ids)
        start_time = time.time()
        unique_id = uuid.uuid4()
        datetime_now = datetime.datetime.now()

        try:
            if not patient_ids:      # catches None or empty list
                elapsed = time.time() - start_time
                payload = {
                    "status": "warning",
                    "run_id": str(unique_id),
                    "patients_processed": 0,
                    "total_recommendations": 0,
                    "per_patient": [],
                    "start_time": datetime_now.isoformat(),
                    "elapsed_seconds": elapsed,
                    **context,
                    "message": (
                        context.get("message")
                        or f"No patients provided or resolved for context={context}"
                    ),
                }
                logger.info("No patients to process. Context: %s | Result: %s", context, payload)
                return payload

            raw_inputs, protocol_similarity = self.data_service.prepare(patient_list=patient_ids)
            scores = self.pipeline.process(raw_inputs, scoring_date or pd.Timestamp.today())
            cdss = CDSS(scoring=scores, n=n, days=days, protocols_per_day=protocols_per_day)

            # Patient start date dict [PATIENT_ID, CLINICAL_START]
            patient_data = raw_inputs.patient
            patient_dict = dict(zip(patient_data[PATIENT_ID], patient_data[CLINICAL_START]))

            patient_results = []
            success_count = 0
            fail_count = 0

            # --------- Per-patient Processing --------
            for p in patient_ids:
                result = self._process_patient(
                    patient=p,
                    cdss=cdss,
                    protocol_similarity=protocol_similarity,
                    scores=scores,
                    unique_id=unique_id,
                    datetime_start=patient_dict[p],
                    scoring_date=(scoring_date or pd.Timestamp.today()),
                    force=force,
                )
                if result.get("status") == "success":
                    success_count += 1
                else:
                    fail_count += 1
                patient_results.append(result)

            if success_count == 0 and fail_count > 0:
                top_status = "failure"
            elif success_count > 0 and fail_count > 0:
                top_status = "partial_success"
            else:
                top_status = "success"

            total_recommendations = sum(r.get("num_recommendations", 0) for r in patient_results)
            elapsed = time.time() - start_time

            payload = {
                "status": top_status,
                "run_id": str(unique_id),
                "patients_processed": len(patient_ids),
                "total_recommendations": total_recommendations,
                "per_patient": patient_results,
                "start_time": datetime_now.isoformat(),
                "elapsed_seconds": elapsed,
                **context,
            }

            if self.debug:
                payload["debug"] = {
                    "scores": {
                        "file": self.debug_service.dump_df(scores, unique_id, "scores.csv"),
                        "preview": self.debug_service.preview_df(scores),
                    }
                }

            # ---- persist payload ----           
            log_path = DEFAULT_LOG_DIR / f"{str(unique_id)}_{datetime_now.date().isoformat()}.json"
            with log_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, default=_json_default, indent=2)
            payload["log_file"] = str(log_path)

            logger.info(
                "Finished recommendation generation. Context: %s | status=%s | "
                "success=%d fail=%d total_recs=%d | Log file: %s",
                context, top_status, success_count, fail_count, total_recommendations, log_path
            )

            return payload

        except Exception as e:

            logger.error(
                "Failed to generate recommendations. Context=%s Error=%s (%s)",
                context, e, type(e).__name__, exc_info=True
            )
            failure_payload = {
                "status": "failure",
                "run_id": str(unique_id),
                "error": f"{type(e).__name__}: {e}",
                "patients_processed": 0,
                "total_recommendations": 0,
                "per_patient": [],
                "start_time": datetime_now.isoformat(),
                **context,
                "message": "Failed to generate recommendations",
            }

            # ---- persist failure payload too ----
            log_path = DEFAULT_LOG_DIR / f"{str(unique_id)}_{datetime_now.date().isoformat()}.json"
            try:
                with log_path.open("w", encoding="utf-8") as f:
                    json.dump(failure_payload, f, default=_json_default, indent=2)
                failure_payload["log_file"] = str(log_path)
            except Exception as log_err:
                logger.error(
                    "Failed to persist failure payload for run_id=%s: %s (%s)",
                    str(unique_id), log_err, type(log_err).__name__, exc_info=True
                )

            return failure_payload

    def _process_patient(
        self,
        patient,
        cdss,
        protocol_similarity,
        scores,
        unique_id,
        datetime_start,
        scoring_date: Optional[pd.Timestamp] = None,
        force: bool = False,
    ):
        """
        Process recommendations and metrics for a single patient.

        Args:
            patient: The patient ID to process.
            cdss: The CDSS instance for generating recommendations.
            protocol_similarity: Protocol similarity data for recommendations.
            scores: DataFrame of all scored protocols.
            unique_id: UUID for this batch run.
            datetime_start: Timestamp for trial start.
            scoring_date: Day from which we derive the patient's current
                week (defaults to today). Used for the duplication guard.
            force: When False (default) we abort early with status
                ``"skipped_already_prescribed"`` if any prescription_staging
                row already exists for ``(patient, week_start)``. When
                True we proceed regardless — useful for replays / forced
                reruns where duplicates are accepted.
        """
        try:
            datetime_now = datetime.datetime.now()
            scoring_ts = scoring_date or pd.Timestamp.today()

            # Idempotency guard — covers two duplication paths:
            # (1) manual rerun on a patient already prescribed this week,
            # (2) cohort fetch picking up a patient before clinical_start
            #     and bootstrapping a fresh RID every cron tick.
            if not force and self._already_prescribed(patient, datetime_start, scoring_ts):
                wk_idx, wk_start = self._current_week_window(datetime_start, scoring_ts)
                logger.info(
                    "Patient %s already has prescription_staging rows for "
                    "week %s (start=%s); skipping. Pass force=True to rerun.",
                    patient, wk_idx, wk_start,
                )
                return {
                    "patient_id":          patient,
                    "num_recommendations": 0,
                    "n_rows":              0,
                    "n_days":              0,
                    "n_protocols":         0,
                    "trace":               None,
                    "skipped_reason":      "already_prescribed",
                    "skipped_week":        wk_idx,
                    "skipped_week_start":  wk_start.isoformat() if wk_start else None,
                    "status":              "skipped",
                }

            result = cdss.recommend(patient, protocol_similarity)
            # result is a RecommendationResult — `.recommendations` is the
            # final DataFrame, `.trace` is the structured audit dict, plus
            # typed views like `.swap_decisions`, `.mvt_mean`, etc.
            prescription_df = self._transform_recommendations(result.recommendations)

            patient_scores = scores[scores[PATIENT_ID] == patient]
            all_metrics_df = self._transform_metrics(patient_scores)

            if not self.debug:
                self._save_prescriptions(prescription_df, unique_id, datetime_start)
                self._save_metrics(all_metrics_df, unique_id, datetime_now)

            # Persist staging shape per patient so under-coverage (e.g. only
            # one weekday produced) is visible in the run log without having
            # to query the DB.
            n_rows      = int(len(prescription_df))
            n_days      = int(prescription_df["WEEKDAY"].nunique()) if "WEEKDAY" in prescription_df.columns and not prescription_df.empty else 0
            n_protocols = int(prescription_df["PROTOCOL_ID"].nunique()) if "PROTOCOL_ID" in prescription_df.columns and not prescription_df.empty else 0

            # Trace is now a first-class field on the result; legacy code
            # that reads `recommendations.attrs["trace"]` still works
            # (RecommendationResult proxies that).
            trace = result.trace

            logger.info(
                "Patient %s shape n_rows=%d n_days=%d n_protocols=%d branch=%s swaps=%d topup=%d",
                patient, n_rows, n_days, n_protocols,
                (trace or {}).get("branch"),
                len((trace or {}).get("swaps") or []),
                len((trace or {}).get("topup") or []),
            )
            for ev in (trace or {}).get("swaps", []) or []:
                logger.info(
                    "  swap: removed=%s (score=%s) -> added=%s (sim=%s) days=%s reason=%s",
                    ev.get("removed"), ev.get("removed_score"),
                    ev.get("added"),   ev.get("similarity"),
                    ev.get("inherited_days"), ev.get("reason"),
                )

            payload = {
                "patient_id": patient,
                "num_recommendations": len(result.recommendations),
                "n_rows": n_rows,
                "n_days": n_days,
                "n_protocols": n_protocols,
                "trace": trace,
                "status": "success",
            }

            if self.debug:
                logger.info("Debug mode enabled - skipping persistence to db for patient %s", patient)
                artifacts = self.debug_service.make_artifacts(
                    run_id=unique_id,
                    scores=scores[scores[PATIENT_ID] == patient],
                    recs=result.recommendations,
                    presc=prescription_df,
                    metrics=all_metrics_df,
                    subdir=f"patient_{patient}",   # creates <base>/<run_id>/patient_<id>/
                    format="csv",
                    preview=False
                )
                payload['debug'] = artifacts

            return payload
        
        except Exception as e:
            logger.exception(
                "Failed to process patient %s: %s (%s)", patient, e, type(e).__name__
            )
            return {
                "patient_id": patient,
                "num_recommendations": 0,
                "status": "failure",
                "error": f"{type(e).__name__}: {e}",
            }

    def _transform_recommendations(self, recommendations: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the recommendations DataFrame into prescription DataFramee.
        Only for recommended protocols.
        """
        prescription_df = recommendations.explode(DAYS).rename(
            columns={DAYS: "WEEKDAY"}
        )
        return prescription_df

    def _transform_metrics(self, scores: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the full scored protocols DataFrame into a metrics DataFrame for all protocols.
        """
        metrics_df = pd.melt(
            scores,
            id_vars=BY_PP,
            value_vars=[
                PPF,
                DELTA_DM,
                RECENT_ADHERENCE,
                SCORE,
                USAGE,
                USAGE_WEEK,
                SESSION_INDEX,
            ],
            var_name="KEY",
            value_name="VALUE",
        )
        return metrics_df

    @staticmethod
    def _current_week_window(
        datetime_start, scoring_ts: pd.Timestamp
    ) -> tuple[int, "datetime.date | None"]:
        """Return ``(week_index, week_start_date)`` for the patient relative
        to the scoring day. ``week_index`` is clamped at 0 — a patient
        whose ``datetime_start`` is in the future is treated as week 0
        (which is exactly the case the duplication guard needs to catch:
        cohort entries ahead of clinical_start)."""
        try:
            start_date = pd.Timestamp(datetime_start).normalize().date()
        except Exception:
            return 0, None
        scoring_day = scoring_ts.normalize().date()
        delta = (scoring_day - start_date).days
        wk_idx = max(0, delta // 7)
        wk_start = start_date + timedelta(days=7 * wk_idx)
        return wk_idx, wk_start

    def _already_prescribed(
        self, patient_id: int, datetime_start, scoring_ts: pd.Timestamp
    ) -> bool:
        """True if ``prescription_staging`` already has any row (any STATUS)
        for ``(patient_id, week_start)``."""
        _, wk_start = self._current_week_window(datetime_start, scoring_ts)
        if wk_start is None:
            return False
        engine = getattr(self.loader.interface, "engine", None)
        if engine is None:
            return False
        sql = (
            "SELECT COUNT(*) AS n FROM prescription_staging "
            "WHERE PATIENT_ID = :pid AND DATE(STARTING_DATE) = :wk"
        )
        try:
            df = self.loader.interface._fetch(
                query=sql, params={"pid": int(patient_id), "wk": wk_start.isoformat()}
            )
            return bool(df is not None and not df.empty and int(df.iloc[0]["n"]) > 0)
        except Exception:
            logger.exception(
                "Duplication check failed for patient %s; treating as not-prescribed.",
                patient_id,
            )
            return False

    def _save_prescriptions(
        self,
        prescription_df: pd.DataFrame,
        unique_id: uuid.UUID,
        datetime_start: datetime.datetime,
    ) -> None:
        """
        Persist prescription data.
        """
        for _, row in prescription_df.iterrows():
            weeks = row[WEEKS_SINCE_START]
            weeks = 0 if pd.isna(weeks) else float(weeks)
            start = (datetime_start + timedelta(weeks=weeks)).date()
            self.loader.interface.add_prescription_staging_entry(
                PrescriptionStagingRow.from_row(
                    row, recommendation_id=unique_id, start=start
                )
            )

    def _save_metrics(
        self,
        metrics_df: pd.DataFrame,
        unique_id: uuid.UUID,
        datetime_now: datetime.datetime,
    ) -> None:
        """
        Persist metrics data.
        """
        for _, row in metrics_df.iterrows():
            self.loader.interface.add_recsys_metric_entry(
                RecsysMetricsRow.from_row(
                    row, recommendation_id=unique_id, metric_date=datetime_now
                )
            )

    def compute_patient_fit(self, patient_id: List[int]) -> dict:
        """
        Compute and persist the Patient-Protocol Fit (PPF) matrix for a single patient.
        Delegates to PPFService.
        """
        ppf_contrib = self.ppf_service.compute_patient_fit(patient_id)
        file_path = self.ppf_service.persist_ppf(ppf_contrib)
        return {
            "message": f"Computation and persistence successful for patient {patient_id}",
            "patient_id": patient_id,
            "subscales_used": list(ppf_contrib.attrs.get("SUBSCALES", [])),
            "saved_to": file_path,
        }

    def compute_protocol_similarity(self) -> dict:
        """
        Compute and persist the protocol similarity matrix using the ProtocolSimilarityService.
        Returns a dict with the file path and a message.
        """
        file_path = (
            self.protocol_similarity_service.compute_and_persist_protocol_similarity()
        )
        return {
            "message": "Protocol similarity computation and persistence successful.",
            "saved_to": file_path,
        }

