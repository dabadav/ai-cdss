import datetime
from datetime import timedelta
import logging
import time
import uuid
from typing import Any, Dict, List, Optional
import json

import pandas as pd
from ai_cdss.recommender import Recommender
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
from ai_cdss.precompute import (
    compute_ppf_for_patients,
    compute_protocol_similarity_matrix,
    persist_ppf,
    persist_similarity,
)
from ai_cdss.data import (
    CohortRepository,
    PrescriptionStore,
    RGSCohortRepository,
    RGSPrescriptionStore,
)
from ai_cdss.scoring import DataPipeline
from ai_cdss.interface.debug import DebugReport
from ai_cdss.utils import _json_default
from rgs_interface.data.schemas import PrescriptionStagingRow, RecsysMetricsRow

logger = logging.getLogger(__name__)


class RecommendationService:
    """Application orchestrator: cohort fetch → scoring pipeline → engine
    → persistence, for one or many patients.

    Wires three injected collaborators (all default to production
    implementations, all swappable for tests / backtests):

      * `repository` (`CohortRepository`) — reads the input `Cohort`.
      * `pipeline`   (`DataPipeline`)      — Cohort → scored frame.
      * `store`      (`PrescriptionStore`) — idempotency check + writes
                                             the recommendation output.

    The per-patient recommendation itself is delegated to `Recommender`
    (`engine` locally). Note the layering: this class is the *service*;
    `Recommender` is the *engine*. They are distinct — don't conflate.
    """

    def __init__(
        self,
        repository: Optional[CohortRepository] = None,
        pipeline: Optional[DataPipeline] = None,
        store: Optional[PrescriptionStore] = None,
        debug: bool = False,
    ):
        self.repository = repository or RGSCohortRepository()
        self.pipeline = pipeline or DataPipeline()
        # Default store shares the repository's DB interface (one
        # connection); a None interface makes RGSPrescriptionStore open
        # its own. Inject a fake/in-memory store for tests + backtests.
        self.store = store or RGSPrescriptionStore(
            db=getattr(self.repository, "interface", None),
        )
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
        patient_ids = self.repository.fetch_and_validate_patients(study_ids=study_id)
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
        """Run the full pipeline for a set of patient_ids and return the
        run payload. `context` echoes caller metadata (study_id, message)
        back into the response. Never raises — failures are caught and
        returned as a `status="failure"` payload."""
        logger.info("Starting recommendation generation for patients: %s", patient_ids)
        start_time = time.time()
        unique_id = uuid.uuid4()
        datetime_now = datetime.datetime.now()

        if not patient_ids:  # catches None or empty list
            return self._empty_payload(unique_id, datetime_now, start_time, context)

        try:
            cohort = self.repository.find(patient_ids)
            if cohort.missing_ppf:
                raise RuntimeError(
                    f"PPF data missing for patients: {cohort.missing_ppf}. "
                    "Please compute PPF before proceeding."
                )
            scores = self.pipeline.process(cohort, scoring_date or pd.Timestamp.today())
            engine = Recommender(
                scoring=scores, n=n, days=days, protocols_per_day=protocols_per_day,
            )
            patient_results = self._recommend_each(
                patient_ids, engine=engine, cohort=cohort, scores=scores,
                unique_id=unique_id, scoring_date=scoring_date, force=force,
            )
            payload = self._build_run_payload(
                patient_results, unique_id=unique_id, datetime_now=datetime_now,
                start_time=start_time, context=context, scores=scores,
            )
            payload["log_file"] = self._persist_run_log(payload, unique_id, datetime_now)
            logger.info(
                "Finished recommendation generation. Context: %s | status=%s | "
                "total_recs=%d | log=%s",
                context, payload["status"], payload["total_recommendations"],
                payload.get("log_file"),
            )
            return payload

        except Exception as e:
            return self._failure_payload(e, unique_id, datetime_now, context)

    # ==================================================================
    # Run assembly — per-patient loop, payload builders, log persistence.

    def _recommend_each(
        self,
        patient_ids: List[int],
        *,
        engine: Recommender,
        cohort: Any,
        scores: pd.DataFrame,
        unique_id: uuid.UUID,
        scoring_date: Optional[pd.Timestamp],
        force: bool,
    ) -> List[Dict[str, Any]]:
        """Process every patient against the shared engine. Per-patient
        failures are captured inside `_process_patient` (one bad patient
        never aborts the batch)."""
        protocol_similarity = cohort.similarity
        start_dates = dict(zip(cohort.patient[PATIENT_ID], cohort.patient[CLINICAL_START]))
        scoring_ts = scoring_date or pd.Timestamp.today()
        return [
            self._process_patient(
                patient=p,
                engine=engine,
                protocol_similarity=protocol_similarity,
                scores=scores,
                unique_id=unique_id,
                datetime_start=start_dates[p],
                scoring_date=scoring_ts,
                force=force,
            )
            for p in patient_ids
        ]

    @staticmethod
    def _rollup_status(results: List[Dict[str, Any]]) -> str:
        """Batch status from per-patient outcomes. Anything not `"success"`
        (failures + skips) counts against success — matches v0.3.1."""
        success = sum(1 for r in results if r.get("status") == "success")
        fail = len(results) - success
        if success == 0 and fail > 0:
            return "failure"
        if success > 0 and fail > 0:
            return "partial_success"
        return "success"

    def _build_run_payload(
        self,
        patient_results: List[Dict[str, Any]],
        *,
        unique_id: uuid.UUID,
        datetime_now: datetime.datetime,
        start_time: float,
        context: Dict[str, Any],
        scores: pd.DataFrame,
    ) -> Dict[str, Any]:
        payload = {
            "status": self._rollup_status(patient_results),
            "run_id": str(unique_id),
            "patients_processed": len(patient_results),
            "total_recommendations": sum(
                r.get("num_recommendations", 0) for r in patient_results
            ),
            "per_patient": patient_results,
            "start_time": datetime_now.isoformat(),
            "elapsed_seconds": time.time() - start_time,
            **context,
        }
        if self.debug:
            payload["debug"] = {
                "scores": {
                    "file": self.debug_service.dump_df(scores, unique_id, "scores.csv"),
                    "preview": self.debug_service.preview_df(scores),
                }
            }
        return payload

    def _empty_payload(
        self,
        unique_id: uuid.UUID,
        datetime_now: datetime.datetime,
        start_time: float,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = {
            "status": "warning",
            "run_id": str(unique_id),
            "patients_processed": 0,
            "total_recommendations": 0,
            "per_patient": [],
            "start_time": datetime_now.isoformat(),
            "elapsed_seconds": time.time() - start_time,
            **context,
            "message": (
                context.get("message")
                or f"No patients provided or resolved for context={context}"
            ),
        }
        logger.info("No patients to process. Context: %s | Result: %s", context, payload)
        return payload

    def _failure_payload(
        self,
        error: Exception,
        unique_id: uuid.UUID,
        datetime_now: datetime.datetime,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        logger.error(
            "Failed to generate recommendations. Context=%s Error=%s (%s)",
            context, error, type(error).__name__, exc_info=True,
        )
        payload = {
            "status": "failure",
            "run_id": str(unique_id),
            "error": f"{type(error).__name__}: {error}",
            "patients_processed": 0,
            "total_recommendations": 0,
            "per_patient": [],
            "start_time": datetime_now.isoformat(),
            **context,
            "message": "Failed to generate recommendations",
        }
        log_file = self._persist_run_log(payload, unique_id, datetime_now)
        if log_file:
            payload["log_file"] = log_file
        return payload

    @staticmethod
    def _persist_run_log(
        payload: Dict[str, Any],
        unique_id: uuid.UUID,
        datetime_now: datetime.datetime,
    ) -> Optional[str]:
        """Write the run payload to the log dir as JSON. Returns the path,
        or None if persistence itself failed (logged, never raised)."""
        log_path = DEFAULT_LOG_DIR / f"{unique_id}_{datetime_now.date().isoformat()}.json"
        try:
            with log_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, default=_json_default, indent=2)
            return str(log_path)
        except Exception as log_err:
            logger.error(
                "Failed to persist run payload for run_id=%s: %s (%s)",
                str(unique_id), log_err, type(log_err).__name__, exc_info=True,
            )
            return None

    def _process_patient(
        self,
        patient,
        engine,
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
            engine: The Recommender instance for generating recommendations.
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

            result = engine.recommend(patient, protocol_similarity)
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
        """True if the patient already has a prescription for their current
        trial week. The week-window math is domain logic and stays here;
        the persistence query is delegated to the store."""
        _, wk_start = self._current_week_window(datetime_start, scoring_ts)
        if wk_start is None:
            return False
        return self.store.already_prescribed(patient_id, wk_start)

    def _save_prescriptions(
        self,
        prescription_df: pd.DataFrame,
        unique_id: uuid.UUID,
        datetime_start: datetime.datetime,
    ) -> None:
        """Build staging rows from the prescription frame and hand them to
        the store. Each row's STARTING_DATE is the patient's trial start
        plus its WEEKS_SINCE_START offset."""
        rows = []
        for _, row in prescription_df.iterrows():
            weeks = row[WEEKS_SINCE_START]
            weeks = 0 if pd.isna(weeks) else float(weeks)
            start = (datetime_start + timedelta(weeks=weeks)).date()
            rows.append(
                PrescriptionStagingRow.from_row(
                    row, recommendation_id=unique_id, start=start
                )
            )
        self.store.save_prescriptions(rows)

    def _save_metrics(
        self,
        metrics_df: pd.DataFrame,
        unique_id: uuid.UUID,
        datetime_now: datetime.datetime,
    ) -> None:
        """Build metric rows from the metrics frame and hand them to the
        store."""
        rows = [
            RecsysMetricsRow.from_row(
                row, recommendation_id=unique_id, metric_date=datetime_now
            )
            for _, row in metrics_df.iterrows()
        ]
        self.store.save_metrics(rows)

    def compute_patient_fit(self, patient_id: List[int]) -> dict:
        """Compute + persist PPF for one or more patients.

        Delegates to `compute.compute_ppf_for_patients` + `persist_ppf`.
        Requires the repository to expose `patient_subscales` +
        `protocol_attributes` (production-only accessors —
        `RGSCohortRepository` has them).
        """
        subscales = self.repository.patient_subscales(patient_id)
        attributes = self.repository.protocol_attributes()
        ppf_contrib = compute_ppf_for_patients(subscales, attributes)
        file_path = persist_ppf(ppf_contrib)
        return {
            "message": f"Computation and persistence successful for patient {patient_id}",
            "patient_id": patient_id,
            "subscales_used": list(ppf_contrib.attrs.get("SUBSCALES", [])),
            "saved_to": str(file_path),
        }

    def compute_protocol_similarity(self) -> dict:
        """Compute + persist the protocol similarity matrix.

        Delegates to `compute.compute_protocol_similarity_matrix` +
        `persist_similarity`.
        """
        attributes = self.repository.protocol_attributes()
        similarity = compute_protocol_similarity_matrix(attributes)
        file_path = persist_similarity(similarity)
        return {
            "message": "Protocol similarity computation and persistence successful.",
            "saved_to": str(file_path),
        }


# Back-compat alias. The orchestrator was named `CDSS` through v0.3.1;
# external callers (cdss-supervisor, examples) still import that name.
# Removal is tracked in SUPERVISOR_MIGRATION_PLAN.md — migrate consumers
# to `RecommendationService`, then drop this.
CDSS = RecommendationService

