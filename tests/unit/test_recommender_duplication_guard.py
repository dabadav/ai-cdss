"""Idempotency / duplication-guard tests for CDSS._process_patient.

Covers both duplication paths the guard addresses:
  (1) manual rerun on a patient already prescribed this week
  (2) cohort entry picked up before clinical_start (every cron tick would
      otherwise bootstrap a fresh RID)
"""
import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from ai_cdss.interface.cdss import CDSS


def _make_iface(staging_count: int):
    """Build a CDSS whose duplication-check query returns
    `staging_count` for any (patient, week_start)."""
    fake_df = pd.DataFrame({"n": [staging_count]})
    fake_interface = SimpleNamespace(
        engine=object(),  # truthy
        _fetch=MagicMock(return_value=fake_df),
        add_prescription_staging_entry=MagicMock(),
        add_recsys_metric_entry=MagicMock(),
    )
    fake_repository = SimpleNamespace(interface=fake_interface)
    fake_processor = SimpleNamespace()
    iface = CDSS.__new__(CDSS)
    iface.repository = fake_repository
    iface.processor = fake_processor
    # debug=True skips _save_prescriptions/_save_metrics — keeps the test
    # focused on the duplication guard rather than persistence wiring.
    iface.debug = True
    from ai_cdss.interface.debug import DebugReport
    from ai_cdss.constants import DEFAULT_DEBUG_DIR
    iface.debug_service = DebugReport(DEFAULT_DEBUG_DIR)
    return iface, fake_interface


def _fake_cdss():
    """Mock CDSS.recommend to return a RecommendationResult (the new
    typed return type as of phase F1)."""
    from ai_cdss.recommender import PatientState, RecommendationResult
    fake = MagicMock()
    rec = pd.DataFrame({
        "PATIENT_ID":  [1, 1],
        "PROTOCOL_ID": [200, 201],
        "DAYS":        [[0], [1]],
    })
    rec.attrs = {"trace": {"branch": "update"}}
    # PatientState requires a scoring frame; build a minimal one.
    scoring = pd.DataFrame({
        "PATIENT_ID":  [1, 1],
        "PROTOCOL_ID": [200, 201],
        "DAYS":        [[0], [1]],
    })
    fake.recommend.return_value = RecommendationResult(
        recommendations=rec,
        trace={"branch": "update"},
        patient_state=PatientState(scoring, 1),
        branch="update",
        swap_decisions=[],
        topup_events=[],
        mvt_mean=None,
        swap_targets=[],
        swap_reasons={},
        scoring_attrs={},
    )
    return fake


def test_process_patient_skips_when_already_prescribed():
    iface, fake_iface = _make_iface(staging_count=35)
    cdss = _fake_cdss()
    scoring = pd.Timestamp("2026-05-05")
    start   = datetime.datetime(2026, 4, 7)  # patient in week 4

    res = iface._process_patient(
        patient=4904,
        cdss=cdss,
        protocol_similarity=None,
        scores=pd.DataFrame({
            "PATIENT_ID": [1], "PROTOCOL_ID": [200],
            "PPF": [0.5], "DELTA_DM": [0.0], "ADHERENCE_RECENT": [0.5],
            "SCORE": [1.0], "USAGE": [1], "USAGE_WEEK": [1], "SESSION_INDEX": [1],
        }),
        unique_id="run-uuid",
        datetime_start=start,
        scoring_date=scoring,
        force=False,
    )

    assert res["status"] == "skipped"
    assert res["skipped_reason"] == "already_prescribed"
    assert res["skipped_week"] == 4
    assert res["skipped_week_start"] == "2026-05-05"
    cdss.recommend.assert_not_called()
    fake_iface.add_prescription_staging_entry.assert_not_called()


def test_process_patient_skips_pre_trial_cohort_entries():
    """If clinical_start is in the future, current_week clamps to 0 and the
    guard checks STARTING_DATE == clinical_start. With existing rows, skip."""
    iface, fake_iface = _make_iface(staging_count=35)
    cdss = _fake_cdss()
    scoring = pd.Timestamp("2026-05-05")
    future_start = datetime.datetime(2026, 5, 12)  # 7d in future

    res = iface._process_patient(
        patient=4999,
        cdss=cdss,
        protocol_similarity=None,
        scores=pd.DataFrame({
            "PATIENT_ID": [1], "PROTOCOL_ID": [200],
            "PPF": [0.5], "DELTA_DM": [0.0], "ADHERENCE_RECENT": [0.5],
            "SCORE": [1.0], "USAGE": [1], "USAGE_WEEK": [1], "SESSION_INDEX": [1],
        }),
        unique_id="run-uuid",
        datetime_start=future_start,
        scoring_date=scoring,
        force=False,
    )

    assert res["status"] == "skipped"
    assert res["skipped_week"] == 0
    cdss.recommend.assert_not_called()


def test_process_patient_force_bypasses_guard():
    iface, fake_iface = _make_iface(staging_count=35)
    cdss = _fake_cdss()
    scoring = pd.Timestamp("2026-05-05")
    start   = datetime.datetime(2026, 4, 7)

    res = iface._process_patient(
        patient=4904,
        cdss=cdss,
        protocol_similarity=None,
        scores=pd.DataFrame({
            "PATIENT_ID": [1], "PROTOCOL_ID": [200],
            "PPF": [0.5], "DELTA_DM": [0.0], "ADHERENCE_RECENT": [0.5],
            "SCORE": [1.0], "USAGE": [1], "USAGE_WEEK": [1], "SESSION_INDEX": [1],
        }),
        unique_id="run-uuid",
        datetime_start=start,
        scoring_date=scoring,
        force=True,
    )

    assert res["status"] == "success"
    cdss.recommend.assert_called_once()


def test_process_patient_runs_when_no_existing_rows():
    iface, fake_iface = _make_iface(staging_count=0)
    cdss = _fake_cdss()
    scoring = pd.Timestamp("2026-05-05")
    start   = datetime.datetime(2026, 4, 7)

    res = iface._process_patient(
        patient=4904,
        cdss=cdss,
        protocol_similarity=None,
        scores=pd.DataFrame({
            "PATIENT_ID": [1], "PROTOCOL_ID": [200],
            "PPF": [0.5], "DELTA_DM": [0.0], "ADHERENCE_RECENT": [0.5],
            "SCORE": [1.0], "USAGE": [1], "USAGE_WEEK": [1], "SESSION_INDEX": [1],
        }),
        unique_id="run-uuid",
        datetime_start=start,
        scoring_date=scoring,
        force=False,
    )

    assert res["status"] == "success"
    cdss.recommend.assert_called_once()
