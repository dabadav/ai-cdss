# Architecture review — top-edge cleanup (post-F7)

Status: **plan**. No code changes yet. This branch
(`refactor/orchestrator-api`) is where the public-API-changing fixes
land, because items 1–2 alter the orchestrator + repository surface
that `cdss-supervisor` and any external caller import.

## Context

The F0–F7 refactor made the *engine / data substrate* layer genuinely
clean: `engine.py` (EngineState / SimilarityMatrix Protocols + adapters),
`data.py` (CohortRepository Protocol + Cohort bundle), `scoring.py`
(typed stage contracts), `precompute.py` (pure offline functions). All
section-bannered, substrate-agnostic, 83/83 tests green.

The weakness is the **top edge**: the orchestrator (`interface/cdss.py`,
class `CDSS`) and its write path never got the same discipline, and
there is no write-side abstraction mirroring the read-side
`CohortRepository`. Fixing items 1 + 2 makes the architecture uniform
end-to-end.

Ranked by payoff.

---

## Progress

- ✅ **#1 — orchestrator rename + decomposition** — `f9`. `CDSS` →
  `RecommendationService` (+ back-compat alias), `cdss` local → `engine`,
  `_recommend_for_patients_core` split into 6 helpers.
- ✅ **#2 — PrescriptionStore write-side port** — `f8`. Writes + idempotency
  no longer touch `repository.interface`.
- ✅ **#3 — `metrics.py`** — resolved as no-change (already bannered by
  aggregation level; see item 3).
- ✅ **Smaller seams** — `scoring.py`→`pipeline.py` + `Imputer` copy done
  (`f10`); `Cohort` bundle won't-fix; `interface/` flatten deferred.

---

## 1. `interface/cdss.py` is the unrefactored island — biggest gap [DONE — f9]

The F-series rewrote everything except this module. It is still
v0.3.1-style: 544 lines, no section banners, `"""` docstrings, bare
`except Exception`.

### 1a. Naming collision

Class `CDSS` (the app orchestrator) holds a local variable
`cdss = Recommender(...)` (`cdss.py:163`). So `cdss.recommend()` calls
the *Recommender*, not the CDSS. Four overlapping names for distinct
things: package `ai_cdss`, module `cdss`, class `CDSS`, engine
`Recommender`.

**Fix:** rename the orchestrator class → `RecommendationService`
(or move to `app.py` at root). Kill the `cdss` local var (call it
`engine`). Public-API change — `from ai_cdss.interface.cdss import CDSS`
is consumed externally.

### 1b. God-method

`_recommend_for_patients_core` (~150 lines) mixes: empty-guard, fetch,
PPF check, pipeline run, per-patient loop, status rollup, payload build,
JSON-log persist, debug dump. The success and failure paths duplicate
~40 lines of payload assembly.

**Fix:** extract a `RunReport`/payload builder, split the per-patient
loop, section-banner the file like the rest of the package.

---

## 2. Read/write asymmetry — missing write-side port [DONE — f8]

`CohortRepository` is a clean **read** abstraction (`find() → Cohort`).
Every **write** bypasses it and reaches into the concrete DB internals:

| Site | Leak |
|---|---|
| `cdss.py:458` | `self.repository.interface.engine` |
| `cdss.py:466` | `self.repository.interface._fetch(...)` — calls a **private** method for the idempotency SQL |
| `cdss.py:490` | `self.repository.interface.add_prescription_staging_entry(...)` |
| `cdss.py:506` | `self.repository.interface.add_recsys_metric_entry(...)` |

So `CDSS` depends on `CohortRepository` (clean) **and** on
`RGSCohortRepository.interface` being a `DatabaseInterface` exposing
that (partly private) API. The substrate-agnostic win evaporates on the
write path: a synthetic / in-memory repository satisfies `find()` but
`_already_prescribed` + persistence crash.

**Fix:** add a `PrescriptionStore` Protocol symmetric to
`CohortRepository`:

```python
@runtime_checkable
class PrescriptionStore(Protocol):
    def already_prescribed(self, patient_id: int, week_start: date) -> bool: ...
    def save_prescriptions(self, rows: list[PrescriptionStagingRow]) -> None: ...
    def save_metrics(self, rows: list[RecsysMetricsRow]) -> None: ...
```

Production impl wraps `DatabaseInterface`; an in-memory impl unblocks
synthetic backtests + the cdss-supervisor migration. The idempotency
SQL moves behind `already_prescribed`, off the private `_fetch`.

---

## 3. `metrics.py` has three audiences in one 556-line file

- Signal kernels: `compute_ewma`, `apply_savgol_filter_groupwise`,
  `get_rolling_theilsen_slope`.
- Pipeline feature-builders (consumed by `scoring.py`): `build_delta_dm`,
  `build_usage`, `build_week_usage`, `build_prescription_days`,
  `build_recent_adherence`, `build_week_since_start`,
  `include_missing_sessions`.
- Offline math (consumed by `precompute.py`): `compute_ppf`,
  `compute_protocol_similarity`, `feature_contributions`.

Pipeline-time vs offline-time, different abstraction levels, one module.
**Fix:** split feature-builders from offline kernels, or at minimum
section-banner the three groups.

**RESOLVED — no change. `metrics.py` is already section-bannered into 7
groups organized by tensor aggregation level (the project's chosen mental
model). The "three audiences" map cleanly onto existing sections (1 =
signal kernels, 3–5 = pipeline builders, 6 = offline math). Splitting into
separate files would fight the aggregation-level organization and the
flat-layout preference (few big bannered files > many small ones). The
banners already make the audiences navigable.**

---

## Smaller seams

- **✅ `scoring.py` undersells itself.** Held the entire `DataPipeline`
  (windowing, feature merge, imputation) — actual scoring was the ~20-line
  `Scorer`. **DONE (f10): renamed `scoring.py` → `pipeline.py`** (matches
  the original target end-state layout). 2 importers updated, no shim.
- **✅ `Imputer` mutation inconsistency.** `init_metrics` mutated `data`
  in place; `impute_metrics` did `.copy()` first. **DONE (f10):
  `init_metrics` now copies first — both methods share one no-aliasing
  contract.**
- **`Cohort` half-consumed.** `pipeline.process(cohort)` reads only
  `patient/session/ppf`; `similarity` is pulled out separately in the
  orchestrator; `whitelist` is audit-only (the stored `Cohort.whitelist`
  field is never read downstream — pure trace; left as-is). **WON'T-FIX:
  `Cohort` is a deliberate single-fetch bundle (sklearn.Bunch style).**
  Attempted a `PipelineInputs` Protocol fix (f11) and **reverted it**:
  it duplicated `PreparedInputs` (same three fields, different stage) for
  near-zero gain — `process` is duck-typed, so a 3-field stub already
  worked without the Protocol. The lesson: the cure (a second look-alike
  input type) was worse than the documented smell. A behavioral test
  (`test_pipeline_runs_on_three_field_stub`) was kept to pin the 3-field
  dependency; the Protocol was dropped.
- **`interface/` subdir for one orchestrator + debug helper.**
  **DEFERRED: low value, high churn — moving `interface/cdss.py` → root
  `app.py` ripples through every external `from ai_cdss.interface...`
  import for a cosmetic flatten. Revisit alongside the supervisor
  migration, when those imports are being touched anyway.**

---

## Suggested order

1. `PrescriptionStore` port (item 2) — unblocks everything downstream,
   including the `SUPERVISOR_MIGRATION_PLAN.md` private-method spelunking.
2. Orchestrator rename + decomposition (item 1) — depends on the port.
3. `metrics.py` split + `scoring.py` → `pipeline.py` rename (items 3 +
   smaller seams) — internal, can land independently.

Items 1 + 2 change the public import surface, so they ripple into
cdss-supervisor — coordinate with `SUPERVISOR_MIGRATION_PLAN.md`.
