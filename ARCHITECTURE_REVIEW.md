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

## 1. `interface/cdss.py` is the unrefactored island — biggest gap

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

## 2. Read/write asymmetry — missing write-side port

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
section-banner the three groups. (Not a public-API change if re-exports
are kept — lower priority, can land separately.)

---

## Smaller seams

- **`scoring.py` undersells itself.** Holds the entire `DataPipeline`
  (windowing, feature merge, imputation) — actual scoring is the ~20-line
  `Scorer`. Rename → `pipeline.py` (matches the original target
  end-state layout).
- **`Cohort` half-consumed.** `pipeline.process(cohort)` reads only
  `patient/session/ppf`; `similarity` is pulled out separately in CDSS
  (`cdss.py:161`); `whitelist` is audit-only. Bundle passes 6 fields,
  pipeline reads 3 — fuzzy contract.
- **`Imputer` mutation inconsistency.** `init_metrics` mutates `data`
  in place (`scoring.py:261`); `impute_metrics` does `.copy()` first
  (`:276`). Aliasing footgun — pick one.
- **`interface/` subdir for one orchestrator + debug helper** is mild
  fragmentation against the flat-layout philosophy. `app.py` at root
  would be flatter.

---

## Suggested order

1. `PrescriptionStore` port (item 2) — unblocks everything downstream,
   including the `SUPERVISOR_MIGRATION_PLAN.md` private-method spelunking.
2. Orchestrator rename + decomposition (item 1) — depends on the port.
3. `metrics.py` split + `scoring.py` → `pipeline.py` rename (items 3 +
   smaller seams) — internal, can land independently.

Items 1 + 2 change the public import surface, so they ripple into
cdss-supervisor — coordinate with `SUPERVISOR_MIGRATION_PLAN.md`.
