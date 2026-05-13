# `ai-cdss` Code Structure

Navigation guide for the refactored codebase. Pairs with
`architecture.md` (which covers the *algorithm + dataflow*). This
document covers the *code map* — file by file, what's in each.

## Repo top level

```
ai-cdss-refactor/
├── DATA_LOADING_PLAN_V2.md  Repository-pattern data-layer refactor
├── docs/
│   ├── architecture.md      tensor model + dataflow diagrams
│   ├── class_diagram.md     mermaid class diagram (every public class)
│   ├── class_diagram.html   stand-alone HTML viewer (pan + zoom)
│   ├── code_structure.md    this file
│   └── dataflow.md          end-to-end stage diagram
├── pyproject.toml
├── src/ai_cdss/             the package
└── tests/unit/              83 unit tests, all green
```

## `src/ai_cdss/` — top-level modules

```
__init__.py            12 L   public re-exports (CDSS)
constants.py          158 L   column names, axis defs, thresholds
data.py               501 L   Cohort + CohortRepository + RGSCohortRepository
engine.py             604 L   EngineState protocol + adapters + similarity
metrics.py            556 L   feature reductions over tensor axes
interface/            635 L   CDSS + DebugReport
precompute.py         160 L   PPF + similarity offline computations
recommender.py          783 L   recommendation engine (PatientState + CDSS)
scoring.py            540 L   typed contracts + Imputer + Scorer + DataPipeline
utils.py              107 L   MultiKeyDict + small helpers
                     ──────
                     ~4 056 L
```

Plus two non-code subpackages:

```
config/               YAML configs (scales.yaml, mapping.yaml, protocol_whitelist.yaml)
resources/            Embedded CSV resources (protocol_attributes.csv)
```

## File-by-file map

### `__init__.py`
- Single public entry: `from ai_cdss import CDSS`.
- DataFrame column shapes are documented in `docs/schemas.md` rather
  than exported as types — the recommender does not validate frames
  at runtime.

### `precompute.py` — 2 sections, 4 pure functions
- **Section 1 — PPF**: `compute_ppf_for_patients(patient_subscales,
  protocol_attributes, scales_yaml=None, mapping_yaml=None)` +
  `persist_ppf(df, path=None)`. Pure: takes raw frames in, returns/
  writes the joined `(PPF, CONTRIB)` long-form DataFrame.
- **Section 2 — Protocol similarity**:
  `compute_protocol_similarity_matrix(protocol_attributes,
  mapping_yaml=None)` + `persist_similarity(df, path=None)`. Pairwise
  Gower distance over protocol attributes.
- No classes, no state. Used by the offline patient-registration +
  protocol-addition workflows (NOT by the recommendation hot path,
  which reads precomputed PPF/similarity from disk via the repository).

### `constants.py`
- Column-name constants (`PATIENT_ID`, `PROTOCOL_ID`, `SCORE`, `DAYS`,
  …), axis groupings (`BY_PP`, `BY_PPS`), AISN-trial defaults (`N=12`,
  `N_DAYS=7`, `PROTOCOLS_PER_DAY=5`), file paths
  (`PPF_PARQUET_FILEPATH`, …), enums (`SessionStatus`).
- Every other module imports from here.

### `data.py` — 5 sections, Repository pattern
Replaces v0.3.1's `loader.py` + `service.py` + `clinical.py` (3 files,
651 L, 7 classes) with one module that follows the **Repository
pattern**. Sectioned:
- **§ 1 — File-IO primitives** (pure functions): `read_yaml`,
  `read_csv`, `decode_subscales`, `load_whitelist`,
  `_load_protocol_attributes`, `_load_protocol_similarity`,
  `_load_ppf_data`.
- **§ 2 — `Cohort`** — frozen dataclass bundling the typed frames the
  pipeline + engine consume (`patient`, `session`, `ppf`, `similarity`,
  `whitelist`, `missing_ppf`). Modeled on `sklearn.Bunch` /
  Hugging Face `Dataset`.
- **§ 3 — `CohortRepository`** — PEP 544 `Protocol` with single method
  `find(patient_ids) -> Cohort`. Mirrors the `EngineState` Protocol
  one layer up — substrate-agnostic INPUT to the pipeline.
- **§ 4 — `RGSCohortRepository`** — production implementation. Pulls
  patient + session from RGS MySQL via `rgs_interface.DatabaseInterface`;
  reads precomputed PPF + similarity from `~/.ai_cdss/output/`. Also
  exposes specialized accessors (`patient_subscales`,
  `protocol_attributes`, `fetch_and_validate_patients`) for offline
  workflows — these are NOT part of the protocol contract.
- **§ 5 — Clinical mappers** — `ClinicalSubscales` (deficit matrix) +
  `ProtocolToClinicalMapper` (protocol-attribute → clinical-scale).
  Used by `precompute.py`.

### `engine.py` — 4 sections, Protocol + 4 adapters
- **§ 1 — `EngineState` Protocol** — abstract patient-scoring view.
  7 methods + 5 properties.
- **§ 2 — `SimilarityMatrix` Protocol** — abstract pairwise similarity
  table.
- **§ 3 — `ProtocolRow` dataclass** — one row of an engine state.
- **§ 4 — Adapters** — `PatientState`, `DictPatientState`,
  `DataFrameSimilarity`, `DictSimilarity` + `coerce_engine_state`
  helper.

### `metrics.py` — 6 sections by tensor aggregation level
Organized by which axis gets reduced.

| Section | Reduces over | Operations |
|---|---|---|
| 1 | time within a (PP) group | `compute_ewma`, `apply_savgol_filter_groupwise`, `get_rolling_theilsen_slope` |
| 2 | session-shape gaps | `include_missing_sessions`, `generate_expected_sessions` |
| 3 | time → per-session value | `build_delta_dm`, `build_recent_adherence` |
| 4 | sessions → per-(PP) scalar | `build_usage`, `build_week_usage`, `build_prescription_days` |
| 5 | everything except patient | `build_week_since_start`, `_last_completed_week_window` |
| 6 | cross-cohort matrices | `feature_contributions`, `compute_ppf`, `compute_protocol_similarity` |

Each section has a banner header (`╔═...═╗`).

### `interface/` — 2 modules
- **`recommender.py`** — `CDSS`. Top-level production-DB-aware
  wrapper. Owns a `CohortRepository` and a `DataPipeline`; orchestrates
  `recommend_for_patients` / `recommend_for_study` plus the idempotency
  guard against duplicate prescriptions. Also exposes
  `compute_patient_fit` and `compute_protocol_similarity` (delegates to
  `precompute.py`).
- **`debug.py`** — `DebugReport`. Writes per-run artifacts (scores +
  recs + prescriptions + metrics) under `~/.ai_cdss/debug/<run_id>/`.

### `scoring.py` — 5 sections
- **§ 1 — Typed contracts**: `PreparedInputs`, `SessionLevelFeatures`,
  `ProtocolLevelFeatures`, `MergedFeatures`, `ScoringInput`,
  `ScoringOutput`. Each is a frozen dataclass declaring required
  columns; validates at construction (skippable via
  `validate_on_init=False`).
- **§ 2 — `get_nth`** helper (used by the imputer for first/last per-
  group lookups).
- **§ 3 — `Imputer`** — NaN-fill + count-column zero-fill for the
  scoring input frame.
- **§ 4 — `Scorer`** — linear combination
  `SCORE = w0·RECENT_ADHERENCE + w1·DELTA_DM + w2·PPF` (default weights
  `[1, 1, 1]`).
- **§ 5 — `DataPipeline`** orchestrator. `process(cohort, scoring_date)`
  is the public entry; one method per stage internally:
  - `_prepare` → `PreparedInputs` (consumes `Cohort.patient/session/ppf`)
  - `_build_features` → `MergedFeatures`
  - `_impute_features` → `ScoringInput`
  - `_bootstrap_scoring_input` → `ScoringInput` (no-sessions fallback)
  - `_score` → `ScoringOutput`

### `recommender.py` — 9 sections by algorithm phase
The largest file. Sections map 1:1 to the algorithm phases.

| Section | What's in it |
|---|---|
| 1 | trace helpers (`_init_trace`, `_serialize_prior`, …) |
| 2 | `_bootstrap_strategy`, `_round_robin_across_days`, `_seed_rows_from_schedule` |
| 3 | `_repeat_strategy` |
| 4 | `_below_mean_protocols` (the MVT criterion) |
| 5 | `_similarities_for`, `_top_n_similar`, `_most_similar_within` |
| 6 | `SubstituteResult` dataclass + `_find_substitute`, `_least_used_among` |
| 7 | `_update_strategy` and swap helpers |
| 8 | `_top_up_schedule` and filler helpers |
| 9 | `CDSS` class (public entry-point) — dispatches strategies via `_run_strategy` |

### `utils.py`
- `MultiKeyDict` — dict-like with multi-key support (used by clinical
  YAML parsing).
- `_json_default` — JSON serialization fallback for non-standard types.
- `generate_unique_filename` — file-name uniqueness helper.

## Import graph

```
                              constants.py ← (everyone)
                                    │
                ┌───────────────────┼───────────────────┐
                │                   │                   │
            utils.py            data.py     metrics.py
                                    │                    │
                                    │             ┌──────┘
                                    ▼             ▼
                                precompute.py    scoring.py
                                                  │
                                                  ▼
                                              engine.py
                                                  │
                                                  ▼
                                              recommender.py
                                                  │
                                                  ▼
                                              interface/
                                              (CDSS)
```

No cycles.

## Class inventory

| Class | File | Lines | Role |
|---|---|---|---|
| `CDSS` | interface/cdss.py | ~470 | top-level orchestrator + DB-write side-effects |
| `DebugReport` | interface/debug.py | ~80 | per-run artifact dump |
| `Cohort` | data.py § 2 | ~30 | typed bundle returned by the repository |
| `CohortRepository` | data.py § 3 | ~10 | Protocol — abstract source of cohorts |
| `RGSCohortRepository` | data.py § 4 | ~120 | production implementation (DB + local files) |
| `ClinicalSubscales` | data.py § 5 | ~30 | deficit-matrix computation |
| `ProtocolToClinicalMapper` | data.py § 5 | ~25 | protocol → clinical-scale mapping |
| `DataPipeline` | scoring.py § 3 | ~140 | feature-build → impute → score orchestrator |
| `PreparedInputs`, `SessionLevelFeatures`, `ProtocolLevelFeatures`, `MergedFeatures`, `ScoringInput`, `ScoringOutput` | scoring.py § 1 | ~100 | typed pipeline contracts |
| `EngineState` | engine.py § 1 | ~30 | Protocol — substrate-agnostic scoring state |
| `SimilarityMatrix` | engine.py § 2 | ~10 | Protocol — abstract similarity table |
| `ProtocolRow` | engine.py § 3 | ~30 | one row of an engine state |
| `PatientState`, `DictPatientState` | engine.py § 4 | ~250 | EngineState adapters |
| `DataFrameSimilarity`, `DictSimilarity` | engine.py § 4 | ~90 | SimilarityMatrix adapters |
| `Recommender` | recommender.py § 9 | ~70 | top-level recommendation entry-point |
| `PatientState` | recommender.py § 1 | ~50 | patient-scoped scoring view |
| `RecommendationResult` | recommender.py | ~70 | typed return value of `Recommender.recommend` |
| `SubstituteResult` | recommender.py § 7 | ~10 | substitute search outcome (dataclass) |
| `Imputer`, `Scorer` | scoring.py § 3, § 4 | ~50 | NaN-fill + linear-combo SCORE formula |

## Entry points

| Use case | Entry point |
|---|---|
| Recommend for one patient (in-process, scoring frame in hand) | `Recommender(scoring=...).recommend(patient_id, similarity)` |
| Recommend for one patient (full pipeline) | `CDSS().recommend_for_patients([pid])` |
| Recommend for a study cohort | `CDSS().recommend_for_study([study_id], …)` |
| Run the data pipeline manually | `DataPipeline().process(cohort, scoring_date)` |
| Fetch one cohort | `RGSCohortRepository().find([pid_1, pid_2, …])` |
| Compute PPF for a new patient | `CDSS().compute_patient_fit([pid])` (or `compute.compute_ppf_for_patients(...)` + `compute.persist_ppf(...)` directly) |
| Compute protocol similarity (new protocol added) | `CDSS().compute_protocol_similarity()` (or `compute.compute_protocol_similarity_matrix(...)` + `compute.persist_similarity(...)`) |

## Tests

```
tests/unit/
├── test_cdss_recommender.py                 recommend branches + trace shape
├── test_feature_builder.py                feature module functions
├── test_performance.py                    micro-benchmarks for engine adapters
├── test_processing_contracts.py           typed contract validation
├── test_recommendation_result.py          RecommendationResult shape
├── test_recommender_duplication_guard.py  CDSS dedup guard
└── test_substrate_agnostic.py             EngineState protocol coverage
─────────────────────────────────────
                                  total 83
```

Run with:

```bash
PYTHONPATH=src python -m pytest tests/unit/
```

## How to find anything

| You want to... | Look at... |
|---|---|
| Understand the algorithm flow | `docs/architecture.md` |
| See the class diagram | `docs/class_diagram.md` (mermaid) or `class_diagram.html` (pan+zoom viewer) |
| Find a specific class | "Class inventory" table above |
| Change a feature definition | `metrics.py` (section by aggregation level) |
| Change the score formula | `scoring.py:Scorer.compute_score` |
| Change the MVT criterion | `recommender.py § 5` |
| Change the substitute search | `recommender.py § 7` |
| Add a new cohort source | new class implementing `CohortRepository` in `data.py § 4+` |
| Add a new engine substrate | new class implementing `EngineState` in `engine.py § 4+` |
| Validate a new column at a pipeline boundary | add to the `REQUIRED` list in the relevant contract in `scoring.py § 1` |
| Trace what one recommendation did | the `trace` dict on `RecommendationResult.trace` |
