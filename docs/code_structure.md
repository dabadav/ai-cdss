# `ai-cdss` Code Structure

Navigation guide for the refactored codebase. Pairs with
`architecture.md` (which covers the *algorithm + dataflow*). This
document covers the *code map* — file by file, what's in each.

## Repo top level

```
ai-cdss-refactor/
├── REFACTOR_PLAN.md          5-phase plan + checklist
├── docs/
│   ├── architecture.md       tensor model + dataflow diagrams
│   └── code_structure.md     this file
├── pyproject.toml            uv-managed; same package name `ai_cdss`
├── src/ai_cdss/              the package (12 flat files)
└── tests/                    35 unit tests, all green
```

## `src/ai_cdss/` — 12 flat files

```
__init__.py        29 L   public re-exports
cdss.py            11 L   back-compat: `from ai_cdss.cdss import CDSS`
clinical.py        80 L   clinical-subscale → deficit/attribute mapping
constants.py      158 L   column names, axis defs, thresholds
feature.py        614 L   feature reductions over tensor axes
loader.py         510 L   data loaders (DB, CSV, synthetic)
models.py         303 L   pandera schemas + DataUnit + DataUnitSet
pipeline.py       465 L   typed contracts + DataPipeline + DataProcessor
recommend.py      711 L   recommendation engine (PatientState + CDSS)
score.py           99 L   Imputer + Scorer
service.py        278 L   PPF / similarity / whitelist services
utils.py          107 L   MultiKeyDict + small helpers
                 ──────
                  3365 L
```

Plus three shim subdirs (one-line re-exports for legacy imports):

```
loaders/__init__.py    13 L   → re-exports from loader.py
services/__init__.py   14 L   → re-exports from service.py
interface/             (untouched; CDSSInterface lives here)
```

## File-by-file map

### `__init__.py`
- **Re-exports** the canonical public API: `CDSS`, `DataLoader`, `DataPipeline`, `DataProcessor`, `ClinicalSubscales`, `ProtocolToClinicalMapper`, plus the pandera schemas (`SessionSchema`, `TimeseriesSchema`, `PPFSchema`, `PCMSchema`, `ScoringSchema`).
- All these come from the corresponding flat module: no business logic lives here.

### `cdss.py`
- **11 lines**. Back-compat shim. Imports `CDSS`, `PatientState`, `SubstituteResult` from `recommend.py` and re-exports them. Lets the legacy import `from ai_cdss.cdss import CDSS` keep working.

### `clinical.py`
- **2 classes, both stateless aside from YAML-config paths:**
  - `ClinicalSubscales` — computes a deficit matrix `1 - (patient_scores / max_scores)` from raw subscale evaluations. YAML config under `config/scales.yaml`.
  - `ProtocolToClinicalMapper` — maps protocol-attribute columns into clinical-scale columns via aggregation (default mean). YAML config under `config/mapping.yaml`.
- Used by `service.PPFService` and `service.ProtocolSimilarityService`.

### `constants.py`
- **Untouched from v0.3.1**. Column-name constants (PATIENT_ID, PROTOCOL_ID, SCORE, DAYS, …), axis groupings (BY_PP, BY_PPS, BY_PPST, BY_ID), AISN trial defaults (N=12, N_DAYS=7, PROTOCOLS_PER_DAY=5), file paths (PPF_PARQUET_FILEPATH, …), enums (SessionStatus).
- Every other module imports from here.

### `feature.py` — 7 sections by tensor aggregation level
The most important refactor file: organized by **which axis gets reduced**.

| Section | Reduces over | Operations |
|---|---|---|
| 1 | time within a (PP) group | `compute_ewma`, `apply_savgol_filter_groupwise`, `get_rolling_theilsen_slope` |
| 2 | session-shape gaps | `include_missing_sessions`, `generate_expected_sessions` |
| 3 | time → per-session value | `build_delta_dm`, `build_recent_adherence` |
| 4 | sessions → per-(PP) scalar | `build_usage`, `build_week_usage`, `build_prescription_days`, `build_number_prescriptions` |
| 5 | everything except patient | `build_week_since_start`, `_with_weeks_since_start`, `_last_completed_week_window` |
| 6 | cross-cohort matrices | `feature_contributions`, `compute_ppf`, `compute_protocol_similarity` |
| 7 | OO wrapper | `FeatureBuilder` class — bundles the per-(PP) and per-patient features |

Each section has a banner header (`╔═...═╗`). Search the file for `SECTION N` to jump.

### `loader.py` — 5 sections by loader type
- **Section 1** — File-IO helpers (`_decode_subscales`, `_safe_load_csv`, `_load_patient_subscales`, `_load_protocol_attributes`, `_load_protocol_similarity`, `_load_ppf_data`).
- **Section 2** — `DataLoaderBase` (abstract). Every loader implements 7 methods: `load_session_data`, `load_timeseries_data`, `load_ppf_data`, `load_protocol_similarity`, `load_patient_subscales`, `load_protocol_attributes`, `fetch_and_validate_patients`.
- **Section 3** — `DataLoader` (production, RGS-MySQL via `rgs_interface.DatabaseInterface`; PPF + similarity from local Parquet/CSV).
- **Section 4** — `DataLoaderLocal` (CSV-backed; for tests + offline replay).
- **Section 5** — `DataLoaderMock` (synthetic data via `evaluation.synthetic`).

### `models.py`
- **Untouched from v0.3.1**. Pandera DataFrame schemas (`SessionSchema`, `TimeseriesSchema`, `PPFSchema`, `PCMSchema`, `ScoringSchema`) and pipeline-bundling types (`DataUnit`, `DataUnitSet`, `DataUnitName`, `Granularity` enum).

### `pipeline.py` — 4 sections
- **Section 1** — Typed contracts: `PreparedInputs`, `SessionLevelFeatures`, `ProtocolLevelFeatures`, `MergedFeatures`, `ScoringInput`, `ScoringOutput`. Each is a frozen dataclass declaring required columns; validates at construction (skippable via `validate_on_init=False`).
- **Section 2** — `get_nth` helper (used by the imputer for first/last per-group lookups).
- **Section 3** — `DataPipeline` orchestrator. One method per stage:
  - `_prepare` → `PreparedInputs`
  - `_build_features` → `MergedFeatures` (via `_session_level_features` + `_protocol_level_features` + `_broadcast_session_onto_protocol`)
  - `_impute_features` → `ScoringInput` (via `_impute_per_patient_median`)
  - `_bootstrap_scoring_input` → `ScoringInput` (no-sessions fallback)
  - `_score` → `ScoringOutput`
- **Section 4** — `DataProcessor` back-compat shim (legacy callers do `DataProcessor().process_data(data, date)`).

### `recommend.py` — 10 sections by algorithm phase
The largest file (711 L). Sections map 1:1 to the algorithm phases.

| Section | What's in it |
|---|---|
| 1 | `PatientState` class — patient-scoped scoring view |
| 2 | `_init_trace`, `_serialize_prior`, `_serialize_final`, `_safe_float`, `_safe_int` |
| 3 | `_bootstrap_branch`, `_round_robin_across_days`, `_seed_rows_from_schedule` |
| 4 | `_repeat_branch` |
| 5 | `_below_mean_protocols` (the MVT criterion) |
| 6 | `_similarities_for`, `_top_n_similar`, `_most_similar_within` |
| 7 | `SubstituteResult` dataclass + `_find_substitute`, `_least_used_among` |
| 8 | `_update_branch`, `_select_swap_targets`, `_rows_kept_unchanged`, `_build_swap_rows`, `_materialize_swap_row`, `_swap_trace_event` |
| 9 | `_fill_grid_coverage`, `_index_protocols_by_day`, `_build_filler_pool`, `_apply_filler`, `_record_exhaustion` |
| 10 | `CDSS` class (public entry-point) |

The `CDSS` class is ~70 lines: it just dispatches to the right branch, runs top-up, attaches the trace.

### `score.py` — 2 trivial classes
- `Imputer` — `init_metrics` (zero-fill counts, default DAYS=`[]`) + `impute_metrics` (fill NaN with per-patient median).
- `Scorer` — linear combination: `SCORE = w0·RECENT_ADHERENCE + w1·DELTA_DM + w2·PPF` (default weights `[1, 1, 1]`).

### `service.py` — 5 sections
- **Section 1** — `load_yaml` helper.
- **Section 2** — `ProtocolWhitelistService` — reads `config/protocol_whitelist.yaml` and returns the allowed-protocol ID list.
- **Section 3** — `RecommendationDataService` — orchestrator. `.prepare(patient_list)` returns `(rgs_data, protocol_similarity)`. Applies the whitelist filter to session, PPF, similarity.
- **Section 4** — `PPFService` — `compute_patient_fit` + `persist_ppf` + `compute_and_persist_patient_fit`. Reads patient subscales + protocol attributes, computes PPF + contribution decomposition, persists to Parquet.
- **Section 5** — `ProtocolSimilarityService` — compute pairwise Gower similarity from protocol attributes; persist to CSV.

### `utils.py`
- **`MultiKeyDict`** — dict-like with multi-key support (used by clinical YAML parsing).
- **`_json_default`** — JSON serialization fallback for non-standard types.
- **`generate_unique_filename`** — file-name uniqueness helper.

### Subpackages (untouched / shim)

- `config/` — YAML configs (`scales.yaml`, `mapping.yaml`, `protocol_whitelist.yaml`).
- `data/` — embedded CSV resources (`protocol_attributes.csv`).
- `evaluation/synthetic.py` — synthetic data generators (used by `DataLoaderMock`).
- `interface/` — `CDSSInterface` (the production-DB-aware top-level wrapper).
- `loaders/__init__.py` — re-export shim → `loader.py`.
- `services/__init__.py` — re-export shim → `service.py`.

## Import graph

```
                              constants.py ← (everyone)
                                    │
                              models.py
                                    │
                ┌───────────────────┼───────────────────┐
                │                   │                   │
            loader.py           clinical.py         utils.py
                │                   │
                ├──────► service.py◄┘
                │             │
                ▼             ▼
            feature.py    pipeline.py
                              │
                              ▼
                          score.py
                              │
                              ▼
                         recommend.py
                              │
                              ▼
                            cdss.py
                              │
                              ▼
                          interface/
                          (CDSSInterface)
```

No cycles. Top-of-stack (cdss.py / recommend.py) re-exports the public surface.

## Class inventory

| Class | File | Lines | Role |
|---|---|---|---|
| `CDSS` | recommend.py § 10 | ~70 | top-level recommendation entry-point |
| `PatientState` | recommend.py § 1 | ~50 | patient-scoped scoring view |
| `SubstituteResult` | recommend.py § 7 | ~5 | substitute search outcome (dataclass) |
| `DataPipeline` | pipeline.py § 3 | ~140 | feature-build → impute → score orchestrator |
| `DataProcessor` | pipeline.py § 4 | ~10 | v0.3.1 back-compat shim |
| `PreparedInputs` | pipeline.py § 1 | ~25 | typed input contract |
| `SessionLevelFeatures` | pipeline.py § 1 | ~10 | session-level feature contract |
| `ProtocolLevelFeatures` | pipeline.py § 1 | ~15 | protocol-level feature contract |
| `MergedFeatures` | pipeline.py § 1 | ~10 | session × protocol broadcast contract |
| `ScoringInput` | pipeline.py § 1 | ~15 | one-per-(PP) contract |
| `ScoringOutput` | pipeline.py § 1 | ~15 | final scored contract |
| `DataLoaderBase` | loader.py § 2 | ~25 | abstract loader interface |
| `DataLoader` | loader.py § 3 | ~120 | production DB-backed loader |
| `DataLoaderLocal` | loader.py § 4 | ~50 | CSV-backed loader |
| `DataLoaderMock` | loader.py § 5 | ~55 | synthetic loader |
| `RecommendationDataService` | service.py § 3 | ~50 | pipeline-prepare orchestrator |
| `PPFService` | service.py § 4 | ~70 | compute + persist PPF |
| `ProtocolSimilarityService` | service.py § 5 | ~40 | compute + persist similarity |
| `ProtocolWhitelistService` | service.py § 2 | ~20 | load allowed-protocols YAML |
| `FeatureBuilder` | feature.py § 7 | ~50 | OO wrapper for feature module functions |
| `ClinicalSubscales` | clinical.py | ~40 | deficit-matrix computation |
| `ProtocolToClinicalMapper` | clinical.py | ~25 | protocol → clinical-scale mapping |
| `Imputer` | score.py | ~25 | NaN-fill + dtype coercion |
| `Scorer` | score.py | ~20 | linear-combo SCORE formula |

## Entry points

| Use case | Entry point |
|---|---|
| Recommend for one patient (in-process) | `CDSS(scoring=...).recommend(patient_id, similarity)` |
| Recommend for one patient (full pipeline) | `CDSSInterface(...).recommend_for_patients([pid])` |
| Run the data pipeline manually | `DataPipeline().process(data, scoring_date)` |
| Compute PPF for a new patient | `PPFService(loader).compute_and_persist_patient_fit([pid])` |
| Compute protocol similarity (when adding a protocol) | `ProtocolSimilarityService(loader).compute_and_persist_protocol_similarity()` |

## Tests

```
tests/unit/
├── test_cdss_recommend.py             3 tests — recommend branches + trace shape
├── test_feature_builder.py           10 tests — feature module functions
├── test_processing_contracts.py      14 tests — typed contract validation
└── test_recommender_duplication_guard.py  4 tests — CDSSInterface dedup guard
─────────────────────────────────────
                              total   35
```

All passing as of `3d496c2`. Run with:

```bash
PYTHONPATH=src python -m pytest tests/unit/
```

## How to find anything

| You want to... | Look at... |
|---|---|
| Understand the algorithm flow | `docs/architecture.md` |
| Find a specific class | "Class inventory" table above |
| Change a feature definition | `feature.py` (section by aggregation level) |
| Change the score formula | `score.py:Scorer.compute_score` |
| Change the MVT criterion | `recommend.py § 5` |
| Change the substitute search | `recommend.py § 7` |
| Add a new loader | new class in `loader.py § 5+`, implement `DataLoaderBase` |
| Add a new service | new class in `service.py § 6+` |
| Validate a new column at a pipeline boundary | add to the `REQUIRED` list in the relevant contract in `pipeline.py § 1` |
| Trace what one recommendation did | the `trace` dict on `recommendations.attrs["trace"]` |
