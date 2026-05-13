# Functionality Refactor — engine API redesign

Branch: `functionality-refactor` (off `readable`).

This is the **objective-driven** branch. Backward compatibility is
*not* a goal here — the `readable` branch already preserves it
byte-for-byte. This branch is free to break legacy surfaces in service
of three objectives:

  1. **Introspectability** — every intermediate artifact accessible
     as a property on a result object (PCA / sklearn / TensorFlow
     model style).
  2. **Substrate-agnostic engine input** — the recommendation engine
     does not require pandas. Synthetic / dict / NumPy state plugs in
     directly.
  3. **Performance** — groupby hoisting, vectorization, then
     polars-or-bust if the engine remains slow.

## Compatibility policy

- **Keep**: `ai_cdss.interface.CDSSInterface` — the only top-level
  abstraction. Internal callers (cli, supervisor) use this.
- **Drop** without ceremony:
  - `from ai_cdss.cdss import CDSS` — `cdss.py` re-export shim
  - `from ai_cdss.loaders import DataLoader` — `loaders/__init__.py` shim
  - `from ai_cdss.services import …` — `services/__init__.py` shim
  - `from ai_cdss.processing import DataProcessor` — already gone in
    Phase 3; remaining mentions get scrubbed
  - `FeatureBuilder` OO wrapper class (`feature.py § 7`) — keep module
    functions, drop the class
  - `DataProcessor` (`pipeline.py § 4`) — fold into `DataPipeline`
  - Anything else that exists only for v0.3.1 callers
- **Public API after this branch**: `CDSSInterface` + the types it
  returns. That's it.

## Why this branch exists

The user said:

> "I feel from the main class only you should be able to access the
> intermediate step data from class properties as in PCA where you can
> access the .data or as in tensorflow."

> "Plug-and-play for non-engine callers: NOT YET — engine still tied
> to DataFrame substrate."

> "do not care so much about backwards compatibility but on the
> objectives we have set only keep the last abstraction CDSS interface
> for now"

So we delete legacy paths aggressively and rebuild the engine API
around two new types: `EngineState` (input) and `RecommendationResult`
(output).

## Phases

### Phase F0 — Rip back-compat surface

  - Delete `cdss.py` (the 11-line shim).
  - Delete `loaders/` and `services/` directories entirely.
  - Update `__init__.py` to re-export only `CDSSInterface` (and the
    pandera schemas, since they're public types).
  - Remove `FeatureBuilder` class from `feature.py`; pipeline calls
    module functions directly.
  - Remove `DataProcessor` shim from `pipeline.py`; `CDSSInterface`
    instantiates `DataPipeline` directly.
  - All 35 unit tests should still pass IF they don't depend on the
    legacy import paths. Tests that DO will be rewritten.

  Commit: `f0: rip v0.3.1 back-compat surface`

### Phase F1 — `RecommendationResult` (introspectable)

  - New `RecommendationResult` dataclass in `recommend.py`.
  - `CDSSInterface.recommend_for_patients` returns per-patient
    `RecommendationResult`s in its existing payload structure.
  - Internal `CDSS.recommend` (the engine core) returns
    `RecommendationResult` directly (not a `pd.DataFrame`).
  - Score lineage threaded through `Scorer.compute_score` → captured
    in `RecommendationResult.scores_by_protocol`.

  Properties:

  ```python
  result.recommendations        # final pd.DataFrame
  result.trace                  # full structured trace dict
  result.patient_state          # PatientState
  result.branch                 # bootstrap / repeat / update
  result.swap_decisions         # list[SubstituteResult]
  result.topup_events           # list[dict]
  result.mvt_mean               # float | None
  result.swap_targets           # list[int]
  result.swap_reasons           # dict[int, str]
  result.scores_by_protocol     # dict[int, ProtocolScoreBreakdown]
  result.candidate_pool_for(removed_id)  # method
  ```

  Commit: `f1: introspectable RecommendationResult`

### Phase F2 — `EngineState` protocol + `DictBackedState`

  - New `engine.py` (top-level): `EngineState` protocol, `ProtocolRow`
    dataclass, `DictBackedState`, `DataFrameBackedState`,
    `SimilarityMatrix` protocol, `DictSimilarity`,
    `DataFrameSimilarity`.
  - `PatientState` becomes `DataFrameBackedState` (renamed).
  - The engine core in `recommend.py` rewritten to call methods on
    `EngineState` — no direct pandas inside the engine body.
  - `CDSSInterface` keeps its current input (loader → pipeline → scoring
    DataFrame), wraps in `DataFrameBackedState` at the boundary, calls
    the engine.
  - Tests added for `DictBackedState` end-to-end recommend.

  Commit: `f2: EngineState protocol + DictBackedState`

### Phase F3 — Performance pass

  - Profile cdss-supervisor backtest sweep (chained mode, ~150
    patient-weeks).
  - Hoist `groupby` calls, vectorize hot lambdas, cache slice
    operations.
  - Document before/after.

  Commit: `f3: groupby hoisting + vectorization`

### Phase F4 — `polars` substrate (decide after F3 profile)

  - If F3 doesn't hit the 5× speedup target, add a
    `PolarsBackedState` that wraps a `pl.LazyFrame`. The
    `EngineState` protocol from F2 makes this drop-in.
  - cdss-supervisor / cli unchanged; the substrate swap is internal
    to `CDSSInterface`.

  Commit: `f4: polars substrate (optional)`

### Phase F5 — Docs + v0.5.0 tag

  - Update `architecture.md` + `code_structure.md` with the new API.
  - New `quickstart_synthetic.md`: 10-line synthetic backtest example.
  - New `adoption.md`: migration guide for cli + supervisor (mostly
    "use `CDSSInterface`, the rest changed shape").
  - Tag `v0.5.0`.

  Commit: `f5: docs + v0.5.0`

## End-state layout

```
src/ai_cdss/
├── __init__.py        public: CDSSInterface + schemas only
├── clinical.py        unchanged
├── constants.py       unchanged
├── engine.py          NEW — EngineState protocol + state adapters
├── feature.py         drop FeatureBuilder class; keep module fns
├── interface/
│   ├── __init__.py    CDSSInterface
│   └── recommender.py CDSSInterface impl (returns RecommendationResult)
├── loader.py          unchanged
├── models.py          unchanged
├── pipeline.py        drop DataProcessor; fold into DataPipeline
├── recommend.py       PatientState removed → moved to engine.py;
                       all engine fns rewritten against EngineState;
                       returns RecommendationResult
├── score.py           Scorer returns (frame, breakdown_dict)
├── service.py         unchanged
└── utils.py           unchanged
```

Removed: `cdss.py`, `loaders/`, `services/`.

## Acceptance per phase

- Tests pass at every commit. Tests pinned to legacy import paths get
  rewritten to import from current locations.
- `CDSSInterface(loader, pipeline).recommend_for_patients([pid])`
  works end-to-end against the real DB at every commit.
- Phase F2 onward: a 10-line synthetic example produces a valid
  `RecommendationResult` without instantiating `pandas.DataFrame`.

## Tracking

- [ ] F0 — rip back-compat
- [ ] F1 — RecommendationResult
- [ ] F2 — EngineState + DictBackedState
- [ ] F3 — performance pass
- [ ] F4 — polars (if needed)
- [ ] F5 — docs + v0.5.0
