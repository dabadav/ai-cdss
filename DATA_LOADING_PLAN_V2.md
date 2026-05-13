# Data Loading Refactor — v2 (Repository pattern)

Supersedes `DATA_LOADING_PLAN.md` (v1, PyTorch-framing — wrong reference).

Status: **proposal, no code changes**.

## Why v2

v1 framed the cleanup as "do what PyTorch does." Wrong reference. PyTorch's
`Dataset` / `DataLoader` solves per-sample iteration + batching + parallel
workers — that's an ML training pipeline. The recsys data-loading problem
is **cohort-wide, one-shot, mixed-source**. Different axis entirely.

The right reference for this codebase is the **Repository pattern** (from
DDD), with a typed `Cohort` return object modeled on **sklearn's `Bunch`**
or **Hugging Face's `Dataset`** — a frozen dataclass exposing named-
attribute access to a fixed set of typed frames.

Why Repository pattern wins for THIS codebase:

1. **Names match the domain.** "Cohort", "Repository". Not "Loader", "Service".
2. **You already use it at the engine layer.** F2 introduced `EngineState`
   Protocol with `DataFrameBackedState` + `DictBackedState` implementations —
   that's the Repository pattern. Extending it one layer up gives architectural
   consistency.
3. **Future-proofs SYNTHETIC_DATA_PLAN.md.** Synthetic data injection
   becomes `SyntheticCohortRepository implements CohortRepository`. Direct
   peer of `MySQLCohortRepository`.
4. **Less ceremony than full DDD.** One Cohort dataclass + one Protocol +
   one concrete implementation today. Add more implementations when needed.

## Symmetry with the existing engine layer

```
        ┌──────────────────────────────────┐
        │  DATA layer  (this refactor)     │
        │                                  │
        │  Protocol:   CohortRepository    │  ← abstract source
        │  Concrete:   MySQLCohortRepo     │  ← production
        │  Future:     SyntheticCohortRepo │  ← when synthetic ships
        │  Return:     Cohort              │  ← typed bundle
        └─────────────────┬────────────────┘
                          │ Cohort
                          ▼
        ┌──────────────────────────────────┐
        │  PIPELINE  (unchanged)           │
        │  DataPipeline.process(cohort)    │
        │  → ScoringOutput (DataFrame)     │
        └─────────────────┬────────────────┘
                          │ ScoringOutput.df
                          ▼
        ┌──────────────────────────────────┐
        │  ENGINE layer  (shipped in F2)   │
        │                                  │
        │  Protocol:   EngineState         │  ← abstract scoring source
        │  Concrete:   DataFrameBackedState│  ← production
        │  Concrete:   DictBackedState     │  ← synthetic
        │  Consumer:   CDSS.recommend      │
        └──────────────────────────────────┘
```

Same pattern (Protocol + 2+ implementations) at every layer. Caller picks
which implementation; engine code unchanged.

## The audit (re-stated, condensed)

Today's data layer:
- **3 files**: `loader.py` (294) + `service.py` (277) + `clinical.py` (80) = 651 L.
- **7 classes**: `DataLoader` + `ProtocolWhitelistService` + `RecommendationDataService`
  + `PPFService` + `ProtocolSimilarityService` + `ClinicalSubscales` + `ProtocolToClinicalMapper`.
- **7 module-level helpers** (`_load_*` × 6 + `load_yaml`).
- **3 layers of indirection** per fetch (Loader method → module helper → stdlib).
- **3 inconsistent patterns** inside `DataLoader` (`_fetch` wrapper / bypass /
  direct module-fn delegation).
- **No unified "Cohort" object** — engine inputs scattered across
  `RawInputs` + `similarity_df` + `attrs["missing_patients"]`.

Six concrete problems detailed in v1, all still apply.

## Target shape

### Two files replace three

```
src/ai_cdss/data.py        ~ 380 L
src/ai_cdss/compute.py     ~ 150 L
```

`loader.py`, `service.py`, `clinical.py` all deleted.

### `data.py` structure

```python
# ╔═══ SECTION 1 — File-IO primitives ═══╗
def read_yaml(path) -> dict: ...
def read_csv(path) -> pd.DataFrame: ...
def read_parquet(path) -> pd.DataFrame: ...
def decode_subscales(row) -> pd.Series: ...      # JSON-encoded subscale unpack
def load_whitelist(path=None) -> list[int]: ...  # replaces ProtocolWhitelistService

# ╔═══ SECTION 2 — Cohort dataclass ═══╗
@dataclass(frozen=True)
class Cohort:
    """Complete data bundle for one recommendation call.

    Modeled on sklearn.Bunch / HuggingFace Dataset — named-attribute
    access to a fixed set of typed frames. Replaces today's scatter of
    (RawInputs, similarity_df, missing_patients_via_attrs).
    """
    patient:      pd.DataFrame
    session:      pd.DataFrame
    ppf:          pd.DataFrame
    similarity:   pd.DataFrame
    whitelist:    list[int]    # already applied; here for audit/trace
    missing_ppf:  list[int]    # PPF gaps that triggered placeholder rows

# ╔═══ SECTION 3 — CohortRepository Protocol ═══╗
@runtime_checkable
class CohortRepository(Protocol):
    """Abstract source of cohorts.

    Mirrors the EngineState Protocol (engine.py § 1) one layer up —
    substrate-agnostic INPUT to the recommendation pipeline.

    Implementations:
      MySQLCohortRepository       production — DB + local files
      SyntheticCohortRepository   future — see SYNTHETIC_DATA_PLAN.md
      InMemoryCohortRepository    tests — pre-built Cohort, no I/O
    """
    def find(self, patient_ids: list[int]) -> Cohort: ...

# ╔═══ SECTION 4 — MySQLCohortRepository (production) ═══╗
class MySQLCohortRepository:
    """Pulls from RGS MySQL via DatabaseInterface;
    reads precomputed PPF + similarity from ~/.ai_cdss/output/."""

    def __init__(
        self,
        db: DatabaseInterface | None = None,
        data_dir: Path | None = None,
        whitelist: list[int] | None = None,
    ): ...

    def find(self, patient_ids: list[int]) -> Cohort:
        """One-shot fetch + filter + assemble. The only public method
        in the protocol contract."""

    # Specialized accessors — used by compute.py functions, NOT part of
    # the CohortRepository protocol contract. Production only.
    def patient_subscales(self, patient_ids: list[int]) -> pd.DataFrame: ...
    def protocol_attributes(self) -> pd.DataFrame: ...

# ╔═══ SECTION 5 — Clinical mappers (moved from clinical.py) ═══╗
class ClinicalSubscales: ...
class ProtocolToClinicalMapper: ...
```

### `compute.py` structure

```python
# Pure functions. No classes. No state. Used by the offline patient-
# registration + protocol-addition workflows, NOT by the recommendation
# pipeline (which reads precomputed PPF/similarity from disk).

# ╔═══ PPF ═══╗
def compute_ppf_for_patients(
    patient_subscales: pd.DataFrame,
    protocol_attributes: pd.DataFrame,
    scales_yaml: Path | None = None,
    mapping_yaml: Path | None = None,
) -> pd.DataFrame:
    """Pure: (patient_subscales, protocol_attributes) → PPF DataFrame."""

def persist_ppf(df, path=None) -> Path:
    """Upsert by (PATIENT_ID, PROTOCOL_ID); creates file if absent."""

# ╔═══ Protocol similarity ═══╗
def compute_protocol_similarity_matrix(
    protocol_attributes: pd.DataFrame,
    mapping_yaml: Path | None = None,
) -> pd.DataFrame:
    """Pure: protocol_attributes → Gower-distance similarity DataFrame."""

def persist_similarity(df, path=None) -> Path:
    """Write to CSV at DEFAULT_OUTPUT_DIR / PROTOCOL_SIMILARITY_CSV."""
```

PPF and Similarity are **computations**, not loads. They take frames in,
emit frames out. The "Service" framing buried this. Pure functions match
the actual semantic.

## Caller migration

### CDSSInterface — shrinks 5 attributes → 2

```python
# BEFORE
class CDSSInterface:
    def __init__(self, loader, pipeline=None, data_service=None,
                 ppf_service=None, debug=False):
        self.loader = loader
        self.pipeline = pipeline or DataPipeline()
        self.ppf_service = ppf_service or PPFService(loader)
        self.data_service = data_service or RecommendationDataService(loader)
        self.protocol_similarity_service = ProtocolSimilarityService(loader)

# AFTER
class CDSSInterface:
    def __init__(self, repository: CohortRepository | None = None,
                 pipeline: DataPipeline | None = None, debug: bool = False):
        self.repository = repository or MySQLCohortRepository()
        self.pipeline = pipeline or DataPipeline()
```

### Recommendation flow

```python
# BEFORE
raw_inputs, similarity = self.data_service.prepare(patient_list)
scores = self.pipeline.process(raw_inputs, scoring_date)
cdss = CDSS(scoring=scores, ...)
result = cdss.recommend(patient_id, similarity)

# AFTER
cohort = self.repository.find(patient_ids)
scores = self.pipeline.process(cohort, scoring_date)
cdss = CDSS(scoring=scores, ...)
result = cdss.recommend(patient_id, cohort.similarity)
```

### PPF computation (registration workflow)

```python
# BEFORE
service = PPFService(loader)
result = service.compute_and_persist_patient_fit([pid])

# AFTER
from ai_cdss.data import MySQLCohortRepository
from ai_cdss.compute import compute_ppf_for_patients, persist_ppf
repo = MySQLCohortRepository()
subscales = repo.patient_subscales([pid])
attributes = repo.protocol_attributes()
ppf = compute_ppf_for_patients(subscales, attributes)
persist_ppf(ppf)
```

Verbose? Slightly. But every step does exactly one thing visibly. The old
`compute_and_persist_patient_fit` hid 5 I/O operations behind one method.

## Numbers

| Metric | Today | v2 |
|---|---|---|
| Files for data loading | 3 | 2 |
| Classes | 7 | 3 (MySQLCohortRepository + 2 mappers) |
| Protocols | 0 | 1 (CohortRepository) |
| Module helpers | 7 | 5 (3 IO + load_whitelist + decode_subscales) |
| Compute functions | (buried in 2 services) | 4 (compute + persist × 2) |
| Layers of indirection per fetch | 3 | 1 |
| CDSSInterface attributes for data | 5 | 2 |

## Phasing

| Phase | Scope | Effort |
|---|---|---|
| **f5a** | Add `data.py` with `Cohort` + `CohortRepository` Protocol + `MySQLCohortRepository`. Keep old `loader.py` / `service.py` alongside. Update `CDSSInterface` to use the new path. Tests pass. | 4-6 h |
| **f5b** | Add `compute.py` with the 4 standalone functions. Update ai-cdss-cli usage sites. | 2-3 h |
| **f5c** | Delete `loader.py`, `service.py`, `clinical.py`. Fold clinical mappers into `data.py § 5`. | 1 h |
| **f5d** | Update `docs/architecture.md`, `docs/code_structure.md`, `docs/class_diagram.{md,html}`, `docs/dataflow.md`. | 30 min |

Total: 8-10 h, single branch. 83/83 tests green at every commit.

## Open questions

1. **Where do `ClinicalSubscales` / `ProtocolToClinicalMapper` live?**
   They're transform classes used by PPF computation. Either:
     a) inside `data.py` (as utilities for the loader's `patient_subscales` /
        `protocol_attributes` accessors)
     b) inside `compute.py` (since PPF computation owns them)
   **Lean**: (a). They map "raw frame → normalized frame" — that's loading-
   adjacent, not computation.

2. **Does `Cohort` carry similarity?** Yes. The current 2-tuple
   `(RawInputs, similarity)` is an unnecessary seam. Carrying everything on
   `Cohort` removes it. Pipeline takes `Cohort`, uses `.patient` /
   `.session` / `.ppf`; engine takes `cohort.similarity` alongside scoring
   output.

3. **Should `CohortRepository.find` be cached?**
   The cdss-supervisor backtest sweep makes 150+ repeated calls. Currently
   `replay_cdss.build_replay_context` hoists the load outside the loop.
   Options:
     a) Don't cache. Caller hoists when needed (today's pattern).
     b) Cache per-repository-instance via `@lru_cache` or a manual dict.
   **Lean**: (a). Caching belongs at the call site that knows the access
   pattern, not in the data layer.

4. **Back-compat shim for ai-cdss-cli's `PPFService` import?**
   Two options:
     a) Ship `service.py` with deprecation-warning shims that forward to
        `compute.py` functions.
     b) Coordinated release — update ai-cdss-cli to use the new functions
        in the same change.
   **Lean**: (b). The branch already ripped v0.3.1 back-compat in F0;
   keeping shims defeats that.

5. **Should there be an `InMemoryCohortRepository` for tests?**
   Yes — define alongside `MySQLCohortRepository` in `data.py § 4.5`. Takes
   a `Cohort` at construction, `find()` returns it regardless of `patient_ids`.
   Useful for unit tests that need to exercise the full pipeline without DB.
   ~20 lines.

6. **Do we need a registry / factory pattern?**
   I.e., `repository = build_cohort_repository(source="mysql")` dispatched
   by string. Not yet — there's one implementation today (and one planned).
   YAGNI; revisit when a third implementation arrives.

## What this does NOT do

- Doesn't change the engine.
- Doesn't change recommendation behavior.
- Doesn't change `DataPipeline`'s internal stages — only its input parameter
  type (RawInputs → Cohort).
- Doesn't redesign `RawInputs` / `PreparedInputs` / `MergedFeatures` / etc.
  (the pipeline-internal contracts). Those stay.
- Doesn't address the pandera schema question (separate decision).
- Doesn't address the `ScoringOutput.df` audit gap (covered by
  `DATA_CONTRACTS_PLAN.md`).
- Doesn't ship the synthetic-data piece (covered by
  `SYNTHETIC_DATA_PLAN.md` — but this refactor makes it a one-liner addition).

## Status

- [x] Audit (v1 + v2)
- [x] Pattern choice (Repository, not PyTorch-Loader)
- [ ] Resolve 6 open questions above
- [ ] Implement f5a
- [ ] Implement f5b
- [ ] Implement f5c
- [ ] Implement f5d
- [ ] Run 83-test suite — must stay green at every commit
