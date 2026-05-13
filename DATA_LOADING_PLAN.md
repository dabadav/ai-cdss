# Data Loading Refactor — PyTorch-style consolidation

Status: **inspection + proposal**. No code changes yet.

## The audit — what's actually wrong today

Data loading lives in **3 files, 7 classes, 7 module-level helpers** for what is fundamentally:

> "Fetch 4 DataFrames (patient / session / ppf / similarity) and run 2 compute functions (PPF, similarity)."

### Files involved

| File | LOC | Contents |
|---|---|---|
| `loader.py` | 294 | 6 module helpers + `DataLoader` class with 7 methods |
| `service.py` | 277 | `load_yaml` + 4 service classes |
| `clinical.py` | 80 | 2 classes that read YAML configs + transform frames |
| | **651** | total |

### The call graph

```
CDSSInterface.recommend_for_patients
  ↓
RecommendationDataService.prepare(patient_list)         (service.py)
  ↓
  ├─ DataLoader.load_ppf_data(patient_list)             (loader.py)
  │    ↓ DataLoader._fetch(...)                          (loader.py)
  │    ↓ _load_ppf_data(patient_list)                    (loader.py — module fn)
  │    ↓ pd.read_parquet                                 (stdlib)
  │
  ├─ DataLoader.load_session_data(patient_list)
  │    ↓ DataLoader._fetch(...)
  │    ↓ interface.fetch_rgs_data                        (rgs_interface)
  │
  ├─ DataLoader.load_patient_data(patient_list)
  │    ↓ DataLoader._fetch(...)
  │    ↓ interface.fetch_clinical_data                   (rgs_interface)
  │
  ├─ DataLoader.load_protocol_similarity()
  │    ↓ _load_protocol_similarity()                     (module fn)
  │    ↓ pd.read_csv
  │
  ├─ ProtocolWhitelistService().load_whitelist()
  │    ↓ load_yaml(...)                                  (service.py)
  │
  ↓
assemble RawInputs (pipeline.py)
```

### Six concrete problems

1. **Three layers of indirection per fetch**.
   `DataLoader.load_ppf_data` → `_load_ppf_data` (module fn) → `pd.read_parquet`.
   Each layer adds nothing. The class method just delegates to the module function of similar name.

2. **Inconsistent patterns inside `DataLoader`**.
   - Some methods route through `_fetch` (SchemaError → empty frame).
   - `load_patient_subscales` bypasses `_fetch` entirely (calls `load_patient_data` then decodes).
   - `load_protocol_attributes` / `load_protocol_similarity` call module functions directly.
   Three different patterns. Reader has to inspect each.

3. **`ProtocolWhitelistService` is a 14-line class for a single YAML read**.
   `load_whitelist()` reads YAML, extracts one key. Could be one module function.

4. **Two "computation services" that hold a `DataLoader`** but only use 1-2 of its methods each.
   - `PPFService` uses `load_patient_subscales` + `load_protocol_attributes`.
   - `ProtocolSimilarityService` uses `load_protocol_attributes`.
   They're really pure computations + I/O wrappers. Calling them "services" implies stateful orchestration that isn't there.

5. **No unified "Cohort" object**.
   The data the pipeline needs is 4 frames. They're loaded by 4 separate methods, on 4 separate calls, then assembled into `RawInputs` (which only holds 3 of them — similarity is passed separately). Five places to look.

6. **`load_yaml` is in `service.py`**.
   It's a utility. Belongs in `utils.py` or wherever the YAML producers live. Currently buried inside a "services" module.

## The PyTorch parallel

PyTorch separates three concerns cleanly:

| PyTorch | What it does | RGS analog today | RGS analog ideal |
|---|---|---|---|
| `Dataset` | "I represent the data. Ask me for item i." | scattered: loader + 4 services | **`Cohort` + `CohortLoader`** |
| `DataLoader` | "I iterate / batch / shuffle." | `DataPipeline` (transforms inputs → scores) | `DataPipeline` (already good) |
| `transforms` | "I preprocess one sample." | `Imputer`, `Scorer`, feature builders | (already good) |

The recommendation engine doesn't need to iterate / batch / shuffle (cohort-wide ops are one-shot). So the "DataLoader" role maps onto `DataPipeline`. The "Dataset" role is what's currently spread across loader.py + service.py + clinical.py.

**Target shape**: one Cohort dataclass + one CohortLoader class. The engine asks the loader for a cohort; the loader knows where data lives and returns a typed object. Pipeline + engine downstream are substrate-agnostic.

## Proposed structure

### Two files replace three

```
src/ai_cdss/data.py        ~ 360 lines    (was loader.py 294 + service.py orchestration ~80 + clinical.py 80)
src/ai_cdss/compute.py     ~ 150 lines    (was service.py compute ~190)
```

`loader.py`, `service.py`, `clinical.py` all deleted.

### `data.py` — the canonical data-loading file

Section-banner organized:

```python
# ╔═══ SECTION 1 — File-IO primitives ═══╗
def read_yaml(path) -> dict: ...
def read_csv(path) -> pd.DataFrame: ...
def read_parquet(path) -> pd.DataFrame: ...
def decode_subscales(row, ...) -> pd.Series: ...    # JSON-encoded subscale unpack

# ╔═══ SECTION 2 — Cohort dataclass (what the engine needs) ═══╗
@dataclass(frozen=True)
class Cohort:
    """One cohort's worth of data — typed bundle the pipeline consumes.

    Replaces the current scatter of (RawInputs, similarity_df,
    missing_patients) passed around as a 3-tuple."""
    patient:               pd.DataFrame
    session:               pd.DataFrame
    ppf:                   pd.DataFrame
    similarity:            pd.DataFrame
    whitelist:             list[int]            # already applied; here for audit
    missing_ppf_patients:  list[int]            # PPF gaps that triggered placeholders

# ╔═══ SECTION 3 — CohortLoader (PyTorch Dataset analog) ═══╗
class CohortLoader:
    """One entry point for all data loading.

    Equivalent of PyTorch's Dataset: encapsulates where the data
    lives and how to fetch it. The pipeline + engine downstream
    don't care about the source.
    """
    def __init__(
        self,
        db: DatabaseInterface | None = None,
        data_dir: Path | None = None,
        whitelist: list[int] | None = None,
    ): ...

    def load(self, patient_ids: list[int]) -> Cohort:
        """One-shot fetch + filter + assemble."""

    def patient_subscales(self, patient_ids: list[int]) -> pd.DataFrame:
        """Specialized — for PPF computation only (not part of the
        normal pipeline)."""

    def protocol_attributes(self) -> pd.DataFrame:
        """Specialized — for PPF + similarity computation."""

# ╔═══ SECTION 4 — Whitelist + config helpers ═══╗
def load_whitelist(path: Path | None = None) -> list[int]: ...
# Single function, replaces ProtocolWhitelistService class

# ╔═══ SECTION 5 — Subscale + attribute mappers (from clinical.py) ═══╗
class ClinicalSubscales: ...
class ProtocolToClinicalMapper: ...
```

### `compute.py` — pure computation, no orchestration

```python
# ╔═══ PPF ═══╗
def compute_ppf_for_patients(
    patient_subscales: pd.DataFrame,
    protocol_attributes: pd.DataFrame,
    scales_yaml: Path | None = None,
    mapping_yaml: Path | None = None,
) -> pd.DataFrame:
    """Pure function. Patient deficit × protocol attributes → PPF."""

def persist_ppf(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Upsert-style write to Parquet."""

# ╔═══ Protocol similarity ═══╗
def compute_protocol_similarity_matrix(
    protocol_attributes: pd.DataFrame,
    mapping_yaml: Path | None = None,
) -> pd.DataFrame:
    """Pure function. Gower over attribute embeddings."""

def persist_similarity(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Write to CSV in the default output dir."""
```

No classes. The "PPFService" and "ProtocolSimilarityService" were stateful wrappers around pure functions. The state (loader reference, optional YAML paths) is now passed as function arguments.

## Wiring after the refactor

`CDSSInterface.__init__` shrinks from:

```python
self.loader = loader
self.pipeline = pipeline or DataPipeline()
self.ppf_service = ppf_service or PPFService(loader)
self.data_service = data_service or RecommendationDataService(loader)
self.protocol_similarity_service = ProtocolSimilarityService(loader)
```

to:

```python
self.cohort_loader = cohort_loader or CohortLoader()
self.pipeline = pipeline or DataPipeline()
```

`_recommend_for_patients_core` shrinks from:

```python
raw_inputs, protocol_similarity = self.data_service.prepare(patient_list=patient_ids)
scores = self.pipeline.process(raw_inputs, scoring_date or pd.Timestamp.today())
```

to:

```python
cohort = self.cohort_loader.load(patient_ids)
scores = self.pipeline.process(cohort, scoring_date or pd.Timestamp.today())
```

(Pipeline's `process` accepts `Cohort` instead of `RawInputs`. Internally it uses the same patient/session/ppf fields — drop the `similarity` field since pipeline doesn't touch it; pipeline-output then handed to engine alongside `cohort.similarity`.)

PPF computation (when registering a new patient) becomes:

```python
# OLD
PPFService(loader).compute_and_persist_patient_fit([pid])
# NEW
from ai_cdss.compute import compute_ppf_for_patients, persist_ppf
from ai_cdss.data import CohortLoader
loader = CohortLoader()
subscales = loader.patient_subscales([pid])
attributes = loader.protocol_attributes()
ppf = compute_ppf_for_patients(subscales, attributes)
persist_ppf(ppf)
```

Verbose? Slightly. But each step is doing exactly one thing visibly. The previous chain `PPFService(loader).compute_and_persist_patient_fit([pid])` hid 5 different I/O operations behind one method call.

## What the cleanup wins

| Metric | Before | After |
|---|---|---|
| Files for data loading | 3 (loader.py, service.py, clinical.py) | 2 (data.py, compute.py) |
| Classes | 7 (DataLoader + 4 services + 2 mappers) | 3 (CohortLoader + 2 mappers) |
| Module-level fns | 7 (`_load_*` × 6 + `load_yaml`) | 3 (read_yaml/csv/parquet) + 4 compute fns |
| Layers of indirection per fetch | 3 (Loader method → module helper → stdlib) | 1 (CohortLoader.load → stdlib) |
| What CDSSInterface holds | loader + 3 services + pipeline | cohort_loader + pipeline |
| Where the "what does the engine consume" contract lives | scattered (RawInputs + similarity_df + missing_patients tuple) | `Cohort` dataclass (one place) |
| `loader.py` lines | 294 | merged into data.py |
| `service.py` lines | 277 | split: 80 → data.py orchestration, 100 → compute.py, 14 (whitelist) → 1 function, rest dropped |
| `clinical.py` lines | 80 | merged into data.py |

## Migration risks

1. **`PPFService.compute_and_persist_patient_fit([pid])` is a documented public API**. The ai-cdss-cli probably uses it. Need a back-compat shim or to update the cli.
2. **`RecommendationDataService.prepare` return signature changes** from `(RawInputs, similarity_df)` to `Cohort`. Pipeline's `process` signature has to change too. Tests will need updates. Behavior preserved.
3. **`ProtocolWhitelistService` deletion** — anyone else constructing it? Grep says no, but worth a final check.

## Phasing options

| Phase | Scope | Effort |
|---|---|---|
| **f5a** | Introduce `Cohort` dataclass + `CohortLoader` class alongside existing service classes. Both work. Update CDSSInterface to use CohortLoader. Tests pass. | 4-6 hours |
| **f5b** | Delete `RecommendationDataService` + `PPFService` + `ProtocolSimilarityService` + `ProtocolWhitelistService`. Update or delete the ai-cdss-cli usage of PPF service. | 2-3 hours |
| **f5c** | Fold `clinical.py` into `data.py`. Fold `compute.py` functions out of (now-deleted) `service.py`. Final file structure. | 1 hour |
| **f5d** | Update `docs/architecture.md`, `docs/class_diagram.md`, `docs/dataflow.md` to reflect the new shape. | 30 min |

Total: ~8-10 hours. Single branch. 83/83 tests must still pass at every commit.

## My recommendation

Ship **f5a + f5b** as ONE commit (`f5: PyTorch-style data loading — Cohort + CohortLoader + compute.py`). The intermediate "both shapes coexist" state is awkward and tempts deferral. Single commit, clean cut, every caller updated.

Defer **f5c/f5d** to a follow-up if there's caller-migration friction.

## Open questions before starting

1. **Where does subscales-decoding live?** Currently `_decode_subscales` (loader.py) processes the JSON-encoded `CLINICAL_SCORES` column. It's used inside `DataLoader.load_patient_subscales`. Stays a module helper in `data.py § 1`? Yes — file-IO primitive.
2. **Does `Cohort` carry `similarity_df` or is that passed separately?**
   - YES, carry it on `Cohort`. The current 2-tuple `(RawInputs, similarity)` is the seam between service + pipeline; carrying everything on `Cohort` removes that seam.
   - Pipeline's `process` then needs `cohort.similarity` available — but pipeline doesn't USE similarity (it just builds the scoring frame). CDSSInterface passes `cohort.similarity` to the engine alongside the scoring frame. The pipeline can either ignore the field or take just the (patient, session, ppf) it needs.
3. **What about the `ClinicalSubscales` / `ProtocolToClinicalMapper` YAML config injection?** They take optional `*_yaml_path` parameters. Stays as constructor args. The PPF compute function in `compute.py` accepts the same paths and forwards.
4. **Back-compat for ai-cdss-cli?** The cli imports `PPFService` and calls `compute_and_persist_patient_fit`. Options:
   - Ship a deprecated `PPFService` shim that forwards to the new `compute.py` functions.
   - Update the cli alongside ai-cdss (preferred — one coordinated release).
5. **Should `CohortLoader` cache anything?** Today's `DataLoader` makes a fresh DB roundtrip per call. The cdss-supervisor backtest sweep made this expensive (hoisted out via `replay_cdss.build_replay_context`). Worth baking a per-instance LRU cache into `CohortLoader.load`? Likely yes, but separable from this refactor.

## What this does NOT do

- Doesn't touch the engine.
- Doesn't change behavior of any recommendation.
- Doesn't redesign the `RawInputs`/`PreparedInputs` pipeline contracts — those stay (or RawInputs becomes a slice of `Cohort`).
- Doesn't address the pandera schemas question (separate decision).
- Doesn't unify with the `SYNTHETIC_DATA_PLAN.md` — but `CohortLoader` is the abstraction that the future `SyntheticDataService` should implement. Worth a Protocol declaration: `class CohortSource(Protocol): def load(ids) -> Cohort: ...`. Defer the Protocol to when there's a second implementation.

## Status

- [ ] Resolve open questions above
- [ ] Implement Cohort + CohortLoader in `data.py`
- [ ] Implement `compute.py` with 4 standalone functions
- [ ] Update CDSSInterface + DataPipeline to consume `Cohort`
- [ ] Update ai-cdss-cli (or add shim)
- [ ] Run 83-test suite — must stay green
- [ ] Update docs
