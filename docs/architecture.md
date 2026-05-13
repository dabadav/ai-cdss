# `ai-cdss` Architecture

Sibling project: `D:/Projects/RGS/ai-cdss/` (original). This refactor
focuses on **readability** — same algorithm, same behavior, same tests,
flat layout.

## The mental model — a tensor with aggregation levels

Data is a tensor with axes:

    patient × protocol × prescription × session × time → values

Every operation in this package is a **reduction over one or more
axes** of that tensor. The file layout mirrors those reductions: each
file contains operations that live at one aggregation level (or
collapse from one level to another).

```
              (patient × protocol × prescription × session × time)
                                 │
                  Section 1 ─────┤  feature.py
                  time axis      │    EWMA / Savgol / Theil-Sen primitives
                                 ▼
              (patient × protocol × session × time)
                                 │
                  Section 3 ─────┤  feature.py
                  reduce time    │    build_delta_dm
                                 │    build_recent_adherence
                                 ▼
              (patient × protocol × session_date)
                                 │
                  Section 4 ─────┤  feature.py
                  reduce session │    build_usage, build_week_usage
                                 │    build_prescription_days
                                 ▼
                    (patient × protocol)
                                 │
                  Section 5 ─────┤  feature.py
                  patient scalar │    build_week_since_start
                                 ▼
                       (patient)
                                 │
                  Section 6 ─────┤  feature.py
                  cross-cohort   │    compute_ppf
                                 │    compute_protocol_similarity
                                 ▼
                  (patient × protocol  pairs)
                  (protocol × protocol  pairs)
```

## File layout

11 modules at `src/ai_cdss/` root (`data.py` is a package because the
embedded `protocol_attributes.csv` lives alongside it):

```
src/ai_cdss/
├── __init__.py            (22   — public API: CDSSInterface + schemas)
├── compute.py             (160  — PPF + similarity offline computations)
├── constants.py           (158  — column names, axis defs, thresholds)
├── data/__init__.py       (503  — Cohort + CohortRepository + RGSCohortRepository)
├── engine.py              (604  — EngineState protocol + adapters)
├── feature.py             (556  — feature reductions over the tensor axes)
├── interface/             (635  — CDSSInterface + DebugReport)
├── models.py              (113  — pandera schemas, documentation-only)
├── pipeline.py            (446  — typed contracts + DataPipeline orchestrator)
├── recommend.py           (783  — branches + MVT + substitute + topup + CDSS)
├── score.py               (99   — Imputer + Scorer)
└── utils.py               (107  — MultiKeyDict + small helpers)
                           ─────
                           ~4 186
```

Plus `config/` (YAML configs) and the rest of `data/` (embedded CSV
resources alongside `data/__init__.py`).

## Dataflow — from raw DB rows to a recommendation

```
                          ┌──────────────────────────┐
                          │  rgs-interface MySQL     │
                          │  (sessions, patient,     │
                          │  prescription_plus)      │
                          └────────────┬─────────────┘
                                       │
                ┌──────────────────────▼──────────────────────┐
                │  data.RGSCohortRepository.find              │
                │    fetch patient + session via DB           │
                │    read PPF parquet + similarity csv        │
                │    apply protocol whitelist                 │
                │    return Cohort (patient / session / ppf / │
                │                   similarity / whitelist /  │
                │                   missing_ppf)              │
                └──────────────────────┬──────────────────────┘
                                       │  Cohort
                                       ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │  pipeline.DataPipeline.process                                   │
   │  ┌────────────────────────────────────────────────────────────┐  │
   │  │ Stage 1  _prepare        →  PreparedInputs                 │  │
   │  │   (clean + window the three frames)                        │  │
   │  ├────────────────────────────────────────────────────────────┤  │
   │  │ Stage 2  _build_features →  MergedFeatures                 │  │
   │  │   2a  _session_level_features  →  SessionLevelFeatures     │  │
   │  │       (recent_adherence + delta_dm)                        │  │
   │  │   2b  _protocol_level_features →  ProtocolLevelFeatures    │  │
   │  │       (usage + week_usage + prescription_days +            │  │
   │  │        week_since_start)                                   │  │
   │  │   2c  _broadcast_session_onto_protocol                     │  │
   │  ├────────────────────────────────────────────────────────────┤  │
   │  │ Stage 3  _impute_features →  ScoringInput                  │  │
   │  │   (collapse to one row per (PP) via groupby-last,          │  │
   │  │    fill NaN with per-patient median)                       │  │
   │  ├────────────────────────────────────────────────────────────┤  │
   │  │ Stage 4  _score           →  ScoringOutput                 │  │
   │  │   (RECENT_ADHERENCE·w0 + DELTA_DM·w1 + PPF·w2)             │  │
   │  └────────────────────────────────────────────────────────────┘  │
   └───────────────────────────────┬──────────────────────────────────┘
                                   │  scoring DataFrame
                                   ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │  recommend.CDSS.recommend                                        │
   │  ┌────────────────────────────────────────────────────────────┐  │
   │  │ PatientState — patient-scoped view of scoring              │  │
   │  ├────────────────────────────────────────────────────────────┤  │
   │  │ Branch dispatch:                                           │  │
   │  │   _bootstrap_branch    (no prior)                          │  │
   │  │   _repeat_branch       (week skipped — USAGE_WEEK=0)       │  │
   │  │   _update_branch       (MVT swap loop)                     │  │
   │  ├────────────────────────────────────────────────────────────┤  │
   │  │ _fill_grid_coverage  (universal top-up post-step)          │  │
   │  ├────────────────────────────────────────────────────────────┤  │
   │  │ Trace dict attached to .attrs["trace"]                     │  │
   │  └────────────────────────────────────────────────────────────┘  │
   └───────────────────────────────┬──────────────────────────────────┘
                                   │  recommendations DataFrame + trace
                                   ▼
                          ┌────────────────────┐
                          │  to caller         │
                          │  (CDSSInterface,   │
                          │   replay engine,   │
                          │   backtest sweep)  │
                          └────────────────────┘
```

## Typed contracts at every pipeline boundary

Each stage's input and output is wrapped in a frozen dataclass declared
in `pipeline.py` SECTION 1. Construction validates required columns —
fail-fast at the boundary instead of cryptic KeyErrors deep in a
groupby.

```
PreparedInputs           patient / session / ppf
   ↓
SessionLevelFeatures     BY_PP + [SESSION_DATE, RECENT_ADHERENCE, DELTA_DM]
ProtocolLevelFeatures    BY_PP + [PPF, USAGE, USAGE_WEEK, DAYS, WEEKS_SINCE_START]
   ↓ merge
MergedFeatures           union of the above
   ↓ groupby-last + impute
ScoringInput             BY_PP + [PPF, DELTA_DM, RECENT_ADHERENCE, USAGE,
                                  USAGE_WEEK, DAYS, WEEKS_SINCE_START]
   ↓ Scorer
ScoringOutput            ScoringInput columns + SCORE
```

Skip validation in hot paths with `validate_on_init=False`.

## The recommendation algorithm — section map

`recommend.py` is one file with 10 banner-delimited sections that map
1:1 to the algorithm. Read top-to-bottom:

```
1.  PatientState        — patient-scoped scoring view
2.  trace               — trace dict construction helpers
3.  bootstrap branch    — first-week schedule (top-N + round-robin)
4.  repeat branch       — week skipped → copy prior unchanged
5.  MVT swap criterion  — below-mean selection (strict <, prescribed-mean)
6.  similarity queries  — slice / rank protocol-similarity table
7.  substitute search   — two-tier (unused / least-used-similar)
8.  update branch       — swap loop assembly
9.  top-up coverage     — fill 7×ppd grid (existing → top_pool → exhausted)
10. CDSS orchestrator   — entry-point class
```

Each section banner is a CSS-style box (`╔═...═╗`) — visible in any
editor with monospace fonts.

## Where to look for what

| Question | File / Section |
|---|---|
| What columns does the scoring DataFrame have? | `pipeline.py` § 1 (`ScoringOutput.REQUIRED`) |
| Where does DELTA_DM come from? | `feature.py` § 3 (`build_delta_dm`) |
| How is the prescribed-days window computed? | `feature.py` § 5 (`_last_completed_week_window`) |
| What does the MVT criterion test? | `recommend.py` § 5 (`_below_mean_protocols`) |
| Why is a substitute picked? | `recommend.py` § 7 (`_find_substitute`) |
| What does top-up do to the grid? | `recommend.py` § 9 (`_fill_grid_coverage`) |
| How does the engine know if a patient has prior? | `recommend.py` § 1 (`PatientState.prescriptions`) |
| Where does PPF come from? | `compute.py` § 1 (`compute_ppf_for_patients`) |
| What's in the trace? | `recommend.py` § 2 (`_init_trace`, `_serialize_*`) |

## Backward compatibility

The v0.3.1 back-compat shims were all retired during the F0-F5
refactor. The single public entry is `from ai_cdss import CDSSInterface`
(plus the three pandera schemas — also re-exported at the package root).
Internal callers (e.g. ai-cdss-cli, cdss-supervisor) coordinate via
versioned releases rather than import-path shims.

## Tests

83 unit tests at `tests/unit/`. Run with:
```bash
PYTHONPATH=src python -m pytest tests/unit/
```

## Refactor phase history

| Phase | Status | Notes |
|---|---|---|
| 1 | ✓ reverted | Split `cdss.py` into a `recommend/` subpackage of 10 files. Over-fragmented; rolled back in phase 2. |
| 2 | ✓ done | Single `recommend.py` with 10 section banners. |
| 3 | ✓ done | `processing/` flattened to `feature.py`, `score.py`, `pipeline.py` at root. |
| 4 | ✓ done | `loaders/` + `services/` flattened to `loader.py` + `service.py`. |
| 5 | ✓ done | Repository-pattern data layer: `loader.py` + `service.py` + `clinical.py` replaced by `data/__init__.py` (Cohort + CohortRepository + RGSCohortRepository) + `compute.py` (4 pure functions for offline PPF / similarity). |
