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
                  Section 1 ─────┤  metrics.py
                  time axis      │    EWMA / Savgol / Theil-Sen primitives
                                 ▼
              (patient × protocol × session × time)
                                 │
                  Section 3 ─────┤  metrics.py
                  reduce time    │    build_delta_dm
                                 │    build_recent_adherence
                                 ▼
              (patient × protocol × session_date)
                                 │
                  Section 4 ─────┤  metrics.py
                  reduce session │    build_usage, build_week_usage
                                 │    build_prescription_days
                                 ▼
                    (patient × protocol)
                                 │
                  Section 5 ─────┤  metrics.py
                  patient scalar │    build_week_since_start
                                 ▼
                       (patient)
                                 │
                  Section 6 ─────┤  metrics.py
                  cross-cohort   │    compute_ppf
                                 │    compute_protocol_similarity
                                 ▼
                  (patient × protocol  pairs)
                  (protocol × protocol  pairs)
```

## File layout

10 modules at `src/ai_cdss/` root + `interface/` subpackage:

```
src/ai_cdss/
├── __init__.py            (12   — public API: CDSS)
├── constants.py           (158  — column names, axis defs, thresholds)
├── data.py                (501  — Cohort + CohortRepository + RGSCohortRepository)
├── engine.py              (604  — EngineState protocol + adapters)
├── metrics.py             (556  — feature reductions over the tensor axes)
├── interface/             (635  — CDSS + DebugReport)
├── precompute.py          (160  — PPF + similarity offline computations)
├── recommender.py         (783  — strategies + MVT + substitute + topup + Recommender)
├── scoring.py             (540  — typed contracts + Imputer + Scorer + DataPipeline)
└── utils.py               (107  — MultiKeyDict + small helpers)
                           ─────
                           ~4 056
```

Plus `config/` (YAML configs) and `resources/` (embedded CSV — namely
`protocol_attributes.csv`).

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
   │  │ Strategy dispatch:                                         │  │
   │  │   _bootstrap_strategy  (no prior)                          │  │
   │  │   _repeat_strategy     (week skipped — USAGE_WEEK=0)       │  │
   │  │   _update_strategy     (MVT swap loop)                     │  │
   │  ├────────────────────────────────────────────────────────────┤  │
   │  │ _top_up_schedule     (universal top-up post-step)          │  │
   │  ├────────────────────────────────────────────────────────────┤  │
   │  │ Trace dict attached to .attrs["trace"]                     │  │
   │  └────────────────────────────────────────────────────────────┘  │
   └───────────────────────────────┬──────────────────────────────────┘
                                   │  recommendations DataFrame + trace
                                   ▼
                          ┌────────────────────┐
                          │  to caller         │
                          │  (CDSS,   │
                          │   replay engine,   │
                          │   backtest sweep)  │
                          └────────────────────┘
```

## Typed contracts at every pipeline boundary

Each stage's input and output is wrapped in a frozen dataclass declared
in `scoring.py` SECTION 1. Construction validates required columns —
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

`recommender.py` is one file with 9 banner-delimited sections that map
1:1 to the algorithm. Read top-to-bottom:

```
1.  trace               — trace dict construction helpers
2.  bootstrap strategy  — first-week schedule (top-N + round-robin)
3.  repeat strategy     — week skipped → copy prior unchanged
4.  MVT swap criterion  — below-mean selection (strict <, prescribed-mean)
5.  similarity queries  — slice / rank protocol-similarity table
6.  substitute search   — two-tier (unused / least-used-similar)
7.  update strategy     — swap loop assembly
8.  top-up schedule     — fill 7×ppd grid (existing → top_pool → exhausted)
9.  CDSS orchestrator   — entry-point class — dispatches strategies via `_run_strategy`
```

Each section banner is a CSS-style box (`╔═...═╗`) — visible in any
editor with monospace fonts.

## Where to look for what

| Question | File / Section |
|---|---|
| What columns does the scoring DataFrame have? | `scoring.py` § 1 (`ScoringOutput.REQUIRED`) |
| Where does DELTA_DM come from? | `metrics.py` § 3 (`build_delta_dm`) |
| How is the prescribed-days window computed? | `metrics.py` § 5 (`_last_completed_week_window`) |
| What does the MVT criterion test? | `recommender.py` § 5 (`_below_mean_protocols`) |
| Why is a substitute picked? | `recommender.py` § 7 (`_find_substitute`) |
| What does top-up do to the schedule? | `recommender.py` § 8 (`_top_up_schedule`) |
| How does the engine know if a patient has prior? | `recommender.py` § 1 (`PatientState.prescriptions`) |
| Where does PPF come from? | `precompute.py` § 1 (`compute_ppf_for_patients`) |
| What's in the trace? | `recommender.py` § 2 (`_init_trace`, `_serialize_*`) |

## Backward compatibility

The v0.3.1 back-compat shims were all retired during the F0-F5
refactor. The single public entry is `from ai_cdss import CDSS`
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
| 2 | ✓ done | Single `recommender.py` with 10 section banners. |
| 3 | ✓ done | `processing/` flattened to `metrics.py`, `scoring.py`, `scoring.py` at root. |
| 4 | ✓ done | `loaders/` + `services/` flattened to `loader.py` + `service.py`. |
| 5 | ✓ done | Repository-pattern data layer: `loader.py` + `service.py` + `clinical.py` replaced by `data.py` (Cohort + CohortRepository + RGSCohortRepository) + `precompute.py` (4 pure functions for offline PPF / similarity). |
