# Data Contracts — Current State + Options

Status: **discussion document, no solution committed yet**.

Goal: lay out every data shape that crosses a stage boundary today,
mark each one as strongly / weakly / un-typed, and surface the audit
gaps. Once we agree on the picture we can decide which fixes to ship.

---

## A · Inventory of every shape in the package

13 named shapes + a few untyped DataFrames. Grouped by where they live.

### Loader → Service boundary

| Shape | Location | Type | Producer | Consumer |
|---|---|---|---|---|
| `pd.DataFrame` patient | `loader.DataLoader.load_patient_data` | untyped DataFrame | DataLoader | RecommendationDataService |
| `pd.DataFrame` session | `loader.DataLoader.load_session_data` | untyped DataFrame (pandera `SessionSchema` exists but not validated) | DataLoader | RecommendationDataService |
| `pd.DataFrame` ppf | `loader.DataLoader.load_ppf_data` | untyped DataFrame + `attrs["missing_patients"]` side-channel | DataLoader | RecommendationDataService |
| `pd.DataFrame` similarity | `loader.DataLoader.load_protocol_similarity` | untyped (no schema declared) | DataLoader | RecommendationDataService → engine |

### Service → Pipeline boundary

| Shape | Location | Type | Producer | Consumer |
|---|---|---|---|---|
| `RawInputs(patient, session, ppf)` | `pipeline.py § 1` | **frozen dataclass** (no validation) | RecommendationDataService.prepare | DataPipeline.process |
| `pd.DataFrame` similarity (filtered) | passed alongside RawInputs | untyped | service.prepare | engine via CDSS.recommend |

### Pipeline internal boundaries (added in earlier refactor)

| Shape | Location | Type | Producer | Consumer |
|---|---|---|---|---|
| `PreparedInputs(patient, session, ppf, validate_on_init=True)` | `pipeline.py § 1` | **frozen dataclass with column validation** | `_prepare` | `_build_features` / `_bootstrap_scoring_input` |
| `SessionLevelFeatures(df)` | `pipeline.py § 1` | frozen dataclass + required columns | `_session_level_features` | `_broadcast_session_onto_protocol` |
| `ProtocolLevelFeatures(df)` | `pipeline.py § 1` | frozen dataclass + required columns | `_protocol_level_features` | `_broadcast_session_onto_protocol` |
| `MergedFeatures(df)` | `pipeline.py § 1` | frozen dataclass + required columns | `_broadcast_session_onto_protocol` | `_impute_features` |
| `ScoringInput(df)` | `pipeline.py § 1` | frozen dataclass + required columns | `_impute_features` / `_bootstrap_scoring_input` | `_score` |
| `ScoringOutput(df)` | `pipeline.py § 1` | frozen dataclass + required columns | `_score` | engine (via DataFrameBackedState) |

### Engine boundary

| Shape | Location | Type | Producer | Consumer |
|---|---|---|---|---|
| `EngineState` | `engine.py` | **Protocol (PEP 544 structural)** | DataFrameBackedState / DictBackedState | CDSS.recommend internals |
| `SimilarityMatrix` | `engine.py` | Protocol | DataFrameSimilarity / DictSimilarity | _find_substitute / similarity helpers |
| `ProtocolRow` | `engine.py` | **frozen dataclass** | EngineState.score_row, scoring rows | engine internals (list[ProtocolRow]) |
| `SubstituteResult` | `recommend.py § 7` | dataclass | `_find_substitute` | trace + RecommendationResult |
| `RecommendationResult` | `recommend.py § 9` | **dataclass + cached properties** | CDSS.recommend | caller |
| `trace` (the dict) | `recommend.py § 2` | **dict** (structural, JSON-serializable) | engine | RecommendationResult.trace, supervisor, JSON logs |

### CDSSInterface boundary (production wrapper)

| Shape | Location | Type | Producer | Consumer |
|---|---|---|---|---|
| `payload` dict | `interface/recommender.py` | **untyped nested dict** — `{status, run_id, patients_processed, per_patient: [...]}` | CDSSInterface.recommend_for_patients | cli, supervisor, JSON log |

---

## B · The full dataflow

```
┌─────────────────────────────────────────────────────────────────┐
│  rgs_interface.DatabaseInterface (external MySQL)               │
└──────────────────────────────┬──────────────────────────────────┘
                               │ raw rows
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  loader.DataLoader                                              │
│    .load_patient_data() ──┐                                     │
│    .load_session_data() ──┤   four pd.DataFrames                │
│    .load_ppf_data() ──────┤   (no schema enforced)              │
│    .load_protocol_similarity() ─┘                               │
└──────────────────────────────┬──────────────────────────────────┘
                               │ four raw DataFrames
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  service.RecommendationDataService.prepare()                    │
│    + whitelist filter                                           │
│    + check ppf.attrs["missing_patients"]                        │
└──────────────────────────────┬──────────────────────────────────┘
                               │ (RawInputs, similarity_df)
                               │  ← typed at last
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  pipeline.DataPipeline.process(raw_inputs, scoring_date)        │
│                                                                 │
│    Stage 1  _prepare(raw)                                       │
│             → PreparedInputs                                    │
│             (clean + clinical-window clamp on sessions)         │
│                                                                 │
│    Stage 2a _session_level_features                             │
│             → SessionLevelFeatures (DELTA_DM, RECENT_ADHERENCE) │
│                                                                 │
│    Stage 2b _protocol_level_features                            │
│             → ProtocolLevelFeatures (PPF + USAGE/WEEK + DAYS    │
│                + WEEKS_SINCE_START)                             │
│                                                                 │
│    Stage 2c _broadcast_session_onto_protocol                    │
│             → MergedFeatures                                    │
│                                                                 │
│    Stage 3  _impute_features OR _bootstrap_scoring_input        │
│             → ScoringInput                                      │
│             (collapse to one-per-PP, NaN-fill via per-patient   │
│              median or zero-default)                            │
│                                                                 │
│    Stage 4  _score                                              │
│             → ScoringOutput  (adds SCORE column)                │
└──────────────────────────────┬──────────────────────────────────┘
                               │ pd.DataFrame (ScoringOutput.df)
                               │   ── unwrap at boundary
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  engine.DataFrameBackedState(scoring, pid)                      │
│    satisfies EngineState protocol                               │
│                                                                 │
│  recommend.CDSS.recommend(state, sim)                           │
│    branch dispatch (bootstrap / repeat / update)                │
│    work in list[ProtocolRow] internally                         │
│    fill_grid_coverage post-step                                 │
│    materialize → pd.DataFrame at output                         │
│    build RecommendationResult                                   │
└──────────────────────────────┬──────────────────────────────────┘
                               │ RecommendationResult
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  interface.CDSSInterface._process_patient                       │
│    transforms RecommendationResult.recommendations               │
│      → prescription_staging rows + recsys_metrics rows           │
│    persists to DB (unless debug=True)                            │
│    builds the per-patient `result` dict                          │
└──────────────────────────────┬──────────────────────────────────┘
                               │ untyped payload dict
                               ▼
                          to cli / supervisor / log
```

---

## C · Where contracts are STRONG vs WEAK

### Strongly typed (frozen dataclass + column validation OR Protocol)

  * `RawInputs`
  * `PreparedInputs`, `SessionLevelFeatures`, `ProtocolLevelFeatures`,
    `MergedFeatures`, `ScoringInput`, `ScoringOutput`
  * `EngineState`, `SimilarityMatrix` (Protocols)
  * `ProtocolRow`
  * `RecommendationResult`
  * `SubstituteResult`

### Weakly typed (dataclass wraps a DataFrame; columns enforced only
in REQUIRED list, value-level types not checked)

  * Same set as above — the pandera `Schema` machinery EXISTS but is
    not wired into the pipeline anymore (`safe_check_types` was
    removed in F4; schema validation isn't running on the actual data).

### Untyped (raw `pd.DataFrame` passed across stage boundaries)

  * Loader returns (4 DataFrames). No schema validation, no wrapper.
  * `similarity_df` — flows through service → engine without ever
    being wrapped.
  * `trace` dict — structural type only. Keys are strings;
    `trace["swaps"][i].get("reason")` can return None or a string.

### Side-channel state

  * `ppf.attrs["missing_patients"]` — pandas `.attrs` carry meta
    (set by the loader, read by the service). Easy to lose:
    `.copy()` preserves it but most pandas operations don't.
  * `scoring.attrs["SUBSCALES"]` — same risk. Carries subscale names
    for the `CONTRIB` list-column.
  * `recommendations.attrs["trace"]` — used by legacy back-compat path
    (replaced by `RecommendationResult.trace`). Set explicitly inside
    `CDSS._finalize`-equivalent code.

---

## D · Audit problems by stage

### Loader stage

  * No schema validation at fetch time. Bad columns / missing columns
    surface only deep in the pipeline.
  * `similarity_df` has zero contract (any DataFrame with columns
    PROTOCOL_A, PROTOCOL_B, SIMILARITY works).
  * Missing patients leak via `attrs["missing_patients"]` — a string-
    keyed dict carried on a pandas object. Fragile.

### Service stage

  * `RawInputs` is typed but does no column validation
    (`validate_on_init=True` default would, but service constructs with
    defaults; let me check) — actually `RawInputs` has no
    `validate_on_init` field at all today. Just three frames, no checks.
  * The whitelist filter mutates session / ppf / similarity in
    place via reassignment. No record of what got filtered out.

### Pipeline stage

  * Boundary contracts (PreparedInputs etc) DO validate columns.
    Strong.
  * But INSIDE each stage, the DataFrame is mutated freely —
    `_attach_clinical_window`, `_clamp_to_window`, dedup, fillna.
    No provenance is captured. Once a value is imputed, you can't
    tell it apart from a real value.
  * `Imputer.init_metrics` and `Imputer.impute_metrics` mutate in
    place; no record of "this cell was zero-filled" or "this cell
    was patient-median-imputed".

### Scoring stage (Scorer.compute_score)

  * Score is computed as one wide formula. The breakdown into
    `w_ra·RA + w_dm·DM + w_ppf·PPF` is implicit. To answer "why did
    223 score 1.5?" you read source.
  * `CONTRIB` column is a list-of-float. Subscale names live in
    `df.attrs["SUBSCALES"]`. Mismatch waiting to happen.

### Engine stage

  * `ProtocolRow.from_dict` is the only safety net at the
    DataFrame ↔ dataclass boundary. Implicit type coercions
    (e.g., `int(row[USAGE]) if pd.notna(...) else 0`) happen per call.
  * Output materialization (`CDSS._materialize_dataframe`) loses
    typed info — the list[ProtocolRow] gets flattened back to a
    pd.DataFrame for the public return.

### Trace dict

  * Structural-only typing. `trace["swaps"][i]["reason"]` is a string
    that may be one of `below_mean_score`, `aisn_min_one_swap`,
    `unknown`. No enum.
  * `trace["topup"][i]["source"]` similarly: `existing` / `top_pool` /
    `exhausted`. String typo would silently work.

### CDSSInterface output

  * Big nested dict. No schema. Each call site reads `payload["per_patient"][i]["trace"]["branch"]` — string-keyed chain.
  * `interface/recommender.py` builds this dict by hand. Fields like
    `n_rows`, `n_days`, `n_protocols` are computed locally and
    embedded.

---

## E · Where the gaps matter most

Ranked by clinical / audit cost:

1. **Score lineage** (Scorer + ScoringOutput) — high cost. "Why did
   this protocol score what it scored?" is the #1 question backtest /
   supervisor users ask. Today: read source code.
2. **Provenance for imputed cells** (Imputer + everywhere) — high
   cost. Distinguishing "patient actually has this metric" from
   "metric was zero-filled" is needed for trustworthy analytics.
3. **`CONTRIB` ↔ subscale-name binding** (Scorer + models.py) — medium
   cost. Easy mistake source; one rename in YAML can mis-align everything.
4. **`trace` typed shape** (recommend.py) — medium cost. Enum for
   `reason` / `source` strings would prevent typos and add IDE
   autocomplete.
5. **Loader pandera validation** — low cost short-term. The pipeline
   stages currently catch bad shapes at their boundaries. Loader
   validation would just fail-fast 5 lines earlier.
6. **CDSSInterface payload dict** — low cost. Convertible to a typed
   dataclass anytime; no audit pressure.
7. **`similarity_df` contract** — very low cost. Three columns, well
   understood.

---

## F · Options space (no commitment yet)

Lifted from the discussion thread. Each row is a candidate design.

| Option | Scope | Effort | Best for |
|---|---|---|---|
| **A**. Just stop returning DataFrame from pipeline — return `list[ProtocolRow]` | Replace `ScoringOutput.df` with `ScoringOutput.rows` | 1 day + caller updates | Pure functional purity. Loses pandas vectorization. |
| **B**. Wrap `ScoringOutput` with typed accessors but keep DataFrame inside | Pure wrapper layer | 2 hours | Surfaces subscales / scoring_date. Doesn't fix lineage. |
| **C**. Long-form normalized contract (one row per metric per PP) | Full pipeline rewrite | 3-5 days | Maximal auditability. Bigger blast radius. |
| **D**. Hybrid — keep DataFrame + add typed `breakdowns` / `provenance` dicts to `ScoringOutput` and `RecommendationResult` | Additive — old callers unchanged | 1-2 days | Score lineage + provenance without dropping pandas idioms. **Recommended in the earlier discussion.** |
| **E**. Just type the `trace` dict (enum for `reason` / `source` etc.) | Small fix, big readability win | 2 hours | Cheapest concrete improvement. |
| **F**. Wire pandera validation on loader fetches | Adds fail-fast at fetch time | 2 hours | Catches loader-shape bugs earlier; doesn't fix lineage. |

These are not exclusive — most pair naturally (e.g., **D** + **E** + **F**).

---

## G · Open questions for the discussion

Before committing to a specific solution:

1. **Score lineage — first-class or sidecar?**
     * First-class = `ScoringOutput.df` adds 6 new columns
       (`ppf_contribution`, `delta_dm_contribution`, etc).
     * Sidecar = a `dict[(pid, proto), ScoreBreakdown]` alongside.
   Trade-off: columns are vectorized but bloat the frame; sidecar is
   typed but means two structures.

2. **Provenance — column-level or row-level?**
     * Column-level = each metric column gets a paired
       `<column>_source` column.
     * Row-level = `dict[(pid, proto), dict[metric, source]]` sidecar.
   Same trade-off as above.

3. **Should the `trace` dict become a dataclass?**
     * Current: dict of strings, structural-typed via convention.
     * Proposed: `Trace` dataclass with typed fields, `SwapEvent`
       sub-dataclass, etc.
   Cost: every consumer (supervisor, backtest, JSON log) needs an
   adapter. Benefit: typo-proof.

4. **Do we wire pandera validation back in?**
     * `safe_check_types` was removed in F4 for being unused. But the
       SCHEMAS in `models.py` still exist. We could wire them back at
       the loader or pipeline-input boundaries.
     * Open question: does pandera validation cost (~ms per call) hurt
       the F3 perf wins?

5. **Long-form vs wide-form scoring output**:
     * Pure long-form (Option C) is the cleanest audit story but
       biggest refactor.
     * Hybrid (Option D) keeps wide-form for vectorized ops + adds
       typed sidecars.
     * Open question: how often do callers actually do vectorized
       cohort-wide operations on `ScoringOutput.df`? If "never except
       inside the engine", Option C becomes more viable.

6. **CDSSInterface payload — typed or stay dict?**
     * Today it's an arbitrary nested dict. If we type it, do we put
       the dataclass in `interface/recommender.py` or promote it to
       a public type the cli / supervisor reads?

7. **Migration strategy**:
     * Big-bang: rewrite pipeline + engine + interface in one commit.
     * Additive: add new typed fields to existing dataclasses, old
       fields stay. Callers migrate one at a time. Eventually old
       fields get removed.
   Recommendation lean: additive.

---

## H · What "winning" looks like

A future caller asking "why did protocol 223 score 1.5 for patient 4378?"
should be able to write:

```python
result = cdss.recommend(state, sim)
b = result.scoring_breakdown[(4378, 223)]
print(b.formula)                            # "w_ra*RA + w_dm*DM + w_ppf*PPF"
print(b.components['ppf'])
# ComponentContribution(value=0.71, weight=1.0, contribution=0.71,
#                       imputed=False, source='real_session')
print(b.components['delta_dm'])
# ComponentContribution(value=0.0, weight=1.0, contribution=0.0,
#                       imputed=True, source='patient_median')
print(b.subscales)
# {'motor': 0.32, 'cognitive': 0.18, 'attention': 0.12, ...}
```

No source-code reading. No `.attrs` digging. No list-index confusion
with subscale names.

The plan should answer: what minimal set of changes gets us to that
calling pattern?

---

## I · Decision needed before implementation

Pick one row from the matrix below to scope the work:

| Decision | Option |
|---|---|
| Score lineage representation | column-level / sidecar dict / long-form |
| Provenance representation | column-level / sidecar dict / long-form |
| Trace dict → dataclass | yes / no |
| Pandera validation wiring | loader / pipeline / not yet |
| Migration strategy | big-bang / additive |
| Public output of CDSSInterface | typed / dict |

Once these are settled the implementation plan crystallizes.

---

## J · Findings from Q5 audit + revised lean

### Q5 answered: cohort-wide vectorized ops on `ScoringOutput.df`

Walked every caller of `pipeline.process()` (in-package + cdss-supervisor
+ cli). Findings:

| Where | What it does | Cohort-wide? |
|---|---|---|
| `interface/recommender.py:160` | `scores = pipeline.process(...)` | n/a — get |
| `interface/recommender.py:213-214` | debug-mode `dump_df` / `preview_df` | passive |
| `interface/recommender.py:327` | `scores[scores[PATIENT_ID] == pid]` | **the one filter** |
| `interface/recommender.py:375` | same filter again (debug artifacts) | same |
| `interface/recommender.py:412` | `pd.melt(patient_scores, id_vars=BY_PP, value_vars=[PPF, DELTA_DM, ...])` | on patient-filtered slice → **not cohort-wide** |
| `engine.py:239` | `scoring.loc[scoring[PATIENT_ID] == pid]` at `DataFrameBackedState.__init__` | filter once, then cached dict lookups |
| engine internals | per-protocol dict lookups, cached sorts | **never** cohort-wide |
| `cdss-supervisor/cdss-replay/replay_cdss.py:351, 432` | `cdss.scoring[cdss.scoring["PATIENT_ID"] == pid]` | **the same filter** |
| `cdss-supervisor/cdss-replay/replay_cdss.py:352-354` | `nlargest` / `sort_values` | runs on the single-patient slice |
| `ai-cdss-cli` | never touches `scoring` | n/a |

**Verdict**: the only cohort-wide operation anyone does on
`ScoringOutput.df` is `df[df[PATIENT_ID] == pid]` — the "extract one
patient's slice" filter. After that, every operation is per-patient.

Nobody groups across patients. Nobody aggregates cohort-wide. Nobody
joins frames at the cohort scoring level.

This means: **`ScoringOutput.df` as a wide cohort frame buys us
essentially nothing**. The pandas vectorization that justified the
"keep the DataFrame" decision in Option D doesn't actually get
exercised.

### Implication for the option matrix

The "we'd lose pandas vectorization" objection that pushed me toward
Option D (hybrid sidecars) is **weaker than I represented**. Reading
the matrix from Section F with this new info:

| Option | Wide-DataFrame justification | Standing |
|---|---|---|
| **A**: drop DataFrame, return `list[ProtocolRow]` | "loses cohort vectorization" — but **nobody uses it** | Stronger than I said |
| **C**: long-form (one row per metric per PP) | maximal auditability, "bigger blast radius" — but blast radius is moderate, not big, since the wide DataFrame has few callers | Stronger than I said |
| **D**: hybrid keeps DataFrame + adds sidecars | conservative, additive | Cheapest migration, but ends with two row representations |

### New synthesis: Option A++

Adopt Option A's no-DataFrame return shape, but **enrich it with
breakdown + provenance** (the wins from Option D):

```python
@dataclass(frozen=True)
class ScoringOutput:
    per_patient: dict[int, PatientScoring]   # {pid: PatientScoring}
    scoring_date: pd.Timestamp
    weights: tuple[float, float, float]
    subscales: list[str]

@dataclass(frozen=True)
class PatientScoring:
    patient_id: int
    rows:        list[ProtocolRow]                   # the scoring rows
    breakdowns:  dict[int, ScoreBreakdown]           # per-protocol lineage
    provenance:  dict[int, dict[str, str]]           # per-protocol provenance
```

  * NO DataFrame at the boundary.
  * Engine input adapter (`DataFrameBackedState`) is no longer needed —
    `PatientScoring.rows` already satisfies what the engine wants.
    Engine becomes one degree more substrate-agnostic.
  * CDSSInterface's `_transform_metrics` builds a local long-form
    DataFrame on the fly from `[row.as_dict() for row in scoring.rows]`
    + `pd.melt`. The "ScoringOutput as cohort frame" was its only
    callsite — easy to localize.
  * cdss-supervisor's `cdss.scoring[cdss.scoring["PATIENT_ID"] == pid]`
    reach-in becomes `cdss.scoring.per_patient[pid]`. ~10-line patch.

### Two viable plans, different costs

| Plan | What ships | Effort | Blast radius |
|---|---|---|---|
| **Plan D** (cautious) | `ScoringOutput` keeps `df`; adds `breakdowns` + `provenance` + `subscales` sidecars. `RecommendationResult.scoring_breakdown` populated. | 1-2 days | additive, no caller churn |
| **Plan A++** (clean end-state) | `ScoringOutput.per_patient: dict[int, PatientScoring]`. Drop the wide DataFrame entirely. Engine consumes `PatientScoring.rows` directly. | 2-3 days | ~10-line patch in CDSSInterface + supervisor; no engine algorithm change |

### Revised practical recommendation

**Ship Plan D first, migrate to A++ later.** Reasoning:

1. D is additive — no caller breaks. The new fields land alongside the
   existing DataFrame.
2. With D in place, callers naturally migrate from
   `cdss.scoring[scoring[PATIENT_ID] == pid]` to
   `cdss.scoring.breakdowns[pid]` etc.
3. After a quarter of D usage, removing the DataFrame becomes a
   mechanical cleanup — the wide form will have no callers left.
4. A++ in one step is achievable but bundles "add lineage" + "drop
   wide form" — two distinct migrations conflated.

Final commit when ready: `f5: ScoreBreakdown + provenance on
ScoringOutput / RecommendationResult` (Plan D).

Follow-up commit some quarter from now: `f6: drop ScoringOutput.df,
adopt per_patient dict` (Plan A++).
