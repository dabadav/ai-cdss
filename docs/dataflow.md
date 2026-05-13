# Current dataflow — visual reference

Renders as Mermaid in GitHub / VS Code (with `bierner.markdown-mermaid`)
/ Obsidian. Colors track contract strength:

- 🟢 **green** — strongly typed (frozen dataclass with column validation,
  or PEP 544 Protocol)
- 🟡 **yellow** — weakly typed (dataclass wrapping a DataFrame; columns
  enforced only via a REQUIRED list)
- 🔴 **red** — untyped (raw `pd.DataFrame` or dict)

## A · Top-level flow

```mermaid
flowchart TD
    DB[("rgs_interface MySQL")]
    FS[("~/.ai_cdss/output/<br/>PPF parquet + similarity csv")]
    REPO["data.RGSCohortRepository.find<br/><i>fetch + whitelist filter</i>"]
    COHORT["Cohort<br/>patient / session / ppf /<br/>similarity / whitelist / missing_ppf"]
    PIPE["scoring.DataPipeline.process"]
    SO["ScoringOutput<br/><i>wraps pd.DataFrame</i>"]
    STATE["engine.PatientState<br/><i>adapter implements EngineState</i>"]
    CDSS["recommender.Recommender.recommend"]
    RESULT["RecommendationResult<br/>recommendations + trace +<br/>swap_decisions + topup_events"]
    IFACE["interface.CDSS<br/>persist + build payload"]
    PAYLOAD["payload dict<br/>nested per-patient list"]
    CONSUMER["cli / supervisor / JSON log"]

    DB --> REPO
    FS --> REPO
    REPO --> COHORT
    COHORT --> PIPE
    PIPE --> SO
    SO --> STATE
    COHORT -. similarity .-> CDSS
    STATE --> CDSS
    CDSS --> RESULT
    RESULT --> IFACE
    IFACE --> PAYLOAD
    PAYLOAD --> CONSUMER

    style PAYLOAD fill:#fde4d4,stroke:#a8000d
    style COHORT fill:#dff5e6,stroke:#136f3e
    style RESULT fill:#dff5e6,stroke:#136f3e
    style STATE fill:#dff5e6,stroke:#136f3e
    style SO fill:#fdedd4,stroke:#b46b00
```

**Reading the diagram:**

- `RGSCohortRepository.find` is the single boundary fetching from MySQL + local files. It returns one typed `Cohort` (green dataclass) carrying every frame the pipeline + engine consume — patient, session, PPF, similarity, plus the whitelist already applied and any `missing_ppf` patient IDs for guard logic.
- Pipeline takes the `Cohort` and produces `ScoringOutput` (yellow — typed wrapper around a DataFrame; columns checked but values are pure pandas).
- Engine adapts the scoring DataFrame to `PatientState` (green — Protocol-conforming) and consumes `cohort.similarity` alongside. Engine internals operate on `list[ProtocolRow]`.
- CDSS returns `RecommendationResult` (green dataclass with cached_property breakdowns).
- CDSS unwraps the result into an untyped nested dict for the cli / supervisor / JSON log.

## B · Pipeline internals (zoom into the green PIPE box)

```mermaid
flowchart TD
    COHORT["Cohort<br/>patient / session / ppf"]
    P1["_prepare<br/>clinical-window clamp"]
    PI["PreparedInputs"]
    SL["_session_level_features"]
    PL["_protocol_level_features"]
    SLF["SessionLevelFeatures"]
    PLF["ProtocolLevelFeatures"]
    BC["_broadcast_session_onto_protocol"]
    MF["MergedFeatures"]
    IMP["_impute_features<br/><i>per-patient median fill</i>"]
    BS["_bootstrap_scoring_input<br/><i>NaN defaults</i>"]
    SI["ScoringInput"]
    SC["_score<br/><i>w·RA + w·DDM + w·PPF</i>"]
    SO["ScoringOutput"]

    COHORT --> P1 --> PI
    PI -- has_sessions=True --> SL
    PI -- has_sessions=True --> PL
    SL --> SLF
    PL --> PLF
    SLF --> BC
    PLF --> BC
    BC --> MF --> IMP --> SI
    PI -- has_sessions=False --> BS --> SI
    SI --> SC --> SO

    style PI fill:#fdedd4,stroke:#b46b00
    style SLF fill:#fdedd4,stroke:#b46b00
    style PLF fill:#fdedd4,stroke:#b46b00
    style MF fill:#fdedd4,stroke:#b46b00
    style SI fill:#fdedd4,stroke:#b46b00
    style SO fill:#fdedd4,stroke:#b46b00
    style COHORT fill:#dff5e6,stroke:#136f3e
```

**Five named boundaries, each a frozen dataclass with declared
REQUIRED columns.** The yellow color means "DataFrame inside a typed
wrapper" — column presence enforced, value-level pandera validation
not running.

## C · Engine internals (zoom into the green STATE → CDSS box)

```mermaid
flowchart TD
    SO["ScoringOutput.df<br/>pd.DataFrame"]
    SIM["similarity_df<br/>pd.DataFrame"]
    DFBS["PatientState<br/><i>cached_property cache</i>"]
    DFSIM["DataFrameSimilarity<br/><i>_by_a precomputed</i>"]
    ES{{"EngineState protocol"}}
    SM{{"SimilarityMatrix protocol"}}
    CDSS["CDSS.recommend<br/>dispatch + topup"]
    BRANCH{branch?}
    BOOT["bootstrap<br/>top-N + round-robin"]
    REP["repeat_skipped_week<br/>copy prior"]
    UPD["update<br/>MVT swap loop"]
    ROWS["list[ProtocolRow]"]
    TOPUP["_top_up_schedule<br/>existing → top_pool"]
    FINAL["list[ProtocolRow]<br/>+ trace dict"]
    DF["pd.DataFrame<br/>(materialized at boundary)"]
    RR["RecommendationResult"]

    SO --> DFBS
    SIM --> DFSIM
    DFBS -.satisfies.-> ES
    DFSIM -.satisfies.-> SM
    ES --> CDSS
    SM --> CDSS
    CDSS --> BRANCH
    BRANCH -- empty prior --> BOOT
    BRANCH -- usage_week=0 all --> REP
    BRANCH -- has prior --> UPD
    BOOT --> ROWS
    REP --> ROWS
    UPD --> ROWS
    ROWS --> TOPUP --> FINAL --> DF --> RR

    style ES fill:#dff5e6,stroke:#136f3e
    style SM fill:#dff5e6,stroke:#136f3e
    style DFBS fill:#dff5e6,stroke:#136f3e
    style DFSIM fill:#dff5e6,stroke:#136f3e
    style RR fill:#dff5e6,stroke:#136f3e
    style ROWS fill:#dff5e6,stroke:#136f3e
    style FINAL fill:#dff5e6,stroke:#136f3e
    style SO fill:#fdedd4,stroke:#b46b00
    style SIM fill:#fde4d4,stroke:#a8000d
    style DF fill:#fde4d4,stroke:#a8000d
```

**Engine is the most strongly-typed region in the package.** Both
inputs are coerced to Protocol-satisfying types at the boundary;
internals work in `list[ProtocolRow]`; output materializes to
`pd.DataFrame` only at the very last step.

## D · Where each audit problem lives (overlay)

```mermaid
flowchart TD
    REPO["RGSCohortRepository"]
    COHORT_MISS["Cohort.missing_ppf<br/><i>explicit list, no longer side-channel</i>"]
    PIPE["DataPipeline"]
    IMP["Imputer<br/><b>mutates in place</b><br/><i>no provenance kept</i>"]
    SCORE["Scorer<br/><b>w·RA + w·DDM + w·PPF</b><br/><i>no breakdown saved</i>"]
    ATTRS_SUB["scoring.attrs[SUBSCALES]<br/><i>list-index → name binding</i>"]
    SO["ScoringOutput<br/><b>wide DataFrame, no lineage</b>"]
    TRACE["trace dict<br/><i>structural typing only</i>"]
    PAYLOAD["payload dict<br/><i>untyped nested</i>"]

    REPO --> COHORT_MISS
    REPO --> PIPE
    PIPE --> IMP
    IMP --> SCORE
    SCORE --> ATTRS_SUB
    SCORE --> SO
    SO -.engine.-> TRACE
    TRACE --> PAYLOAD

    style COHORT_MISS fill:#dff5e6,stroke:#136f3e
    style IMP fill:#fde4d4,stroke:#a8000d
    style SCORE fill:#fde4d4,stroke:#a8000d
    style ATTRS_SUB fill:#fde4d4,stroke:#a8000d
    style SO fill:#fde4d4,stroke:#a8000d
    style TRACE fill:#fdedd4,stroke:#b46b00
    style PAYLOAD fill:#fde4d4,stroke:#a8000d
```

**Five audit holes**, top-to-bottom by stage (the
`ppf.attrs["missing_patients"]` side-channel was retired in f5 — it's
now an explicit `Cohort.missing_ppf` field):

1. **`Imputer` mutations** — silent. Imputed cells indistinguishable
   from real values after this stage.
2. **`Scorer` formula** — computed wide, no per-component breakdown
   captured.
3. **`scoring.attrs["SUBSCALES"]`** — list-index to subscale-name
   binding lives in `.attrs`. Mismatch waiting to happen.
4. **`ScoringOutput.df`** — wide DataFrame, no lineage column, no
   provenance.
5. **`trace` dict** — structural typing only. `reason` / `source`
   are stringly-typed enums.

## E · Reference: contract strength by stage

```mermaid
quadrantChart
    title "Contract strength × Audit cost"
    x-axis "Untyped" --> "Strong"
    y-axis "Low audit cost" --> "High audit cost"
    quadrant-1 "Worth typing"
    quadrant-2 "Critical — type now"
    quadrant-3 "Leave alone"
    quadrant-4 "Already good"
    "ScoringOutput.df": [0.45, 0.95]
    "trace dict": [0.45, 0.70]
    "Imputer mutation": [0.20, 0.80]
    "scoring.attrs SUBSCALES": [0.20, 0.60]
    "payload dict": [0.30, 0.30]
    "Cohort.similarity": [0.85, 0.20]
    "Cohort": [0.90, 0.15]
    "PreparedInputs": [0.80, 0.20]
    "MergedFeatures": [0.80, 0.15]
    "ScoringInput": [0.80, 0.20]
    "EngineState": [0.95, 0.15]
    "RecommendationResult": [0.90, 0.20]
    "ProtocolRow": [0.95, 0.10]
```

Top-left quadrant (high audit cost, weakly typed) is the priority
work. `ScoringOutput.df`, `trace dict`, `Imputer mutation`, and
`scoring.attrs[SUBSCALES]` all sit there.
