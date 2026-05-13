# Synthetic Data Generators — Future Plan

Branch: not yet started. Target: future phase on `functionality-refactor`
(or its successor). Estimated effort: 1-3 days depending on scope.

## Motivation

After the `functionality-refactor` branch (F0-F4b) the engine already
accepts pandas-free input via `DictBackedState` + `DictSimilarity`. The
"10-line synthetic backtest" test demonstrates this works.

What's MISSING is structured, reusable, deterministic patient/data
generators. Today writing a synthetic case means:

  * Hand-constructing 20+ `ProtocolRow` objects per test.
  * No way to roll a patient's response forward over weeks (chained mode
    in cdss-supervisor uses the `DictBackedState.with_prescribed_set`
    hack — only chains the prior set, not the trajectory).
  * No reusable cohort generators for end-to-end CI.

The substrate is in place; this plan adds the structured layer on top.

## Three injection points (granularity ladder)

```
                                       INJECT HERE
DataLoader.load_*()                          ▲
       ↓                                     │
RecommendationDataService.prepare()  ────── 2. Service-level
       → RawInputs                           │
       ↓                                     │
DataPipeline.process(raw_inputs)             │
       → scoring DataFrame                   │
       ↓                                     │
DataFrameBackedState(scoring, pid)   ────── 1. Engine-level (DictBackedState)
       ↓                                     │
CDSS.recommend(state, similarity)
       → RecommendationResult
                                            ──── 3. Trajectory-level (simulator)
```

| Level | What it replaces | Use case | Status |
|---|---|---|---|
| **1. Engine-level** — `DictBackedState` | the scoring DataFrame | unit tests, single-patient probes | **DONE** (F2) |
| **2. Service-level** — `SyntheticDataService` | `RecommendationDataService` | full-pipeline tests, end-to-end CI | TODO |
| **3. Trajectory-level** — `PatientSimulator` | the patient's *response* over time | chained-mode backtest, drift analysis | TODO |

## Target shape — single `synthetic.py` module

Per the flat-layout principle, one file at the package root with section
banners. Estimated ~500 lines.

```
src/ai_cdss/synthetic.py
├── SECTION 1  PatientGenerator protocol
│              (PEP 544 structural; produces patient/session/PPF
│               DataFrames + similarity)
├── SECTION 2  PatientResponseModel protocol
│              (patient × prescription × time → DM/adherence/sessions)
├── SECTION 3  Concrete generators
│              - RandomCohort   (fast tests; uniform random)
│              - AISNTrialCohort (matches AISN dimensions)
├── SECTION 4  Concrete response models
│              - StaticResponse     (no trajectory; one-shot scores)
│              - LinearDMResponse   (DM rises linearly with usage)
│              - PlateauDMResponse  (DM caps at protocol-specific ceiling)
│              - DecayingAdherence  (motivation drops over weeks)
├── SECTION 5  SyntheticDataService
│              (drop-in for RecommendationDataService — same .prepare()
│               signature, returns RawInputs + similarity)
├── SECTION 6  PatientSimulator
│              (stateful: holds patient + history + response model;
│               .step(prescription) advances one week)
└── SECTION 7  Quickstart helpers
              - synthetic_cohort(n=10, seed=42) -> RawInputs
              - synthetic_state(patient_id, ...) -> DictBackedState
```

## Design decisions

### 1. Synthetic injection at the SERVICE level, NOT loader level

Loaders are DB-specific (talk to `rgs_interface.DatabaseInterface`). A
`SyntheticLoader` returning fake DataFrames to satisfy the loader
contract is awkward indirection. Better: `SyntheticDataService` directly
returns `(RawInputs, similarity_df)` — same contract
`RecommendationDataService.prepare()` produces today.

`CDSSInterface` already accepts a `data_service` parameter. Plug-in is
one line:

```python
cdss = CDSSInterface(
    loader=None,                          # not needed for synthetic
    data_service=SyntheticDataService(generator),
)
```

### 2. Generators and response models are SEPARATE protocols

A `PatientGenerator` produces **data shapes** (DataFrames). A
`PatientResponseModel` defines **trajectory dynamics** (how DM evolves
under a prescription). Compose them:

```python
generator = AISNTrialCohort(
    n_patients=30,
    response_model=LinearDMResponse(slope=0.05, noise=0.1),
    seed=42,
)
```

Why split: the same response model gets reused by multiple generators
(cohort generation, single-patient testing, simulator step). And
different generators want to swap response models (research: linear vs
plateau vs saturating DM).

### 3. PatientSimulator = generator + engine, in a loop

```python
sim = PatientSimulator(
    patient=generator.generate_patient(seed=42),
    response_model=LinearDMResponse(slope=0.05),
    engine=CDSSInterface(...),
)

for week in range(12):
    sim.step()
    print(sim.history[week].result.swap_decisions)
```

`PatientSimulator` is the abstraction needed for chained-mode
backtesting. cdss-supervisor today hacks this via the
`DictBackedState.with_prescribed_set` override; properly-designed
`PatientSimulator` replaces that hack.

### 4. Determinism by construction

Every protocol method takes a `seed: int` (or accepts a seeded RNG).
Synthetic backtests must be reproducible across CI runs.

### 5. Protocols, not abstract base classes

Per the `DataLoaderBase` discussion (the answer was: don't need it after
deleting the other implementations), use PEP 544 `Protocol` for
generators + response models. Structural subtyping = duck-typed test
fakes; no inheritance ceremony.

## Concrete API

```python
# Layer 1 — engine only, fastest (DONE today via DictBackedState)
from ai_cdss.synthetic import synthetic_state
state, sim = synthetic_state(patient_id=1, n_protocols=12, seed=42)
result = CDSS(state, n=12).recommend(1, sim)

# Layer 2 — full pipeline w/ synthetic source
from ai_cdss.synthetic import AISNTrialCohort, SyntheticDataService
gen = AISNTrialCohort(n_patients=10, seed=42)
service = SyntheticDataService(gen)
cdss = CDSSInterface(loader=None, data_service=service)
result = cdss.recommend_for_patients([1, 2, 3])

# Layer 3 — full trajectory simulation
from ai_cdss.synthetic import PatientSimulator, LinearDMResponse
sim = PatientSimulator(
    cohort=AISNTrialCohort(n_patients=1, seed=42),
    response_model=LinearDMResponse(slope=0.05),
    engine=CDSSInterface(...),
)
for week in range(12):
    sim.step()
trajectory = sim.history     # list of (week, prescription, sessions, dm_at_end)
```

## Implementation phases

Three sizing options — pick when reviving this plan:

| Option | Effort | Outcome |
|---|---|---|
| **A**: Section 1 (PatientGenerator protocol) + Section 7 (quickstart helpers) | 1-2 hours | Test-grade synthetic injection. Layers 2 + 3 deferred. |
| **B**: Sections 1-5 (generators + response models + SyntheticDataService) | 1 day | Full pipeline runs from synthetic. Layer 3 deferred. |
| **C**: All 7 sections including PatientSimulator | 2-3 days | Trajectory simulation is a first-class feature. cdss-supervisor's chained mode rewrites against it. |

**Recommended**: **B** as the next concrete step. Ship in one commit;
build the simulator (C) as a follow-up against a stable synthetic
foundation.

## Tests to add when this lands

- `test_random_cohort_generates_deterministic_data` — seed = 42 produces
  identical frames across runs.
- `test_synthetic_service_drop_in_compat` — `CDSSInterface` with
  `SyntheticDataService` produces a valid `RecommendationResult` with
  every branch (bootstrap, update, repeat) reachable.
- `test_aisn_trial_cohort_matches_trial_dimensions` — `n_patients=30,
  n_protocols=27, n_weeks=12`.
- `test_linear_dm_response_trajectory` — over 12 weeks under a fixed
  prescription, DM rises by `slope × weeks` ± noise tolerance.
- `test_patient_simulator_step_advances_history` — `sim.step()` adds
  one entry; `sim.history[week].prescription` matches what the engine
  recommended.
- `test_chained_mode_backtest_via_simulator` — replicates current
  cdss-supervisor chained-mode behavior with the new simulator;
  proves the rewrite is feasible.

## Open questions to resolve before starting

1. **Where do response models source their parameters from?** Per-
   protocol DM ceiling, per-patient learning rate, per-week skip
   probability — these need defaults that match observed AISN trial
   distributions. One option: ship `synthetic/aisn_priors.yaml` with
   empirically-derived defaults. Another: keep parameters constructor-
   only, force the caller to choose.
2. **Does `PatientGenerator` produce one patient or many?** Today
   `RecommendationDataService.prepare(patient_list)` takes a list.
   Generator could expose `generate_patients(n)` returning a cohort
   frame, plus `generate_for_patient(pid)` for single-patient. The
   simpler design is "cohort always" — single-patient is just `n=1`.
3. **How does `PatientSimulator` interact with persistence?**
   `CDSSInterface` writes to `prescription_staging` + `recsys_metrics`
   in production. For simulated patients those writes are unwanted.
   Option: simulator always uses `debug=True` (no DB writes).
4. **Snapshot/restore for replay debugging**. If a simulation crashes
   at week 7, can we replay from week 6? Yes if `PatientSimulator`
   exposes `snapshot() -> dict` and `restore(dict)`. Worth doing in
   phase C.

## Why this matters

Three downstream wins, in order of leverage:

1. **CI/test runtime**: end-to-end synthetic recommendations are
   ~1ms each (per F3 benchmark). A 30-patient × 12-week sweep runs in
   under 1 second. Same scope against the real DB takes ~30s.
2. **Engine policy experiments**: env-wide MVT vs prescribed-mean MVT
   vs median-MVT; with synthetic trajectories you can A/B test policy
   changes against the same patient simulator, no real-patient risk.
3. **cdss-supervisor chained-mode cleanup**: today's chained backtest
   uses an ad-hoc `DictBackedState.with_prescribed_set` hack inside
   `cdss-replay/replay_cdss.py`. Replacing that with `PatientSimulator`
   removes ~30 lines and produces a more principled trajectory (one
   that respects response-model dynamics, not just "carry the prior
   set forward").

## Status

- [ ] Open question pass — decide on the 4 questions above.
- [ ] Implement Option B (Sections 1-5).
- [ ] Add the 6 test cases listed above.
- [ ] Decide whether C (PatientSimulator) ships as a follow-up commit
      on the same branch or a new branch.
- [ ] cdss-supervisor adapter: replace `with_prescribed_set` hack with
      `PatientSimulator` once C is in.
