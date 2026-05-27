# cdss-supervisor migration plan

Goal: move cdss-supervisor (`cdss-replay/replay_cdss.py`, 796 L) onto
the refactored ai-cdss public API, consuming only the new typed return
types (`Cohort`, `RecommendationResult`) instead of poking at engine
internals.

Status: **plan**. No code changes yet.

## Audit of current coupling

Today's `replay_cdss.py` reaches into 13 ai-cdss modules. Roughly half
of those usages are gone after F5/F6:

### Broken imports (modules deleted or renamed)

| Old | Status | New |
|---|---|---|
| `from ai_cdss.cdss import CDSS` | renamed | `from ai_cdss.recommender import Recommender` |
| `from ai_cdss.loaders import DataLoader` | deleted | `from ai_cdss.data import RGSCohortRepository` |
| `from ai_cdss.processing import DataProcessor` | deleted | `from ai_cdss.scoring import DataPipeline` |
| `from ai_cdss.services.data_preparation import RecommendationDataService` | deleted | use `repository.find(pids) → Cohort` |
| `from ai_cdss.services.ppf_service import PPFService` | deleted | `from ai_cdss.precompute import compute_ppf_for_patients, persist_ppf` |
| `from ai_cdss.services.protocol_similarity import ProtocolSimilarityService` | deleted | `from ai_cdss.precompute import compute_protocol_similarity_matrix, persist_similarity` |
| `from ai_cdss.processing.clinical import ProtocolToClinicalMapper` | moved | `from ai_cdss.data import ProtocolToClinicalMapper` |
| `from ai_cdss.services.whitelist_service import ProtocolWhitelistService` | deleted | `from ai_cdss.data import load_whitelist` |

### Private-method spelunking (the bigger problem)

`analyze_replay_swap()` reconstructs the swap rule chain by calling
**8 private methods** on the engine:

```
cdss._get_prescriptions(pid)
cdss._is_week_skipped(pre_df)
cdss._get_patient_protocol_usage(pid)
cdss._get_protocol_similarities(proto, sim, excluded)
cdss._get_unused_candidates(usage)
cdss._select_most_similar(unused, sims)
cdss._get_top_similar_protocols(sims, top_n=5)
cdss._get_least_used_candidates(usage, top5)
```

None of these survived the F2 → F4 refactor. They were re-organized into
strategies + helpers behind `Recommender.recommend()`. Re-implementing
them in supervisor would mean re-implementing the algorithm.

**Now unnecessary.** `RecommendationResult` exposes everything
`analyze_replay_swap` was trying to reconstruct:

| Old reconstruction | New `RecommendationResult` property |
|---|---|
| `cdss._get_prescriptions(pid).empty` | `result.branch == "bootstrap"` |
| `cdss._is_week_skipped(pre_df)` | `result.branch == "repeat_skipped_week"` |
| Swap rule chain | `result.swap_decisions: list[SubstituteResult]` |
| Per-swap reason | `SubstituteResult.reason` ("unused_candidate" / "least_used_among_top_similar" / "exhausted") |
| Per-swap similarity | `SubstituteResult.similarity` |
| Per-swap candidates considered | `SubstituteResult.candidates` |
| MVT mean used | `result.mvt_mean` |
| Targets the engine flagged for swap | `result.swap_targets`, `result.swap_reasons` |
| Final per-protocol schedule | `result.final_protocols`, `result.recommendations` |
| Top-up filler events | `result.topup_events` |
| Candidate pool for a swap | `result.candidate_pool_for(removed_pid)` |
| Full audit | `result.trace` (structured dict) |

That's the win the F2 + F4 refactor was designed to enable: the
supervisor no longer needs to know the algorithm.

## Target shape (post-migration)

### Imports — 1 line of ai-cdss code per concept

```python
from ai_cdss.data import (
    RGSCohortRepository, ProtocolToClinicalMapper, load_whitelist,
)
from ai_cdss.scoring import DataPipeline
from ai_cdss.recommender import Recommender   # only if engine called directly
from ai_cdss.constants import (
    DAYS, PROTOCOL_ID, PATIENT_ID, N, N_DAYS, PROTOCOLS_PER_DAY,
)
```

No `_private_methods`, no `services.*`, no `loaders.*`,
no `processing.*`, no `cdss.cdss`.

### `build_replay_context` — 4 lines

```python
def build_replay_context(*, warmup: bool = False) -> dict:
    db = DatabaseInterface()
    repository = RGSCohortRepository(db=db, rgs_mode="plus")
    pipeline   = DataPipeline()
    proto_name, proto_features, feature_keys = fetch_protocol_meta(repository)
    return {
        "db": db, "repository": repository, "pipeline": pipeline,
        "proto_name": proto_name,
        "proto_features": proto_features,
        "feature_keys": feature_keys,
    }
```

### `prepare_patient_state` — repository, not service

```python
def prepare_patient_state(ctx, pid, *, warmup=False, warmup_force=False):
    ...
    cohort = ctx["repository"].find([pid])   # one call, all four frames
    return {
        "pid": pid,
        "meta": meta,
        "start_date": start_date,
        "cohort": cohort,                    # carries similarity inside
    }
```

`cohort.similarity` replaces the old `protocol_similarity` tuple
element. `cohort.session/.patient/.ppf` replace `rgs_data`.

### `replay_engine_only` — RecommendationResult-driven

```python
def replay_engine_only(ctx, patient_state, week, *, chained_prev_final=None):
    pid = patient_state["pid"]
    cohort = patient_state["cohort"]
    scoring_date = pd.Timestamp(patient_state["start_date"] + timedelta(weeks=week))

    scores = ctx["pipeline"].process(cohort, scoring_date)
    if chained_prev_final is not None:
        scores[DAYS] = scores[PROTOCOL_ID].map(
            lambda p: list(chained_prev_final.get(int(p), []))
        )

    recommender = Recommender(scoring=scores, n=N,
                              days=N_DAYS, protocols_per_day=PROTOCOLS_PER_DAY)
    result = recommender.recommend(pid, cohort.similarity)

    swap_analysis = summarize_swap(result, ctx["proto_name"])
    swap_analysis["protocol_features"] = ctx["proto_features"]
    swap_analysis["feature_keys"] = ctx["feature_keys"]

    return {
        "meta": patient_state["meta"],
        "week": week,
        "scoring_date": str(scoring_date.date()),
        "scores": scores,
        "recommendations": result.recommendations,
        "result": result,                # full RecommendationResult
        "swap_analysis": swap_analysis,
    }
```

### `analyze_replay_swap` → `summarize_swap` (50 → ~15 lines)

Replaces the ~100-line private-method reconstruction with a switch on
`result.branch`. Each branch reads exclusively off the result object.

```python
def summarize_swap(result, proto_name=None):
    pn = proto_name or {}
    def name(p): return pn.get(int(p), "")

    if result.branch == "bootstrap":
        return {
            "path": "bootstrap",
            "explanation": "No prior prescriptions. Top-N by SCORE.",
            "selected": [
                {"PROTOCOL_ID": p, "PROTOCOL_NAME": name(p)}
                for p in result.final_protocols
            ],
        }
    if result.branch == "repeat_skipped_week":
        return {
            "path": "skipped",
            "explanation": "USAGE_WEEK == 0 for all prior. Repeated as-is.",
            "pre": result.trace.get("prior", []),
        }
    # update branch
    return {
        "path": "update",
        "mvt_mean": result.mvt_mean,
        "swap_targets": result.swap_targets,
        "swap_reasons": result.swap_reasons,
        "substitutes": [
            {
                "out": s.removed_id, "out_name": name(s.removed_id),
                "in":  s.protocol_id, "in_name": name(s.protocol_id),
                "similarity": s.similarity,
                "rule": s.reason,
                "candidates_considered": s.candidates,
            }
            for s in result.swap_decisions
        ],
        "topup": result.topup_events,
    }
```

### `fetch_protocol_meta` — read from repository, not loader

```python
def fetch_protocol_meta(repository):
    raw = repository.protocol_attributes()
    if raw is None or raw.empty:
        return {}, {}, []
    name_map = {int(v): str(k) for k, v in zip(raw.index.astype(str), raw["PROTOCOL_ID"])}
    feats = ProtocolToClinicalMapper().map_protocol_features(raw).fillna(0.0)
    return (
        name_map,
        {int(p): {str(k): float(v) for k, v in row.items()}
         for p, row in feats.iterrows()},
        list(feats.columns),
    )
```

## What this saves

| Metric | Before | After |
|---|---|---|
| ai-cdss modules imported | 7 (`cdss`, `loaders`, `processing`, `processing.clinical`, `services.{data_preparation, ppf_service, protocol_similarity, whitelist_service}`, `constants`) | 4 (`data`, `scoring`, `recommender`, `constants`) |
| Private methods called | 8 | 0 |
| Lines in `analyze_replay_swap` | ~100 | ~25 |
| Algorithm knowledge in supervisor | high (re-walks unused/least-used/similarity logic) | none (consumes typed result) |
| Coupling to ai-cdss algorithm changes | tight (any internal rename = supervisor break) | loose (only public types + `result.*` properties) |

## Phasing

| Phase | Scope | Effort |
|---|---|---|
| **S1** | Bump dependency version pin; update imports to new module paths. Replace `DataLoader/DataProcessor` instantiation with `RGSCohortRepository/DataPipeline`. | 30 min |
| **S2** | Replace `data_service.prepare()` with `repository.find()`. Replace `processor.process_data()` with `pipeline.process()`. Stop unpacking `(rgs_data, sim)` tuple — read off `cohort.similarity`. | 30 min |
| **S3** | Replace `analyze_replay_swap` with `summarize_swap` consuming `RecommendationResult`. Delete the 8 private-method calls. | 1 h |
| **S4** | `fetch_protocol_meta` from `repository.protocol_attributes()`. Drop `services.whitelist_service` use; if whitelist is needed elsewhere call `load_whitelist()`. | 20 min |
| **S5** | Verify backtest sweep still runs; check the cdss-dashboard front-end consumes the new payload shape (no shape change expected since the returned dict keys are the same). Smoke-test one full backtest run. | 1 h |

Total: **3-4 h**, single branch.

## Open questions

1. **Chained-mode DAYS override** — current code mutates the scoring
   DataFrame's DAYS column directly. The new pipeline still emits a
   plain `pd.DataFrame` from `DataPipeline.process`, so this still
   works. But it's brittle — relying on engine input being a mutable
   DataFrame. Long-term: build an `EngineState` adapter that injects
   chained DAYS at the boundary (similar to how `DictPatientState`
   works for synthetic data).
2. **Frontend payload changes** — `replay()` returns `recommendations`
   (DataFrame). It also now has `result` (RecommendationResult). Does
   the dashboard consume `result.*` or just `recommendations` + the
   `swap_analysis` dict? If only the dict, no front-end change.
3. **Backtest cache** — current pattern hoists `prepare_patient_state`
   outside the week loop (per project memory). Same hoist works for
   `repository.find([pid])` — the Cohort is reusable across all weeks
   of the same patient. No regression.
4. **Branch label compatibility** — `result.branch` values are
   `"bootstrap"`, `"repeat_skipped_week"`, `"update"`. The old code
   wrote `"path": "bootstrap" / "skipped" / "update"` to the
   `swap_analysis` dict. Keep the old `"path"` value mapping in
   `summarize_swap` so the dashboard sees no change.

## What this does NOT do

* Doesn't change the backtest semantics — same sweep, same outputs.
* Doesn't touch the cdss-dashboard frontend (assumes the `swap_analysis`
  dict shape is unchanged, which `summarize_swap` is written to honor).
* Doesn't change the recommendation algorithm.
* Doesn't introduce new tests (existing cdss-supervisor tests, if any,
  should pass).
