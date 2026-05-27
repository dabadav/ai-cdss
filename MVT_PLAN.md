# MVT criterion refinement — cohort-wide mean + cohort-wide candidate pool

Status: **IMPLEMENTED** (threshold only). Decisions taken:
- mvt_mean → **replace** semantic (now cohort-wide, no second field).
- pick rule → **Option A** (keep two-tier similarity; only the
  threshold moved). Candidate pool was already cohort-wide.
- aisn_min_one_swap fallback **survives** unchanged.

Verified: `result.mvt_mean` = cohort mean (0.81 on the distributed
test fixture) not prescribed-set mean (0.89). 83/83 unit tests green.
Changed in `recommender.py`: `_below_mean_protocols` (mean over
`state.all_protocols`, targets still from `prior`), `_select_swap_targets`
(threads mean out), `_update_strategy` (stores `trace["mvt_mean"]`),
`_compute_mvt_mean` (reads cohort value from trace).

## What's there now

Two coupled things both look only at the **prescribed set**:

1. **Threshold** — `_below_mean_protocols(prior)` averages SCORE over
   `state.prescribed_rows` only. Protocols below that local mean are
   swap targets.
2. **Candidate pool** — `_find_substitute` ranks by similarity to the
   removed protocol and picks the most-similar UNUSED protocol (tier 1)
   or least-used among top-5 most-similar (tier 2). "Unused" is
   relative to the patient's session history; the universe queried is
   `state.all_protocols`.

So the *threshold* is local to the prescribed set, but the *candidate
pool* is already cohort-wide. The asymmetry is the gap to close.

## Proposed change

### Threshold

Mean over **all of the patient's protocols** instead of only the
prescribed set. Keep/swap criterion still applies only to prescribed
protocols (the only ones we can swap out).

```
threshold       = mean( state.score_row(p).score
                        for p in state.all_protocols )
swap_targets    = [ r.protocol_id
                    for r in state.prescribed_rows
                    if r.score < threshold ]
keep_set        = prescribed - swap_targets
```

### Candidate pool

Candidate pool = **every non-prescribed protocol** the patient has, NOT
filtered by similarity up front:

```
candidate_pool  = state.all_protocols - {r.protocol_id for r in prescribed}
```

The pool is mutable across the swap loop: each protocol picked as a
substitute is **removed from the pool** before the next swap target is
considered. Sequential picks deplete the pool.

### Pick rule within the pool

Open question — same two-tier (similarity + usage) as today applied to
the depleting pool, or simpler "highest-score among remaining"?

* **Option A** — keep two-tier similarity rule, just run it over the
  shrinking pool instead of the full `all_protocols`.
* **Option B** — drop similarity, pick by SCORE rank: top candidate
  remaining → next swap. Closer to the "MVT" name (marginal value
  theorem: take the best remaining alternative).
* **Option C** — hybrid: SCORE-first ranking with similarity used as a
  tie-breaker.

Lean Option B if we're already trusting SCORE for the threshold —
consistent signal end-to-end. Option A keeps current similarity
behavior so historical swap audits compare like-for-like.

## Why bother

The current local mean is **degenerate when the prescribed set is
homogeneous**: 5 protocols all sitting at SCORE ≈ 0.4 average to 0.4,
so swap targets are only the ones a fraction below — even if the
patient has 15 unprescribed protocols all scoring 0.7+. The cohort-wide
mean exposes this — every prescribed protocol scoring below the
patient's average is a candidate for swap pressure.

Symmetrical to the candidate pool that's already cohort-wide.

## Trace + audit impact

* `trace["mvt_mean"]` is currently the prescribed-set mean.
  Replace with cohort-wide mean. (Or add `mvt_mean_all` and keep both.)
* `RecommendationResult.mvt_mean` currently averages `trace["prior"]`
  scores. Same call: replace semantic, OR add a second field
  (`mvt_mean_all`) for the new value while keeping the legacy field for
  back-compat with cdss-supervisor's existing dashboards.
* `trace["swaps"][i]["candidate_pool"]` already records the pool at
  the time of each pick. Under the new rule, this becomes the
  **post-removal** pool — useful for replaying why a given pick
  happened.

## Tests to revisit

* `test_recommendation_result.py:test_result_update_branch_has_mvt_mean`
  — recompute the expected mean across all patient protocols.
* `test_cdss_recommend.py:test_update_branch_*` — swap counts may
  shift (more or fewer prescribed protocols below the cohort mean
  vs. below the prescribed mean).
* `test_substrate_agnostic.py:test_recommend_with_dict_state_update_branch`
  — confirm the DictPatientState path computes the same threshold
  (it should — `all_protocols` + `score_row` are protocol-defined).

## Effort

* Code change: ~15 lines + 1 call site + result-field decision.
* Test update: 3–5 assertions to recompute.
* Trace audit: 1 new field (or 1 semantic change).

Total: 30–60 min including tests, depending on Option A vs B for the
pick rule.

## Open questions

1. Replace semantic of `mvt_mean` or add `mvt_mean_all`?
2. Option A (similarity-ranked depletion) vs Option B (SCORE-ranked
   depletion) for the candidate pick rule?
3. Should the `aisn_min_one_swap` fallback rule (force a single swap
   when no prescribed protocol is below the mean) survive when the
   threshold is cohort-wide? Probably yes — same behavior contract for
   the clinician.

## What this does NOT do

* Doesn't change the bootstrap or repeat strategies.
* Doesn't change the top-up step (still fills 7×ppd grid post-swap).
* Doesn't touch the `branch` taxonomy in the trace.
* Doesn't change the EngineState Protocol — `all_protocols` +
  `score_row` already cover it.
