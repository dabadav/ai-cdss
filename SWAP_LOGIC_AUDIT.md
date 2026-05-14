# Swap Logic Audit — `ai-cdss` v0.3.1

Audit of `src/ai_cdss/cdss.py` against the intended swap semantics.
Performed 2026-05-14. Source HEAD: `57728fd` (tag `v0.3.1`).

Follow-up patches landed on branch `feat/env-wide-mvt-threshold`:
- `5102b0f` — fix invariant 1 (env-wide MVT mean + sorted swap order)
- this branch — fix observation (a) brittle `_get_unused_candidates`

## Intended invariants

1. **MVT criterion** is computed over **all** protocols available to the
   patient, but **applied** only to the protocols prescribed in the
   patient's last week. I.e. mark a prescribed protocol for swap iff
   `score(p) < mean(score over all protocols)`.
2. **Candidate pool** for a swap is **last week's non-prescribed
   protocols**. A protocol that was just swapped out does **not**
   re-enter the pool — once removed, gone for this week.

---

## Invariant 1: MVT over all protocols → applied to prescribed

**Status: VIOLATED**

`cdss.py:349-357`:

```python
def _decide_prescription_swap(self, patient_id: int) -> List[int]:
    prescriptions = self._get_prescriptions(patient_id)   # filtered to DAYS non-empty = prescribed
    return prescriptions[
        prescriptions[SCORE].transform(lambda x: x < x.mean())
    ].PROTOCOL_ID.to_list()
```

`_get_prescriptions` (`cdss.py:580-588`) filters `self.scoring` to rows
with non-empty `DAYS` — i.e. prescribed protocols only. The
`x.mean()` inside `.transform()` therefore operates on **prescribed
SCOREs only**, not the full protocol set. Mean is biased toward the
~12 prescribed protocols rather than the ~23-protocol whitelist.

### Suggested fix

```python
def _decide_prescription_swap(self, patient_id: int) -> List[int]:
    all_scores = self.scoring.loc[
        self.scoring[PATIENT_ID] == patient_id, SCORE
    ]
    mean_all = all_scores.mean()
    prescribed = self._get_prescriptions(patient_id)
    return prescribed.loc[
        prescribed[SCORE] < mean_all, PROTOCOL_ID
    ].tolist()
```

---

## Invariant 2: Candidate pool = non-prescribed; swapped-out stay out

**Status: HOLDS** (correctly, but indirectly)

`cdss.py:221-275`:

| Line | Effect |
|------|--------|
| 223  | `protocols_excluded = prescriptions[PROTOCOL_ID].tolist()` — seeds excluded set with **all 12 last-week-prescribed** protocols |
| 240  | `_get_protocol_similarities(..., protocols_excluded)` filters those out → candidate pool == non-prescribed |
| 275  | After each swap, `protocols_excluded.append(sub_id)` — newly-added substitute joins excluded so the next swap can't pick it |

Swapped-out protocol `A` was already in `protocols_excluded` (it was
prescribed) and is never removed → stays excluded → never re-enters
the pool. Invariant holds.

Important precondition: `protocols_to_swap ⊆ prescriptions[PROTOCOL_ID]`.
True via construction in `_decide_prescription_swap` and
`_get_lowest_performing_protocol`.

---

## Other observations

### a) `_get_unused_candidates` bypasses the excluded list **(FIXED)**

`cdss.py:_get_substitute` previously called
`self._get_unused_candidates(usage)` directly. `usage` is the full
per-patient usage Series — not filtered by `protocols_excluded` — so
`unused_candidates` could contain protocols already in this week's
prescription or just-added substitutes. Correctness was preserved only
indirectly: `_select_most_similar` intersects candidates with
`similarities` (which IS excluded-filtered), so excluded candidates
fell out silently.

Brittle pattern — a future refactor that decoupled similarity
filtering from candidate selection would have leaked excluded
protocols back into the pool. Now patched to filter explicitly:

```python
unused_candidates = [
    p for p in self._get_unused_candidates(usage) if p not in excluded
]
```

No functional change in normal cases; intent now matches behaviour.

### b) Forced swap (`aisn_min_one_swap`) picks from prescribed

`cdss.py:227-230, 436-443`:

```python
if not protocols_to_swap:
    forced = self._get_lowest_performing_protocol(patient_id)  # min SCORE among prescribed
    protocols_to_swap.append(forced)
```

Correctly draws from the prescribed set — forcing a swap requires an
already-prescribed protocol to displace. Consistent with invariant 2.

### c) Substitute-not-found falls back to a no-op swap

`_swap_protocol` (`cdss.py:388-389`):

```python
# Else return same protocol
return self._get_scores(patient_id, protocol_id)
```

When `_get_substitute` returns `None`, the protocol "swaps with
itself" and stays in the schedule. Supervisor backtest surfaces this
as the `swap_returned_same_protocol` warning.

#### When does `_get_substitute` return `None`?

Two consecutive branches must both yield no candidate. Walk-through:

```
usage         = scoring[PATIENT_ID == pid][USAGE]            # full per-patient series
similarities  = protocol_similarity rows where               # candidate pool view
                  PROTOCOL_A == protocol_id
                  PROTOCOL_B != PROTOCOL_A
                  PROTOCOL_B ∉ protocols_excluded

Branch 1 (unused):
    unused = [p for p in usage[usage == 0].index            # zero-usage protocols
                       if p not in protocols_excluded]
    if unused:
        return _select_most_similar(unused, similarities)   # ← may still return None
                                                            #   if similarities rows
                                                            #   miss every unused

Branch 2 (least-used among top-similar):
    top_similar = similarities.nlargest(5, SIMILARITY).PROTOCOL_B
    least_used  = candidates in top_similar with min(usage)
    if least_used:
        return _select_most_similar(least_used, similarities)
```

Assuming the system's invariants:
- `USAGE` coerced to 0 via `imputer.init_metrics` (`imputer.py:33`) — no
  NaN-driven misses.
- `protocol_similarity` is a full matrix over the whitelist — no
  similarity-row gaps.
- `scoring` has a row per (patient, whitelist protocol) — no
  scoring/similarity index mismatch.

Under these invariants there is **exactly one** real failure mode:

**Candidate pool exhausted by exclusions.** With ~23 therapy protocols
and 12 prescribed, `protocols_excluded` starts at 12 and the available
pool is `W − P ≈ 11`. Each successful swap appends its substitute to
`protocols_excluded` (`cdss.py:275`), so the pool shrinks by 1 per
swap. After ≈11 cumulative swaps in the same week, the pool reaches 0
→ `similarities` (post-exclusion) is empty → branch 1's
`_select_most_similar` returns None (empty intersection with
similarities), and branch 2's `top_similar` is also empty → `_get_substitute`
returns None → `_swap_protocol` no-ops via the same-protocol fallback.

**Case 2 (all zero-usage protocols already in `protocols_excluded`) is
not a failure on its own**: branch 1 yields `[]` and the call falls
through to branch 2, which still finds candidates as long as the pool
is non-empty.

#### Why pool exhaustion is rare in practice

`protocols_to_swap` = prescribed protocols scoring below the env-wide
MVT mean. The recommender tends to seed prescriptions with the
top-scoring protocols (above env mean), so early weeks typically swap
0-2 protocols and the forced-swap floor (≥1) lifts that to exactly 1
when nothing is genuinely below the mean. Reaching 11 swaps in a
single week requires nearly every prescribed protocol to score below
the env-wide mean — implausible unless score distribution flips
relative to the whitelist.

Mitigations available (not yet applied):
- Skip swap (don't add row, don't keep stale) once pool is exhausted.
- Widen pool to non-therapy diagnostic protocols when therapy pool
  empties (probably undesirable).
- Surface trace field `pool_exhausted_at` per swap so the supervisor
  view can flag retroactively.

---

## Summary

| # | Invariant | Status | Lines |
|---|-----------|--------|-------|
| 1 | MVT over all, applied to prescribed | **FIXED** on `feat/env-wide-mvt-threshold` | 349-357, 580-588 |
| 2 | Candidate pool = non-prescribed, swapped-out stay out | HOLDS | 221-275 |
| a | `_get_unused_candidates` doesn't filter excluded | **FIXED** (explicit filter) | 408, 454-457, 497-508 |
| b | Forced swap picks from prescribed | correct | 227-230, 436-443 |
| c | Substitute-not-found → no-op swap | by design; 6 failure modes catalogued above | 388-389 |
