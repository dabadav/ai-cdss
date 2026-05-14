# `_enforce_rules` — planning doc v2

Supersedes `NORMALIZE_TO_SPEC_PLAN.md` (v1).

Engine architecture for grid-shape compliance, refactored around the
order of operations user prefers:

```
trim distinct (before swap)  →  swap loop  →  rebalance days
```

The swap loop already exists. v2 adds bookends:

1. `_normalize_input` — trims `n > 12` BEFORE the swap loop sees it
2. `_rebalance_days` — replaces today's `_top_up_coverage`, enforces
   both per-day count AND per-protocol day count

Both new steps live as private helpers; a thin `_enforce_rules(...)`
wrapper composes them with the swap loop for callers that want the
full pipeline.

---

## Why this order

| Concern | Old (swap → trim → fill) | New (trim → swap → rebalance) |
|---------|--------------------------|--------------------------------|
| Substitutes can be trimmed away | Yes (D3 needed to protect) | No — trim runs before swap |
| Trace clarity | `swaps` may not all appear in `final` | `swaps` and `final` agree |
| Day stabilization | One-sided (fills shortfalls only) | Two-sided (caps + floors) |
| Compute | Low | Medium (rebalance is a local-search loop) |
| Determinism | High | High if tie-breaks are explicit |

Trade: a bit more compute and a need for explicit tie-breaking, in
exchange for a clean separation between input shaping and output
shaping.

---

## Phase 1: `_normalize_input(patient_id, prescriptions)`

**Input**: the patient's prescribed protocol set, BEFORE the swap
loop. May have `n != 12`.

**Output**: at most 12 distinct prescribed protocols.

**Rule**:
- If `n <= 12`: return as-is (top-pool fill happens later in
  `_rebalance_days`).
- If `n > 12`: keep the 12 highest-scoring, drop the rest. DAYS lists
  for kept protocols are preserved verbatim (they remain candidates
  for trim in Phase 3 if the per-day cap is violated).

**Tie-breaks**: sort key = `(SCORE desc, PROTOCOL_ID asc)`.

**Trace**: append a `trimmed_input` event per dropped protocol with
score, days, reason `"n_distinct_input"`.

Inserted at `cdss.py:107`, right before:

```python
prescriptions = self._get_prescriptions(patient_id)
prescriptions = self._normalize_input(patient_id, prescriptions)   # NEW
```

---

## Phase 2: swap loop (unchanged)

`_decide_prescription_swap` operates on the trimmed-to-≤12 input.
`_update_existing_recommendations` runs the swap loop as today.
Substitutes inherit DAYS from removed protocol; trace records swaps.

No D3 protection needed — there's no later step trimming distinct
protocols, so substitutes are safe.

---

## Phase 3: `_rebalance_days(patient_id, rows)`

**Input**: 12 distinct prescribed protocols (post-swap) with their
DAYS lists.

**Output**: same protocols (no distinct-count changes here), DAYS
lists redistributed so that:

- Each protocol's `|DAYS|` is in `[3, 4]`
- Each day's protocol count is in `[4, 6]` (AISN soft cap)
- Total slots ≤ `days * protocols_per_day_max` (7 × 6 = 42)
- Total slots ≥ `days * protocols_per_day_min` (7 × 4 = 28)

Target total slots: try to land near `n * 3 = 36` (= 5.14 ppd avg).
If input has more slots than that, prefer to leave at 36-42; if
fewer, fill to ~36.

### Why `[3, 4]` per protocol fits

`n=12`, `days=7`, ppd target = 5, AISN allows ppd in [4, 6].
- 12 × 3 = 36 slots → avg ppd 5.14 (one day at 6 OK)
- 12 × 4 = 48 slots → avg ppd 6.86 (overshoot)

So the practical range is `n × 3 = 36` to `n × 3.5 = 42` total slots.
Most protocols at 3, a few at 4. Equivalent to: ppd in [4, 6] AND
per-protocol-days in [3, 4].

### Algorithm sketch (greedy local-search)

```python
def _rebalance_days(self, patient_id, rows):
    PROTO_MIN, PROTO_MAX = 3, 4
    PPD_MIN,   PPD_MAX   = self.protocols_per_day_min, self.protocols_per_day_max

    # Build day_protos: day -> [protocol_ids]
    # Build proto_days: protocol_id -> set(days)
    # ... (init from rows)

    # Phase 3a: cap per-protocol (trim protocols with > PROTO_MAX days)
    for p, days in proto_days.items():
        while len(days) > PROTO_MAX:
            # remove the day where day-count is highest
            d_remove = max(days, key=lambda d: len(day_protos[d]))
            days.remove(d_remove)
            day_protos[d_remove].remove(p)
            _trace_rebalance(p, d_remove, -1, "proto_max")

    # Phase 3b: cap per-day (trim days that exceed PPD_MAX)
    for d in range(self.days):
        while len(day_protos[d]) > PPD_MAX:
            # remove the protocol with the most days already, lowest score on tie
            victim = max(day_protos[d], key=lambda p: (len(proto_days[p]), -score_by[p]))
            day_protos[d].remove(victim)
            proto_days[victim].remove(d)
            _trace_rebalance(victim, d, -1, "ppd_max")

    # Phase 3c: floor per-day (fill days below PPD_MIN)
    for d in range(self.days):
        while len(day_protos[d]) < PPD_MIN:
            # candidates: protocols not on day d with |DAYS| < PROTO_MAX
            cand = [p for p in proto_days
                    if d not in proto_days[p] and len(proto_days[p]) < PROTO_MAX]
            if not cand:
                # fall back: top-pool unused protocols
                cand = [p for p in self._get_top_protocols(patient_id) if p not in proto_days]
            if not cand:
                _trace_topup(d, None, "exhausted")
                break
            pick = max(cand, key=lambda p: (
                -len(proto_days.get(p, set())),       # prefer fewest current days
                score_by.get(p, 0),                    # higher score wins tie
                -p,
            ))
            proto_days.setdefault(pick, set()).add(d)
            day_protos[d].append(pick)
            _trace_topup(d, pick, "existing" if pick in [r[PROTOCOL_ID] for r in rows] else "top_pool")

    # Phase 3d: floor per-protocol (lift protocols with |DAYS| < PROTO_MIN)
    for p in list(proto_days):
        while len(proto_days[p]) < PROTO_MIN:
            cand_days = [d for d in range(self.days)
                          if d not in proto_days[p] and len(day_protos[d]) < PPD_MAX]
            if not cand_days:
                break  # no headroom anywhere; accept proto under-coverage
            d_pick = min(cand_days, key=lambda d: len(day_protos[d]))
            proto_days[p].add(d_pick)
            day_protos[d_pick].append(p)
            _trace_rebalance(p, d_pick, +1, "proto_min")

    # Rebuild rows from proto_days
    return _rows_from_proto_days(rows, proto_days, patient_id)
```

Phases 3a-3d are executed in order. Each is monotonic with respect to
the constraint it addresses (caps don't grow, floors don't shrink) so
the loop terminates.

### Trace shape

```json
"rebalance": [
  {"protocol_id": 200, "day": 0, "delta": -1, "reason": "ppd_max"},
  {"protocol_id": 215, "day": 5, "delta": +1, "reason": "proto_min"}
],
"topup": [
  {"day": 6, "protocol_id": 233, "source": "top_pool"},
  {"day": 4, "protocol_id": null, "source": "exhausted", "deficit": 1}
]
```

`rebalance` for moves WITHIN existing protocols, `topup` for new
protocol insertions or exhausted days (kept compatible with today's
shape).

---

## `_enforce_rules` composer

Optional thin wrapper so callers don't have to remember the order:

```python
def _enforce_rules(self, patient_id, prescriptions, protocol_similarity):
    """End-to-end shape enforcement: trim input, swap, rebalance days."""
    prescriptions = self._normalize_input(patient_id, prescriptions)
    recommendations = self._update_existing_recommendations(
        patient_id, prescriptions, protocol_similarity
    )
    rows = recommendations.to_dict("records")
    rows = self._rebalance_days(patient_id, rows)
    return pd.DataFrame(rows).sort_values(by=PROTOCOL_ID).reset_index(drop=True)
```

Called from `recommend()`'s `update` branch. The `bootstrap` and
`repeat_skipped_week` branches stay on their existing paths but also
go through `_rebalance_days` at the universal post-step.

---

## Decisions to lock

### D1 — per-protocol day range

- **[3, 3]** — strict equality. 12 × 3 = 36 slots. Forces some day to
  hit 6 ppd. Most uniform output.
- **[3, 4]** (default) — average ~3.5. Allows 36-48 slots; pair with
  ppd cap [4, 6] (total ≤ 42).
- **[2, 4]** — looser; tolerates one rarely-played protocol.

Pick one.

### D2 — when `_normalize_input` ties on SCORE

Sort key `(SCORE desc, PROTOCOL_ID asc)` is deterministic but the
secondary `PROTOCOL_ID` is arbitrary. Acceptable for the trial since
within-tie protocols are clinically equivalent by definition (same
SCORE → same fit metric).

### D3 — total-slot target

- Option A: aim for exactly `n × 3 = 36` (most uniform; sometimes 6 ppd)
- Option B: aim for exactly `days × ppd = 35` (matches existing top-up
  target; means one protocol at 2 days, eleven at 3)
- Option C: just enforce [3, 4] per protocol and [4, 6] per day, no
  exact-slots target (default)

### D4 — what if rebalance cannot reach floor

If at the end of phases 3c/3d a day still has < 4 protocols or a
protocol still has < 3 days, the trace records the deficit but the
output remains non-compliant. Supervisor surfaces as
`per_day_count_mismatch` / `topup_exhausted`. **Default: accept.**

### D5 — when to run rebalance for bootstrap/repeat_skipped_week branches

Engine has three branches in `recommend()`:
- `bootstrap` (week 0): `_generate_new_recommendations` already picks
  top-N and schedules them.
- `repeat_skipped_week`: copies prior week verbatim.
- `update` (the normal case): swap loop.

For all three, run `_rebalance_days` at the universal post-step. For
`bootstrap`, also run `_normalize_input` (already does top-N selection
but the day distribution may need rebalance). For `repeat_skipped_week`,
rebalance only — input is by-definition already at-spec from a prior
week's enforcement (unless that prior week was non-compliant, in which
case fix it now).

### D6 — backward compatibility

`_top_up_coverage` is removed. Trace key `topup` is kept (same shape)
so existing backtest viewers don't break. New trace key `rebalance`
is additive — older viewers ignore it gracefully.

---

## Worked example

Same as v1 Example A, plus a rebalance step.

Input to `_enforce_rules`:

| protocol | source       | SCORE  | DAYS              | count |
|----------|--------------|--------|-------------------|-------|
| 218      | inherited    | 1.50   | [Mon, Wed]        | 2 |
| 209      | inherited    | 1.52   | [Tue, Thu]        | 2 |
| 222      | inherited    | 1.55   | [Wed, Fri]        | 2 |
| 224      | inherited    | 1.58   | [Tue, Sat]        | 2 |
| 214      | inherited    | 1.61   | [Sun]             | 1 |
| 226      | inherited    | 1.63   | [Mon, Thu]        | 2 |
| 215      | inherited    | 1.65   | [Tue, Fri, Sat]   | 3 |
| 203      | inherited    | 1.67   | [Mon, Wed, Fri]   | 3 |
| 219      | inherited    | 1.71   | [Mon]             | 1 |
| 206      | inherited    | 1.74   | [Wed, Fri]        | 2 |
| 208      | inherited    | 1.77   | [Tue, Thu, Sat]   | 3 |
| 223      | inherited    | 1.80   | [Sun]             | 1 |
| 200      | inherited    | 1.85   | [Mon, Tue, Wed, Sat] | 4 |
| 201      | inherited    | 1.91   | [Tue, Fri, Sat]   | 3 |

n=14. **Phase 1**: drop the 2 lowest-scoring → drop 218 (1.50) and
209 (1.52). Remaining 12 protocols.

Per-day counts now:
- Mon: 219, 226, 203, 200 = 4 ✓
- Tue: 224, 215, 208, 200, 201 = 5 ✓
- Wed: 222, 203, 206, 200 = 4 ✓
- Thu: 226, 208 = 2 ✗ (under)
- Fri: 222, 215, 203, 206, 201 = 5 ✓
- Sat: 224, 215, 208, 200, 201 = 5 ✓
- Sun: 214, 223 = 2 ✗ (under)

**Phase 2 (swap loop)**: runs on 12 protocols, may swap below-mean.
Assume 2 swaps happen, mostly preserving DAYS.

**Phase 3 (rebalance)**:
- 3a (per-proto cap): 200 has 4 days — OK (within [3, 4]).
- 3b (per-day cap): no day > 6.
- 3c (per-day floor): Thu has 2 → fill to 4. Sun has 2 → fill to 4.
  - Pick protocols with `|DAYS| < 4` and not on Thu: e.g. 222 (2 days), 224 (2 days), 226 (already Thu, skip), 219 (1 day)... add 219 → Thu, 222 → Thu. Thu now 4.
  - Sun: add 226, 224 → Sun. Sun now 4.
- 3d (per-proto floor): 214 (1 day), 223 (1 day), 219 (now 2)... lift to 3.
  - 214 add to under-covered days (any remaining under-covered? Mon at 4, but cap reached for ppd_min — Mon could go to 5/6 if any day has slack).
  - Continue until 214 hits 3 days or no headroom.

Final state: all protocols in [3, 4] days, all days in [4, 6] protocols.

---

## Tests

`tests/unit/test_enforce_rules.py` (new file):

1. **`test_normalize_input_drops_lowest_score`** — 15 prescribed protocols,
   2 lowest dropped, 12 highest preserved with DAYS intact.
2. **`test_normalize_input_passthrough_at_spec`** — exactly 12 distinct
   → no trim, no `trimmed_input` events.
3. **`test_rebalance_caps_per_protocol_max`** — protocol with 5 DAYS gets
   trimmed to 4; the day removed is the busiest one.
4. **`test_rebalance_caps_per_day_max`** — day with 7 protocols gets
   trimmed to 6; the victim is the protocol with the most other days.
5. **`test_rebalance_floors_per_day_min`** — day with 2 protocols gets
   filled to 4 by existing-protocol day expansion.
6. **`test_rebalance_floors_per_protocol_min`** — protocol with 1 day
   lifted to 3 by adding to under-covered days.
7. **`test_rebalance_topup_exhausted`** — every protocol at proto_max
   AND every day at ppd_min; further floor fills emit `exhausted`.
8. **`test_full_pipeline_at_spec_output`** — given n=15 input with
   uneven day distribution, output is n=12, all per-proto in [3,4],
   all per-day in [4,6].

---

## Out of scope

- Score-based swap criterion changes (the env-wide MVT fix on
  `feat/env-wide-mvt-threshold` is already in).
- Substitute selection logic (`_get_substitute`).
- Supervisor "TRIMMED" / "REBALANCED" badges — separate ticket.
- Backfilling `rebalance` / `trimmed_input` events into archived runs.

---

## Migration plan

1. Land `_normalize_input` first as a no-op pass-through when n ≤ 12.
   Test on existing data — should be identical output for clean
   prescriptions.
2. Enable the trim branch (n > 12 → trim). Test on real production
   data that triggers the violation.
3. Land `_rebalance_days` replacing `_top_up_coverage`. Existing
   `topup` trace key kept; new `rebalance` key added.
4. Remove `_top_up_coverage`.
5. Cut a new ai-cdss tag (v0.4.0 — minor bump, new behaviour).
