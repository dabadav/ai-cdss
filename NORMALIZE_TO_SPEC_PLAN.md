# `_enforce_spec` — planning doc

Engine post-step to enforce the AISN trial grid shape:
`n == 12` distinct protocols, `days == 7`, `protocols_per_day == 5`.
Today the engine only enforces these as **floors** via `_top_up_coverage`;
nothing trims when the input is **over** spec. Real production data
routinely violates the cap (e.g. 18 distinct, 7+ protocols on Tuesday)
and the engine inherits those violations verbatim.

**Approach**: single `_enforce_spec(patient_id, rows)` method that trims
DOWN and fills UP in one pass. Absorbs (and replaces) the existing
`_top_up_coverage` so callers see one shape-enforcement step instead
of two adjacent ones.

## Problem statement

`CDSS.recommend()` post-pipeline currently:

1. Swap loop (`_update_existing_recommendations`) — substitutes preserve
   the swapped-out protocol's DAYS list 1:1, so the (day × slot) shape
   of the input is inherited.
2. `_top_up_coverage` — fills days that have fewer than
   `protocols_per_day` protocols. Never trims.

Net: if production prescribed `n > 12` or any day had `count > ppd`, the
engine output preserves those violations. Backtest surfaces them as
`distinct_protocol_count_mismatch`, `per_day_count_mismatch`,
`total_slots_mismatch`.

## Root cause

```python
# _top_up_coverage
while len(day_protos[day]) < self.protocols_per_day:
    pick = ...
    day_protos[day].append(pick)
```

One-sided loop. No symmetric `while ... > self.protocols_per_day`
loop exists, and no enforcement of `n == self.n`.

## Proposed solution sketch

Single `_enforce_spec` method replacing `_top_up_coverage` at the
universal post-step in `recommend()`:

```python
def _enforce_spec(self, patient_id: int, rows: list[dict]) -> list[dict]:
    # Phase 1: trim distinct protocols if n > N (record to trace["trimmed"])
    # Phase 2: trim per-day if any day > ppd (record to trace["trimmed"])
    # Phase 3: drop protocols whose DAYS became empty
    # Phase 4: fill per-day if any day < ppd, by either:
    #   (a) giving an existing protocol an extra day, or
    #   (b) introducing a top-scored unused protocol
    #   (records to trace["topup"], same semantics as today)
    # Phase 5: if after phase 4 n < N, top-pool fills up to N distinct
    return rows
```

Phases 4-5 carry over today's `_top_up_coverage` logic verbatim — same
filler-pool ordering, same trace shape — so existing tests still cover
that branch. Phases 1-3 are net new.

The old `_top_up_coverage` is removed; this is the single shape-
enforcement step.

## Decisions to lock before coding

### D1 — Trim policy: which protocol/day pair leaves?

Three plausible policies. Pick one or describe a different one.

| Policy | Rule | Pros | Cons |
|--------|------|------|------|
| **Lowest score wins removal** | Drop pairs whose protocol has the worst SCORE for the patient | Aligns with the MVT philosophy ("low SCORE = poor fit") | Same scores on tie → arbitrary; may strip protocols just substituted in |
| **Newest addition leaves** | Prefer to trim recently swapped-in / top-up'd protocols | Preserves historical continuity | Requires tracking insertion order through the pipeline |
| **Coverage-redundant leaves** | Drop pairs from protocols that already cover the most days | Maximises diversity per day | More complex; can interact badly with low-score-already-trimmed protocols |

**Default proposal**: lowest score wins removal, with `(SCORE, PROTOCOL_ID)`
as the sort tuple to make tie-breaking deterministic.

### D2 — Trim ordering: distinct count first, then per-day, or interleaved?

- **Distinct first** — trim n down to 12 globally, then look at per-day.
  Risk: a protocol that would have been per-day trimmed gets dropped
  entirely from the distinct trim.
- **Per-day first** — fix days down to 5, then check distinct count.
  Risk: per-day trims may create unused protocols (DAYS empty), which
  drop out → distinct count may go below 12, top-up fills with new
  protocols, distinct may end up wrong shape again.
- **Interleaved** — global goal: pass both constraints simultaneously.
  Hardest to reason about.

**Default proposal**: distinct first, then per-day. Each step records
which protocol/day pairs left and why.

### D3 — Interaction with `aisn_min_one_swap`

The forced-swap floor at `cdss.py:227-230` ensures `len(protocols_to_swap) >= 1`.
If `_normalize_to_spec` drops the substitute that was just added by
the forced swap, the AISN rule is silently violated — engine reports
`swap_count >= 1` (the substitute was added before trim) but the
post-trim final has no actual swap.

**Default proposal**: when trimming, never remove a protocol that
appears as `added` in `trace["swaps"]`. If forced to choose between two
protocols of equal score for removal, prefer the non-swap one.

### D4 — Trace plumbing

Add `trace["trimmed"]` so the supervisor can show "this row was
trimmed by normalize_to_spec":

```python
trace["trimmed"].append({
    "protocol_id": ...,
    "removed_days": [int(d)],          # the specific day pair pulled
    "reason":       "n_distinct" | "per_day_count",
    "score":        float(...),
})
```

Supervisor work (future):
- Show trimmed protocols as a dimmed-out row in the heatmap with a
  "TRIMMED" badge — similar to the SWAP FAILED badge already added.

### D5 — Failure mode: trim leaves day below ppd

After trim step 2, a day may now have `count < protocols_per_day`.
`_top_up_coverage` will fill it, but the pool may be exhausted (every
non-prescribed protocol already excluded). Same failure mode as
substitute-not-found.

**Default proposal**: rely on existing `topup_exhausted` warning;
don't add new logic for now.

### D6 — Scope: only `update` branch, or all branches?

- `_update_existing_recommendations` — main case (clinician prescribed n protocols).
- `_generate_new_recommendations` (bootstrap, week 0) — calls `_schedule_protocols` which already targets `n * days_target` slots and could produce its own shape issues.
- `_repeat_prescriptions` (skipped week) — copies prior week verbatim.

**Default proposal**: apply `_normalize_to_spec` to the universal
post-step in `recommend()` (line ~132, after `_top_up_coverage`'s
current call site). Catches all three branches in one place.

## Implementation outline (assuming default proposals on D1–D6)

```python
def _enforce_spec(self, patient_id: int, rows: list[dict]) -> list[dict]:
    # Score lookup for tie-breaking. Pull from self.scoring so we use the
    # same snapshot the swap loop ran against.
    score_by_pid = (
        self.scoring[self.scoring[PATIENT_ID] == patient_id]
        .set_index(PROTOCOL_ID)[SCORE]
        .to_dict()
    )

    trace = getattr(self, "_trace", None)

    # Set of protocols added by swap loop; protected from removal where possible.
    added_by_swap: set[int] = set()
    if trace is not None:
        for s in trace.get("swaps") or []:
            if s.get("substitute_found"):
                added_by_swap.add(int(s["added"]))

    def _trim_record(pid_proto: int, days_removed: list[int], reason: str):
        if trace is None:
            return
        trace.setdefault("trimmed", []).append({
            "protocol_id":  int(pid_proto),
            "removed_days": [int(d) for d in days_removed],
            "reason":       reason,
            "score":        float(score_by_pid.get(pid_proto, 0.0)),
        })

    # ── Phase 1: trim distinct if n > N ──────────────────────────────
    if len(rows) > self.n:
        # Prefer to drop non-swap-added first; within each group sort by
        # (SCORE asc, PROTOCOL_ID asc) — lowest goes first.
        rows.sort(key=lambda r: (
            int(r[PROTOCOL_ID]) in added_by_swap,   # False (0) first → drop these
            float(score_by_pid.get(int(r[PROTOCOL_ID]), 0.0)),
            int(r[PROTOCOL_ID]),
        ))
        excess = rows[: len(rows) - self.n]
        rows   = rows[len(rows) - self.n :]
        for r in excess:
            _trim_record(r[PROTOCOL_ID], list(r.get(DAYS) or []), "n_distinct")

    # ── Phase 2: trim per-day if any day > ppd ───────────────────────
    day_pool: dict[int, list[tuple[int, float]]] = {d: [] for d in range(self.days)}
    for r in rows:
        for d in r.get(DAYS) or []:
            day_pool[int(d)].append((int(r[PROTOCOL_ID]), float(score_by_pid.get(int(r[PROTOCOL_ID]), 0.0))))
    for d, items in day_pool.items():
        if len(items) <= self.protocols_per_day:
            continue
        items.sort(key=lambda x: (
            -(int(x[0]) in added_by_swap),
            -x[1],
            x[0],
        ))
        keep = {p for p, _ in items[: self.protocols_per_day]}
        for r in rows:
            if d in (r.get(DAYS) or []) and int(r[PROTOCOL_ID]) not in keep:
                r[DAYS] = [x for x in r[DAYS] if int(x) != d]
                _trim_record(r[PROTOCOL_ID], [d], "per_day_count")

    # ── Phase 3: drop protocols whose DAYS became empty ──────────────
    rows = [r for r in rows if r.get(DAYS)]

    # ── Phase 4: fill per-day deficits (existing top-up logic) ───────
    # (verbatim from today's _top_up_coverage — moved inline)
    proto_to_row: Dict[int, dict] = {r[PROTOCOL_ID]: r for r in rows}
    day_protos: Dict[int, list[int]] = {d: [] for d in range(self.days)}
    for r in rows:
        pid_proto = r[PROTOCOL_ID]
        for d in r.get(DAYS, []) or []:
            if pid_proto not in day_protos[d]:
                day_protos[d].append(pid_proto)
    existing = list(proto_to_row.keys())
    top_pool = [p for p in self._get_top_protocols(patient_id) if p not in proto_to_row]
    filler_pool = existing + top_pool
    for day in range(self.days):
        while len(day_protos[day]) < self.protocols_per_day:
            pick = next((p for p in filler_pool if p not in day_protos[day]), None)
            if pick is None:
                if trace is not None:
                    trace["topup"].append({
                        "day": day, "protocol_id": None,
                        "source": "exhausted",
                        "deficit": self.protocols_per_day - len(day_protos[day]),
                    })
                break
            day_protos[day].append(pick)
            if pick in proto_to_row:
                cur_days = list(proto_to_row[pick].get(DAYS, []) or [])
                proto_to_row[pick][DAYS] = sorted(set(cur_days + [day]))
                source = "existing"
            else:
                new_row = self._get_scores(patient_id, pick)
                new_row[PROTOCOL_ID] = pick
                new_row[PATIENT_ID] = patient_id
                new_row[DAYS] = [day]
                proto_to_row[pick] = new_row
                source = "top_pool"
            if trace is not None:
                trace["topup"].append({
                    "day":         day,
                    "protocol_id": int(pick),
                    "source":      source,
                })

    # Phase 5 (n < N after phases 1-3) is naturally handled by phase 4's
    # top_pool branch — it introduces new protocols to fill day gaps.
    # No separate "distinct fill" step needed.

    return list(proto_to_row.values())
```

Insertion in `recommend()`:

```python
rows = recommendations.to_dict("records")
rows = self._enforce_spec(patient_id, rows)   # replaces _top_up_coverage
```

Trace init in `recommend()` adds `"trimmed": []` alongside `swaps`, `topup`.

## Tests

Add to `tests/unit/test_cdss_recommend.py`:

1. **`test_normalize_trims_excess_distinct`** — 14 prescribed protocols
   → output has 12, 2 trimmed entries.
2. **`test_normalize_trims_per_day_overflow`** — day 0 has 7 protocols,
   trim to 5, lowest-scoring 2 lose their day 0 entry.
3. **`test_normalize_preserves_swap_added`** — a protocol added by the
   swap loop is at low score; trim picks a non-swap protocol of equal
   score instead.
4. **`test_normalize_no_op_when_at_spec`** — exactly-spec input passes
   through unchanged, no `trimmed` entries.
5. **`test_topup_runs_after_trim`** — trim causes day to drop below
   ppd; top-up fills it back.

## Worked examples — what each policy choice produces

### Example A — `n = 14`, per-day already in spec (D1 + D6)

Input from `_update_existing_recommendations`: patient 4711 has 14
distinct prescribed protocols after the swap loop (production
inherited 14, swap loop kept 14). Every day has exactly 5 protocols.

Scoring snapshot for this patient (excerpt — sorted by SCORE asc):

| protocol | source        | SCORE  | DAYS              |
|----------|---------------|--------|-------------------|
| 218      | inherited     | 1.50   | [Mon, Wed]        |
| 209      | swap-added    | 1.52   | [Tue, Thu]        |
| 222      | inherited     | 1.55   | [Wed, Fri]        |
| 224      | inherited     | 1.58   | [Tue, Sat]        |
| 214      | inherited     | 1.61   | [Sun]             |
| 226      | inherited     | 1.63   | [Mon, Thu]        |
| 215      | inherited     | 1.65   | [Tue, Fri, Sat]   |
| 203      | inherited     | 1.67   | [Mon, Wed, Fri]   |
| 219      | swap-added    | 1.71   | [Mon]             |
| 206      | swap-added    | 1.74   | [Wed, Fri]        |
| 208      | swap-added    | 1.77   | [Tue, Thu]        |
| 223      | swap-added    | 1.80   | [Sun]             |
| 200      | inherited     | 1.85   | [Wed, Sat]        |
| 201      | inherited     | 1.91   | [Tue, Fri]        |

Need to drop 2 protocols. Compare three D1 policies:

**Lowest score wins (default)** — drop 218 (1.50) + 209 (1.52)
- 209 is a swap-added → swap visible in trace is now silently undone.
- Final keeps 12 protocols.

**Lowest score wins, protect swap-added (default + D3)** — drop 218 + 222
- Both inherited. Swap loop's work preserved.
- Final keeps 12 protocols including all 5 swap-added.

**Newest addition leaves** — drop 219 + 223 (last two swap-added)
- All inherited preserved; swap loop's work undone in part.
- Surprising clinically: the engine spent compute deciding swaps,
  then the post-step quietly removes them.

The first row of that table illustrates exactly why D3 (protect
swap-added) matters: without it, your policy can erase the swap loop's
output and the `aisn_min_one_swap` violation reappears in disguise.

### Example B — `n = 12` but Monday has 7 protocols (D2)

Input: 12 distinct protocols, but Monday's day list has 7 entries.

| protocol | SCORE  | DAYS (subset relevant) |
|----------|--------|------------------------|
| 218      | 1.50   | [Mon, Wed]             |
| 222      | 1.55   | [Mon, Wed]             |
| 224      | 1.58   | [Tue, Sat]             |
| 214      | 1.61   | [Sun]                  |
| 226      | 1.63   | [Mon, Thu]             |
| 215      | 1.65   | [Mon, Sat]             |
| 203      | 1.67   | [Mon, Wed, Fri]        |
| 219      | 1.71   | [Mon]                  |
| 206      | 1.74   | [Wed, Fri]             |
| 208      | 1.77   | [Tue, Thu]             |
| 223      | 1.80   | [Sun]                  |
| 200      | 1.85   | [Mon, Sat]             |

Monday list: {218, 222, 226, 215, 203, 219, 200} = 7 protocols.
Need to drop 2 from Monday only.

**D2 = distinct first, then per-day** (default):
- Phase 1: n already = 12 → no trim.
- Phase 2: pick the 2 lowest-scoring Monday protocols → drop Mon from
  218 (1.50) and 222 (1.55).
- 218 now has DAYS = [Wed], 222 now has DAYS = [Wed]. Both kept.
- Final: n = 12, Monday has 5 protocols ✓

**D2 alt = per-day first, then distinct**:
- Phase 1 (per-day): same Monday trim as above → 218 DAYS = [Wed], 222 DAYS = [Wed].
- Phase 2 (distinct): n is still 12, no further trim.
- Same result.

In this case D2 ordering doesn't matter. But consider Example A re-run:
if n = 14 and Monday also = 7, distinct-first trims 218 + 222 entirely
(both happen to have low scores) → Monday automatically drops to 5
without needing a per-day pass. Per-day-first would shave Mondays
first (218, 222 stay alive on Wed), then distinct-trim has to pick
different victims. **Distinct-first is fewer dropped pairs.**

### Example C — forced swap meets normalize trim (D3)

Patient at week 3, all 12 prescribed protocols scored ABOVE env MVT
mean (clean run). `_decide_prescription_swap` returns `[]` → AISN
forced-swap fires → engine picks lowest-scoring prescribed (e.g.
protocol 200, score 1.85) and swaps it with 233 (unused-pool pick,
score 1.45).

Result: 12 protocols, but 233 (the forced substitute) now scores
1.45 — the lowest in the patient's set.

Now imagine the patient's prescription also had n = 13 (production
prescribed one extra). Phase 1 trim needs to drop 1.

**Default (D3 protects swap-added)**:
- Eligible to drop: inherited protocols only.
- Pick the lowest-scoring inherited (say 214, 1.61).
- 233 (the forced substitute) survives.
- AISN min-1 invariant holds ✓.

**Without D3**:
- Eligible to drop: any protocol.
- Lowest score is 233 (1.45) → 233 gets dropped.
- Forced swap is silently undone. AISN min-1 violation reappears.
- The trace will show a swap that didn't actually land in `final`.

### Example D — trim drops day below `ppd` (D5)

Patient at week 2, n = 12, Monday has 7 protocols (Example B). After
Phase 2 trims 218 and 222 from Monday, Monday now has 5 protocols ✓.
But also: 222 originally had [Mon, Wed], now has [Wed] only — fine.

Trickier variant: trim more aggressively. Suppose **Wednesday had
only 4 protocols going in, plus 222 (so 5)**, and 222 also covers
Monday (over-coverage day). If Phase 2 strips 222 from Wednesday
trying to deal with Monday overflow… we wouldn't, because Wednesday
isn't over-covered.

But suppose Phase 1 drops protocol 222 entirely (n was 13, 222 was
lowest). Then Wednesday loses one filler → Wednesday now has 4
protocols. Phase 4 fills with `top_pool` → introduces a new protocol
on Wednesday.

If the top_pool is empty (all whitelist protocols already in
prescription somehow → near-impossible with whitelist size 23 and
n = 12, but theoretically possible after multiple swaps + trims),
Phase 4 emits `topup_exhausted` event. **D5 default**: accept this
event, do nothing further. The supervisor surfaces it as a
`topup_exhausted` warning.

## Open questions for review

- D1–D6 default proposals OK? Pick one specifically if you disagree.
- Should trim respect AISN min-1 swap rule beyond just "don't remove the
  substitute"? E.g. roll back the whole swap if trim eats the substitute?
- Per-day trim picks day-level winners; should we instead pick protocol-level
  (drop the entire protocol if its score is low)? Current proposal can leave
  a low-scoring protocol covering one day instead of dropping it outright.

## Out of scope

- Reworking the swap loop's substitute selection.
- Supervisor TRIMMED badge UI — separate ticket once engine ships.
- Backfilling `trimmed` events into existing archived backtest runs.

## Migration

- `_top_up_coverage` is **removed** (logic absorbed into `_enforce_spec`).
- Any external caller of `cdss._top_up_coverage` needs to switch to
  `_enforce_spec` or to the public `recommend()` entrypoint. Internal
  callers: none beyond `recommend()` itself.
