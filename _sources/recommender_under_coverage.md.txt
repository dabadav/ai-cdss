# Recommender under-coverage — investigation notes

## Symptom

Patient `4904`, week 4 (`2026-05-05 → 2026-05-11`).

The 03:00 cron CDSS run produced a single batch insert (`RECOMMENDATION_ID = a495eb0f-8035-4677-9290-4d0e773e2678`) covering several patients. For 4904 that batch only emitted **5 rows / 1 weekday (TUESDAY) / 5 protocols**. Other patients in the same batch got the expected **35 rows / 7 weekdays / 12 protocols**.

| patient_id | n_rows | n_days | n_protocols |
|------------|-------:|-------:|------------:|
| 4763       |     35 |      7 |          12 |
| 4768       |     35 |      7 |          12 |
| 4845       |     35 |      7 |          12 |
| 4973       |     35 |      7 |          12 |
| 4844       |     70 |      7 |          12 (duplicate run, separate issue) |
| **4904**   |  **5** |  **1** |       **5** |

Patient 4904 prior weeks show this is not a one-off:

| week | RID                                  | n_rows | n_days | n_proto | status   |
|------|--------------------------------------|-------:|-------:|--------:|----------|
| 0    | 2305792 + 3                          |   70   |   7    |   12    | Acc/Pend |
| 1    | 5824                                 |   30   |   6    |   12    | Acc      |
| 2    | 281                                  |   30   |   6    |   12    | Acc      |
| 3    | 0 (sentinel)                         |   30   |   6    |   12    | Acc      |
| 4    | a495eb0f-…                           |    5   |   1    |    5    | Pend ←   |

Weeks 1–3 already show the engine systematically dropping one weekday (the Monday at the trailing edge of this Tue-start patient's week). Week 4 collapsed even further — one weekday only.

## Code path

`ai_cdss/processing/feature_builder.py:188 build_prescription_days(session_df, patient_df, scoring_date)`:

```python
anchors = self._last_completed_week_window(patient_df, scoring_date)   # [start + 7*(w-1), start + 7*w)
df      = session_df.merge(anchors, on=PATIENT_ID, how="inner")
df      = df.loc[(df.SESSION_DATE >= df.week_start) & (df.SESSION_DATE < df.week_end)]
prescribed_days = df.groupby([PATIENT_ID, PROTOCOL_ID])[WEEKDAY_INDEX].agg(lambda x: sorted(x.unique()))
```

Key fact about `session_df`: it comes from `rgs_interface/sql/query.sql`, which is `prescription_plus LEFT JOIN session_plus … AND sp.STATUS IN ('CLOSED','ABORTED')`. Therefore `SESSION_DATE = sp.STARTING_DATE`, which is **NULL when the prescription has no session attached**. Those rows are dropped by the `SESSION_DATE >= week_start` predicate.

So `DAYS` per `(patient, protocol)` is the set of weekdays on which the patient **actually performed** a (CLOSED or ABORTED) session in the previous patient-aligned week. It is **not** prescribed coverage.

This `DAYS` flows into the scoring frame and from there into `cdss.py`:

- `CDSS._get_prescriptions(p)` keeps only rows where `DAYS` is a non-empty list — i.e. only protocols the patient actually played somewhere last week.
- `_is_week_skipped` checks `USAGE_WEEK == 0` for those scheduled rows; if **all** zero it returns the prior prescription verbatim (`_repeat_prescriptions`). If even one row had usage, it goes into the swap loop.
- `_update_existing_recommendations` keeps non-swapped rows and substitutes underperformers via `_swap_protocol`. Crucially, the substitute inherits the swapped protocol's `DAYS` list verbatim:
  ```python
  substitute_row[DAYS] = prescriptions.loc[prescriptions[PROTOCOL_ID] == protocol_id, DAYS].values[0]
  ```
- `recommender.py:_transform_recommendations` does `recommendations.explode(DAYS).rename({DAYS: 'WEEKDAY'})` and writes one staging row per `(protocol, weekday)`.

So if the patient played sessions on only one weekday last week, the DAYS lists will all be `[that_weekday]`, and the next prescription collapses to that single column — exactly what 4904 shows.

## Test corroboration

`tests/unit/test_feature_builder.py::test_build_prescription_days_with_fixture` exercises this with two patients. The fixture sets `SESSION_DATE` per row and `WEEKDAY_INDEX` accordingly:

```python
SESSION_DATE      = [Tue,  Wed,  Tue]
WEEKDAY_INDEX     = [1,    2,    1  ]
PATIENT_ID        = [1,    1,    2  ]
STATUS            = [CLOSED, NOT_PERFORMED, CLOSED]
```

The output for patient 1 is `DAYS=[1, 2]` (Tue + Wed) and for patient 2 `DAYS=[1]` (Tue only). The function does **not** filter by `STATUS`, so a `NOT_PERFORMED` row with a `SESSION_DATE` would still contribute. But because the SQL `LEFT JOIN` only keeps `STATUS IN ('CLOSED', 'ABORTED')`, rows that are `NOT_PERFORMED` come through with `SESSION_DATE = NULL` and are dropped by the date filter.

The `test_week_skip_patient` block (currently commented out) documents the design intent: when last week was fully skipped (`USAGE_WEEK == 0` for all scheduled), the engine repeats the previous prescription as-is. There is no test covering the case where last week was *partially* used (one weekday only) — that scenario silently produces a one-day prescription with no warning.

There is **no test** asserting that the produced staging rows ever cover the full 7 weekdays / `protocols_per_day × days` slots, and **no test** asserting `DAYS` length is bounded below in either bootstrap or update paths. The bootstrap branch (`_generate_new_recommendations`) is the only place that fans out across `range(days)` via `_schedule_protocols`; the update branch never re-fans, it only inherits.

## Why 4904 specifically

Same RID emitted 7-day output for five other patients, so this is **not a global engine failure**. It is a per-patient input issue: 4904's `session_plus` rows in `[2026-04-28, 2026-05-05)` only carry `SESSION_DATE` on Tuesday. That makes `DAYS = [TUESDAY]` for 5 protocols, which is the single carrier of weekday coverage into the next week's prescription. (Verification still pending: a direct query of `session_plus` filtered to that window will confirm.)

## Why weeks 1–3 already missed Monday

Patient started 2026-04-07 (Tue). Their patient-aligned week ends on a Monday. The recommender's "last completed week" window is correct (`[start + 7*(w-1), start + 7*w)`), but the *prescriptions it inherits from* contain only the weekdays the patient actually performed sessions on. If the patient never performed a Monday session in any prior week, no Monday prescription ever appeared. The Monday gap therefore propagated forward week-on-week through `_update_existing_recommendations` keeping inherited DAYS verbatim.

## Mitigations to consider (not changed)

1. In `_update_existing_recommendations`, if the union of `DAYS` across kept + substituted rows covers fewer than `days × protocols_per_day` slots, top up via `_schedule_protocols` from `_get_top_protocols`. This restores the AISN trial's "7 days / 4–6 per day / 12 distinct" mandate when the patient's actual usage was thin.
2. Surface low-coverage runs early — the new `n_rows / n_days / n_protocols` per-patient fields (added to the run JSON in `~/.ai_cdss/logs/`) make this trivial to alert on.
3. Add a behavior test: bootstrap → 7-day coverage; update with one-day usage → engine still emits a multi-day plan (assuming we land mitigation #1).
4. Decide whether `DAYS` should derive from **prescribed** weekdays (`prescription_plus` rows last week) instead of **performed** sessions, so the trial's prescribed schedule is preserved across weeks regardless of compliance.

## Logging change made in this investigation

`ai_cdss/interface/recommender.py:_process_patient` now records, per patient, the staging-shape derived from `prescription_df`:

```python
n_rows      = int(len(prescription_df))
n_days      = int(prescription_df["WEEKDAY"].nunique())
n_protocols = int(prescription_df["PROTOCOL_ID"].nunique())
logger.info("Patient %s prescription shape: n_rows=%d n_days=%d n_protocols=%d", ...)
result.update({"n_rows": n_rows, "n_days": n_days, "n_protocols": n_protocols})
```

Those values land both in the structured run log at `~/.ai_cdss/logs/{run_id}_{date}.json` (per-patient) and in the `INFO` line.
