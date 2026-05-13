# DataFrame schemas — column reference

Column reference for the canonical input/output frames of the
recommendation pipeline. Documentation-only — these are NOT validated
at runtime. The pipeline relies on the typed pipeline contracts
(`pipeline.py` § 1) for column enforcement; pandera was retired in F5e.

The three frames documented here are the boundary frames between the
data layer, the pipeline, and the engine. Internal pipeline stages
(SessionLevelFeatures, MergedFeatures, etc.) are documented in
`code_structure.md` § pipeline.

## `SessionSchema` — RGS session-level frame

Source: `RGSCohortRepository.find().session`.
Granularity: one row per (patient, prescription, session).

| Column | Type | Nullable | Constraint |
|---|---|---|---|
| `PATIENT_ID` | int | no | — |
| `PRESCRIPTION_ID` | int | no | — |
| `SESSION_ID` | int | yes | — |
| `PROTOCOL_ID` | int | yes | — |
| `PRESCRIPTION_STARTING_DATE` | datetime | no | — |
| `PRESCRIPTION_ENDING_DATE` | datetime | no | — |
| `SESSION_DATE` | datetime | yes | — |
| `WEEKDAY_INDEX` | int | yes | 0..6 (0=Monday) |
| `STATUS` | str | yes | one of `SessionStatus` enum values |
| `REAL_SESSION_DURATION` | int | yes | ≥ 0 |
| `PRESCRIBED_SESSION_DURATION` | int | yes | ≥ 0 |
| `SESSION_DURATION` | int | yes | ≥ 0 |
| `ADHERENCE` | float | yes | 0..1 |
| `DM_VALUE` | float | yes | — |

## `PPFSchema` — patient-protocol fit

Source: `RGSCohortRepository.find().ppf` (loaded from the precomputed
PPF parquet).
Granularity: one row per (patient, protocol).

| Column | Type | Nullable | Constraint |
|---|---|---|---|
| `PATIENT_ID` | int | no | — |
| `PROTOCOL_ID` | int | no | — |
| `PPF` | float | no | — |
| `CONTRIB` | object (list[float]) | no | per-subscale contribution vector |

## `ScoringSchema` — pipeline output

Source: `DataPipeline.process(cohort)`. Returned wrapped in
`ScoringOutput` (`pipeline.py` § 1).
Granularity: one row per (patient, protocol).

| Column | Type | Nullable | Constraint |
|---|---|---|---|
| `PATIENT_ID` | int | no | > 0 |
| `PROTOCOL_ID` | int | no | > 0 |
| `RECENT_ADHERENCE` | float | no | 0..1 |
| `DELTA_DM` | float | no | — |
| `PPF` | float | no | 0..1 |
| `CONTRIB` | list[float] | no | per-subscale contribution vector |
| `SCORE` | float | no | ≥ 0 |
| `USAGE` | int | no | ≥ 0 |
| `DAYS` | list[int] | no | weekdays the protocol is prescribed (0..6) |

## Where the column names come from

Every column name is a constant in `constants.py`. Use the constant,
never the literal string. The schemas above use the canonical column
NAMES (the constant values) rather than the Python attribute names.

## Why no runtime validation

Pandera was kept through the F4 refactor as documentation but never
exercised at runtime — recovery on `SchemaError` was dead code (no
validator was ever called on the load path). In F5e we removed
pandera from the dependency footprint; the typed pipeline contracts
(`pipeline.py` § 1) carry column enforcement where it matters.
