# `ai-cdss` Readability Refactor

Sibling project alongside `D:/Projects/RGS/ai-cdss/` (original). Same
package name (`ai_cdss`) so you can switch which one your venv installs
to compare implementations.

Branch: `readable` (off `master` @ v0.3.1).

## Mental model

Data is a tensor with axes:

    patient × protocol × prescription × session × time → values

Every pipeline operation lives at a **level of aggregation** on that
tensor — reducing over time, over sessions, over protocols, or
treating each (patient, protocol) cell independently. File structure
**mirrors** these levels.

## Direction history

**Phase 1 (initial cut)** — Split `cdss.py` (588 lines) into a
`recommend/` subpackage of 10 small files. Each file ≤ 240 lines,
each function ≤ 30 lines, named clearly. **Problem**: too many submodules,
too much hopping between files, no mental anchor when reading.

**Phase 2 (consolidation — current)** — Collapse the `recommend/`
subpackage back into a single `recommend.py` with **section banners**
mapping to the algorithm's logical steps. Readability is preserved at
the function level; the file is one coherent concept top-to-bottom.

**Phase 3 (planned)** — Flatten `processing/` into single files at
root: `feature.py` (organized by aggregation level), `score.py`,
`pipeline.py`. Contracts move into `data.py` next to the I/O that
produces them.

**Phase 4 (planned)** — Collapse `loaders/` + `services/` + `models.py`
into `data.py`. Single source of truth for the tensor's typed views.

## Target end-state layout

```
src/ai_cdss/
├── __init__.py
├── constants.py            axis names + thresholds (untouched)
├── data.py                 typed tensor views + I/O (loaders/ + services/ merged)
├── feature.py              ONE file; sections by aggregation level:
│                             ## per-(patient, protocol, session, time)
│                             ## per-(patient, protocol)
│                             ## per-(patient)
├── score.py                score formula + Imputer (trivial module folded in)
├── recommend.py            ONE file; branches + MVT + substitute + topup + trace
├── pipeline.py             orchestrator — wires feature → impute → score
├── interface.py            CDSSInterface — production DB-aware entry
└── cli.py                  entrypoint
```

7-9 leaf files, zero nesting. Each file ≤ 600 lines, read as a
single coherent concept with section banners.

## What gets kept (good ideas from Phase 1)

- **`PatientState` class** — patient-scoped view of scoring. Drops the
  `(scoring, patient_id)` thread from every helper. Lives inside
  `recommend.py` as a section.
- **`SubstituteResult` dataclass** — clear, auditable substitute search
  outcome (no silent `None`).
- **Typed contracts** (`PreparedInputs`, `MergedFeatures`, …) — moved
  into `data.py` so they live next to the I/O that produces them, not
  in a separate `contracts.py`.
- **Small named functions** with one job each — readability preserved
  at the function level even though the file is now larger.
- **Behavior** — every existing test continues to pass.

## What got reverted

- `recommend/` subpackage (10 files) → consolidated into `recommend.py`.
- `processing/contracts.py` as a standalone module → its dataclasses
  move into `data.py` (Phase 3).
- `processing/` directory → flattened to root in Phase 3.

## Acceptance criteria

- The 21 existing unit tests in `tests/unit/` keep passing.
- Each top-level file (`recommend.py`, `feature.py`, etc.) reads
  top-to-bottom as one concept, with section banners flagging which
  tensor axis is being reduced over.
- The full package has ≤ 9 top-level files, no nested submodules beyond
  reasonable utilities.
- Opening any single file gives a coherent picture without needing to
  follow imports across the package.

## Tracking

- [x] Phase 1: split into `recommend/` subpackage (over-fragmented, reverted).
- [x] Phase 2: collapse `recommend/` → single `recommend.py` (10 sections).
- [x] Phase 3: flatten `processing/` → root `feature.py` / `score.py` / `pipeline.py`.
- [x] Phase 4: flatten `loaders/` → root `loader.py`; `services/` → root `service.py`.
       Keep `models.py` as-is (303 lines is fine, no consolidation gain).
- [x] Phase 5: `docs/architecture.md` describing tensor model + dataflow.

## End state — 12 files at `src/ai_cdss/`, zero deep nesting

```
__init__.py        29  public API re-exports
cdss.py            11  back-compat re-export of CDSS
clinical.py        80  ClinicalSubscales + ProtocolToClinicalMapper
constants.py      158  untouched
feature.py        614  feature reductions over tensor axes
loader.py         510  DB / CSV / synthetic I/O
models.py         303  pandera schemas, DataUnit, DataUnitSet
pipeline.py       465  typed contracts + DataPipeline
recommend.py      711  branches + MVT + topup + CDSS orchestrator
score.py           99  Imputer + Scorer
service.py        278  PPF / similarity / whitelist services
utils.py          107  MultiKeyDict + small helpers
                ─────
                 3365 total lines
```

`loaders/` and `services/` subdirs survive as one-line back-compat
re-export shims. `processing/` removed entirely.

35 unit tests pass throughout. Behavior identical to v0.3.1.
