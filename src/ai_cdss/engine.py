"""Substrate-agnostic engine input — `EngineState` + `SimilarityMatrix`.

The recommendation engine in `recommend.py` used to require a
`pandas.DataFrame` for its `scoring` input and another `pandas.DataFrame`
for `protocol_similarity`. This module breaks that requirement:

  * `EngineState` (Protocol) declares **what the engine needs** from
    its patient-scoped state. Any object satisfying the protocol works.
  * `SimilarityMatrix` (Protocol) does the same for protocol pairwise
    similarity.
  * `ProtocolRow` is the row-shape the engine reads from `EngineState`.
    Plain dataclass — no pandas dependency.
  * `DataFrameBackedState` / `DataFrameSimilarity` adapt the existing
    pandas-based pipeline output to the protocols. Used by
    `CDSSInterface` and production code.
  * `DictBackedState` / `DictSimilarity` are pandas-free alternatives.
    Useful for synthetic backtests, unit tests, ad-hoc replays.

The engine internals (`_bootstrap_branch`, `_update_branch`,
`_fill_grid_coverage`, etc. in `recommend.py`) now type-hint
`EngineState` instead of `PatientState` — they work with either
substrate.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Mapping, Protocol, runtime_checkable

import pandas as pd

from ai_cdss.constants import (
    DAYS,
    DELTA_DM,
    PATIENT_ID,
    PPF,
    PROTOCOL_A,
    PROTOCOL_B,
    PROTOCOL_ID,
    RECENT_ADHERENCE,
    SCORE,
    SIMILARITY,
    USAGE,
    USAGE_WEEK,
    WEEKS_SINCE_START,
)


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  ProtocolRow — typed row in the (patient × protocol) plane          ║
# ║                                                                      ║
# ║  Single cell of the scoring tensor. All engine reads from           ║
# ║  EngineState produce these. The engine builds the output schedule    ║
# ║  by accumulating ProtocolRows, then converts to pd.DataFrame at the  ║
# ║  boundary for the final return type.                                 ║
# ╚═════════════════════════════════════════════════════════════════════╝

@dataclass
class ProtocolRow:
    """Per-(patient, protocol) cell. The minimum the engine needs.

    Optional fields default to None / 0 so synthetic callers can omit
    them. The engine treats missing values as "not relevant" rather
    than NaN.
    """
    patient_id:        int
    protocol_id:       int
    score:             float
    days:              list[int]   = field(default_factory=list)
    usage:             int         = 0
    usage_week:        int         = 0
    ppf:               float | None = None
    delta_dm:          float | None = None
    recent_adherence:  float | None = None
    weeks_since_start: int         = 0
    contrib:           list[float] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Render in the column-keyed shape `pandas` expects.

        Keys match the constants used throughout the rest of the
        package (PATIENT_ID, PROTOCOL_ID, SCORE, DAYS, …) — so the
        result drops into a `pd.DataFrame` directly.
        """
        return {
            PATIENT_ID:        self.patient_id,
            PROTOCOL_ID:       self.protocol_id,
            SCORE:             self.score,
            DAYS:              list(self.days),
            USAGE:             self.usage,
            USAGE_WEEK:        self.usage_week,
            PPF:               self.ppf,
            DELTA_DM:          self.delta_dm,
            RECENT_ADHERENCE:  self.recent_adherence,
            WEEKS_SINCE_START: self.weeks_since_start,
            "CONTRIB":         self.contrib,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "ProtocolRow":
        """Inverse of `as_dict` — build a ProtocolRow from a dict-shaped
        row (typically a pandas `to_dict("records")` element)."""
        days = row.get(DAYS) or []
        if not isinstance(days, list):
            days = []
        return cls(
            patient_id=int(row[PATIENT_ID]),
            protocol_id=int(row[PROTOCOL_ID]),
            score=float(row.get(SCORE, 0.0)) if pd.notna(row.get(SCORE)) else 0.0,
            days=list(days),
            usage=int(row.get(USAGE, 0)) if pd.notna(row.get(USAGE)) else 0,
            usage_week=int(row.get(USAGE_WEEK, 0)) if pd.notna(row.get(USAGE_WEEK)) else 0,
            ppf=_safe_float(row.get(PPF)),
            delta_dm=_safe_float(row.get(DELTA_DM)),
            recent_adherence=_safe_float(row.get(RECENT_ADHERENCE)),
            weeks_since_start=int(row.get(WEEKS_SINCE_START, 0))
                if pd.notna(row.get(WEEKS_SINCE_START)) else 0,
            contrib=row.get("CONTRIB"),
        )


def _safe_float(v: Any) -> float | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  EngineState — Protocol the engine consumes                          ║
# ║                                                                      ║
# ║  Every operation the recommendation engine needs to perform on the   ║
# ║  patient's state is declared here. PEP 544 structural subtype —      ║
# ║  any object with these attributes/methods satisfies the protocol.    ║
# ╚═════════════════════════════════════════════════════════════════════╝

@runtime_checkable
class EngineState(Protocol):
    """What the recommendation engine reads from its input state.

    Implementations: `DataFrameBackedState`, `DictBackedState`.
    Adding a new substrate (polars, xarray) = new implementation of
    this protocol; engine code is unchanged.
    """
    patient_id: int

    @property
    def has_data(self) -> bool: ...

    @property
    def all_protocols(self) -> list[int]: ...
    """All protocol IDs in this patient's cohort (PPF cohort)."""

    @property
    def prescribed_rows(self) -> list[ProtocolRow]: ...
    """Rows where DAYS is non-empty — the engine's view of the prior
    week. May be empty (bootstrap case)."""

    def is_week_skipped(self) -> bool: ...
    """True iff every prescribed row recorded USAGE_WEEK == 0."""

    def top_protocols(self, n: int) -> list[int]: ...
    """Top-N protocol IDs by SCORE."""

    @property
    def lowest_scoring_prescribed(self) -> int: ...
    """Protocol with lowest SCORE among prescribed rows. Used for the
    AISN min-1-swap forced swap."""

    def usage_of(self, protocol_id: int) -> int: ...
    """Lifetime per-protocol usage count for this patient."""

    @property
    def protocols_with_zero_usage(self) -> list[int]: ...
    """Protocol IDs with `usage == 0` — substitute-search tier 1
    candidates."""

    def score_row(self, protocol_id: int) -> ProtocolRow: ...
    """The full row for `(this patient, protocol_id)`."""

    @property
    def scoring_attrs(self) -> dict[str, Any]: ...
    """Pass-through metadata (e.g. SUBSCALES list)."""


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  SimilarityMatrix — Protocol for protocol-pairwise similarity        ║
# ╚═════════════════════════════════════════════════════════════════════╝

@runtime_checkable
class SimilarityMatrix(Protocol):
    """Pairwise protocol similarity queries.

    Implementations: `DataFrameSimilarity`, `DictSimilarity`.
    """

    def similarities_for(
        self, protocol_id: int, *, exclude: Iterable[int] = (),
    ) -> list[tuple[int, float]]: ...
    """All `(protocol_b, similarity)` pairs for `protocol_id`, minus
    self and excluded. Order doesn't matter — caller sorts."""

    def top_n_similar(
        self, protocol_id: int, n: int, *, exclude: Iterable[int] = (),
    ) -> list[int]: ...
    """Top-N protocol IDs most similar to `protocol_id`."""


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  DataFrameBackedState — adapter over the existing scoring frame     ║
# ║                                                                      ║
# ║  Used by CDSSInterface. Wraps the `pd.DataFrame` the pipeline       ║
# ║  produces. Same behavior as the v0.3.1 `PatientState`.               ║
# ╚═════════════════════════════════════════════════════════════════════╝

class DataFrameBackedState:
    """`EngineState` backed by a pandas DataFrame.

    The scoring DataFrame has one row per (patient, protocol) for every
    patient in the cohort. This adapter slices to one patient and
    exposes the engine-shaped read methods.

    Read-only — does not mutate the underlying frame.
    """

    def __init__(self, scoring: pd.DataFrame, patient_id: int) -> None:
        self._scoring = scoring
        self.patient_id = patient_id
        self.rows = scoring.loc[scoring[PATIENT_ID] == patient_id]

    # ------------------------------------------------------------------
    # EngineState protocol methods.

    @property
    def has_data(self) -> bool:
        return not self.rows.empty

    @property
    def all_protocols(self) -> list[int]:
        return self.rows[PROTOCOL_ID].astype(int).tolist()

    @property
    def prescribed_rows(self) -> list[ProtocolRow]:
        has_days = self.rows[DAYS].apply(
            lambda d: isinstance(d, list) and len(d) > 0
        )
        return [ProtocolRow.from_dict(r) for r in self.rows.loc[has_days].to_dict("records")]

    def is_week_skipped(self) -> bool:
        has_days = self.rows[DAYS].apply(
            lambda d: isinstance(d, list) and len(d) > 0
        )
        scheduled = self.rows.loc[has_days]
        if scheduled.empty:
            return False
        return bool((scheduled[USAGE_WEEK] == 0).all())

    def top_protocols(self, n: int) -> list[int]:
        return self.rows.nlargest(n, SCORE)[PROTOCOL_ID].astype(int).tolist()

    @property
    def lowest_scoring_prescribed(self) -> int:
        has_days = self.rows[DAYS].apply(
            lambda d: isinstance(d, list) and len(d) > 0
        )
        pres = self.rows.loc[has_days]
        return int(pres.loc[pres[SCORE].idxmin(), PROTOCOL_ID])

    def usage_of(self, protocol_id: int) -> int:
        match = self.rows.loc[self.rows[PROTOCOL_ID] == protocol_id, USAGE]
        return int(match.iloc[0]) if not match.empty else 0

    @property
    def protocols_with_zero_usage(self) -> list[int]:
        zero = self.rows.loc[self.rows[USAGE] == 0, PROTOCOL_ID]
        return zero.astype(int).tolist()

    def score_row(self, protocol_id: int) -> ProtocolRow:
        match = self.rows.loc[self.rows[PROTOCOL_ID] == protocol_id]
        if match.empty:
            raise KeyError(
                f"No scoring row for patient={self.patient_id}, protocol={protocol_id}"
            )
        return ProtocolRow.from_dict(match.iloc[0].to_dict())

    @property
    def scoring_attrs(self) -> dict[str, Any]:
        return dict(self._scoring.attrs)

    # ------------------------------------------------------------------
    # Back-compat aliases (legacy code that imported PatientState may
    # still call these).

    @property
    def prescriptions(self) -> pd.DataFrame:
        """Legacy: same filter as `prescribed_rows` but returns the
        underlying DataFrame slice. Some engine internals + the
        cdss-supervisor read this directly."""
        has_days = self.rows[DAYS].apply(
            lambda d: isinstance(d, list) and len(d) > 0
        )
        return self.rows.loc[has_days]

    @property
    def usage(self) -> pd.Series:
        """Legacy: per-protocol usage Series, indexed by PROTOCOL_ID."""
        return self.rows.set_index(PROTOCOL_ID)[USAGE]


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  DictBackedState — pandas-free patient state                         ║
# ║                                                                      ║
# ║  Useful for synthetic data, ad-hoc backtests, tests that don't       ║
# ║  want to materialize a DataFrame. Construct with a dict of           ║
# ║  ProtocolRows (or dict-shaped row dicts via from_dict).              ║
# ╚═════════════════════════════════════════════════════════════════════╝

class DictBackedState:
    """`EngineState` backed by an in-memory dict of `ProtocolRow`.

    Construct directly:

        state = DictBackedState(
            patient_id=4378,
            rows={
                200: ProtocolRow(patient_id=4378, protocol_id=200,
                                 score=1.8, days=[0, 2, 4]),
                201: ProtocolRow(...),
            },
        )

    Or from a list:

        state = DictBackedState.from_rows(patient_id, [row1, row2, ...])
    """

    def __init__(
        self,
        patient_id: int,
        rows: Mapping[int, ProtocolRow],
        *,
        scoring_attrs: dict[str, Any] | None = None,
    ) -> None:
        self.patient_id = patient_id
        self._rows: dict[int, ProtocolRow] = dict(rows)
        self._scoring_attrs = scoring_attrs or {}

    @classmethod
    def from_rows(
        cls,
        patient_id: int,
        rows: Iterable[ProtocolRow],
        *,
        scoring_attrs: dict[str, Any] | None = None,
    ) -> "DictBackedState":
        """Build from an iterable of `ProtocolRow` objects."""
        return cls(
            patient_id=patient_id,
            rows={r.protocol_id: r for r in rows},
            scoring_attrs=scoring_attrs,
        )

    # ------------------------------------------------------------------
    # EngineState protocol methods.

    @property
    def has_data(self) -> bool:
        return bool(self._rows)

    @property
    def all_protocols(self) -> list[int]:
        return list(self._rows.keys())

    @property
    def prescribed_rows(self) -> list[ProtocolRow]:
        return [r for r in self._rows.values() if r.days]

    def is_week_skipped(self) -> bool:
        scheduled = self.prescribed_rows
        if not scheduled:
            return False
        return all(r.usage_week == 0 for r in scheduled)

    def top_protocols(self, n: int) -> list[int]:
        ranked = sorted(
            self._rows.values(),
            key=lambda r: -r.score,
        )
        return [r.protocol_id for r in ranked[:n]]

    @property
    def lowest_scoring_prescribed(self) -> int:
        scheduled = self.prescribed_rows
        if not scheduled:
            raise ValueError(
                f"Patient {self.patient_id} has no prescribed rows; "
                "cannot pick lowest-scoring prescribed."
            )
        return min(scheduled, key=lambda r: r.score).protocol_id

    def usage_of(self, protocol_id: int) -> int:
        row = self._rows.get(protocol_id)
        return row.usage if row else 0

    @property
    def protocols_with_zero_usage(self) -> list[int]:
        return [pid for pid, r in self._rows.items() if r.usage == 0]

    def score_row(self, protocol_id: int) -> ProtocolRow:
        if protocol_id not in self._rows:
            raise KeyError(
                f"No row for patient={self.patient_id}, protocol={protocol_id}"
            )
        return self._rows[protocol_id]

    @property
    def scoring_attrs(self) -> dict[str, Any]:
        return dict(self._scoring_attrs)

    # ------------------------------------------------------------------
    # Mutation helpers — synthetic state often gets tweaked between
    # recommend calls (e.g. injecting a chained-mode prior).

    def with_prescribed_set(self, days_by_protocol: Mapping[int, list[int]]) -> "DictBackedState":
        """Return a new state with DAYS overrides applied to each
        protocol. Synthetic chained-mode backtest writes one line."""
        new_rows: dict[int, ProtocolRow] = {}
        for pid, row in self._rows.items():
            new_rows[pid] = ProtocolRow(
                **{**asdict(row), "days": list(days_by_protocol.get(pid, []))}
            )
        return DictBackedState(
            patient_id=self.patient_id,
            rows=new_rows,
            scoring_attrs=self._scoring_attrs,
        )


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  DataFrameSimilarity — adapter over the existing similarity table   ║
# ╚═════════════════════════════════════════════════════════════════════╝

class DataFrameSimilarity:
    """`SimilarityMatrix` backed by the long-form similarity DataFrame
    (PROTOCOL_A, PROTOCOL_B, SIMILARITY)."""

    def __init__(self, similarity_table: pd.DataFrame) -> None:
        self._table = similarity_table

    def similarities_for(
        self, protocol_id: int, *, exclude: Iterable[int] = (),
    ) -> list[tuple[int, float]]:
        rows = self._table.loc[self._table[PROTOCOL_A] == protocol_id]
        rows = rows.loc[rows[PROTOCOL_A] != rows[PROTOCOL_B]]
        excl = list(exclude)
        if excl:
            rows = rows.loc[~rows[PROTOCOL_B].isin(excl)]
        return [(int(b), float(s)) for b, s in zip(rows[PROTOCOL_B], rows[SIMILARITY])]

    def top_n_similar(
        self, protocol_id: int, n: int, *, exclude: Iterable[int] = (),
    ) -> list[int]:
        pairs = self.similarities_for(protocol_id, exclude=exclude)
        pairs.sort(key=lambda p: -p[1])
        return [pid for pid, _ in pairs[:n]]


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  DictSimilarity — pandas-free similarity matrix                      ║
# ╚═════════════════════════════════════════════════════════════════════╝

class DictSimilarity:
    """`SimilarityMatrix` backed by a dict of `(a, b) → similarity`.

    Asymmetric: `sim[(a, b)]` and `sim[(b, a)]` may differ.

    Construct:
        DictSimilarity({(200, 201): 0.83, (200, 202): 0.71, ...})
    """

    def __init__(self, pairs: Mapping[tuple[int, int], float]) -> None:
        self._pairs: dict[tuple[int, int], float] = {
            (int(a), int(b)): float(s) for (a, b), s in pairs.items()
        }
        self._by_a: dict[int, list[tuple[int, float]]] = {}
        for (a, b), s in self._pairs.items():
            self._by_a.setdefault(a, []).append((b, s))

    def similarities_for(
        self, protocol_id: int, *, exclude: Iterable[int] = (),
    ) -> list[tuple[int, float]]:
        excl = set(exclude)
        excl.add(protocol_id)  # exclude self
        return [(b, s) for b, s in self._by_a.get(protocol_id, []) if b not in excl]

    def top_n_similar(
        self, protocol_id: int, n: int, *, exclude: Iterable[int] = (),
    ) -> list[int]:
        pairs = self.similarities_for(protocol_id, exclude=exclude)
        pairs.sort(key=lambda p: -p[1])
        return [pid for pid, _ in pairs[:n]]


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  Boundary helpers — coerce DataFrames/dicts to the engine protocols  ║
# ╚═════════════════════════════════════════════════════════════════════╝

def coerce_engine_state(
    state: Any, patient_id: int | None = None,
) -> EngineState:
    """Adapt an input to `EngineState`.

      * `EngineState` instance     → returned as-is.
      * `pd.DataFrame`             → wrapped in `DataFrameBackedState`
                                     (patient_id required).
    """
    if isinstance(state, (DataFrameBackedState, DictBackedState)):
        return state
    if isinstance(state, pd.DataFrame):
        if patient_id is None:
            raise ValueError(
                "Wrapping a DataFrame as EngineState requires patient_id."
            )
        return DataFrameBackedState(state, patient_id)
    raise TypeError(
        f"Cannot coerce {type(state).__name__} to EngineState. "
        "Pass a DataFrame, DataFrameBackedState, or DictBackedState."
    )


def coerce_similarity(sim: Any) -> SimilarityMatrix:
    """Adapt an input to `SimilarityMatrix`.

      * `SimilarityMatrix` instance → returned as-is.
      * `pd.DataFrame`              → wrapped in `DataFrameSimilarity`.
      * `dict[(a,b), float]`        → wrapped in `DictSimilarity`.
    """
    if isinstance(sim, (DataFrameSimilarity, DictSimilarity)):
        return sim
    if isinstance(sim, pd.DataFrame):
        return DataFrameSimilarity(sim)
    if isinstance(sim, dict):
        return DictSimilarity(sim)
    raise TypeError(
        f"Cannot coerce {type(sim).__name__} to SimilarityMatrix. "
        "Pass a DataFrame, dict, DataFrameSimilarity, or DictSimilarity."
    )
