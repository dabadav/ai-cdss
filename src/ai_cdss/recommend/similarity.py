"""Protocol-similarity queries used by the substitute search.

The similarity table is a DataFrame with rows of the form
`(PROTOCOL_A, PROTOCOL_B, SIMILARITY)`. Each protocol appears as both
A and B (the matrix is square-ish; not necessarily symmetric depending
on how the engine was trained).

These helpers slice and rank that table. All functions are pure — they
take inputs, return outputs, never mutate.

Behavior preserved exactly from the v0.3.1 implementations in
`CDSS._get_protocol_similarities`, `_get_top_similar_protocols`,
`_select_most_similar`.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from ai_cdss.constants import PROTOCOL_A, PROTOCOL_B, SIMILARITY


def similarities_for(
    protocol_id: int,
    similarity_table: pd.DataFrame,
    excluded: Iterable[int] | None = None,
) -> pd.DataFrame:
    """All rows in `similarity_table` describing how similar
    `protocol_id` is to other protocols, minus:
      * the self-similarity row (`protocol_id == protocol_id`),
      * any protocol in `excluded` (typically already-prescribed or
        already-chosen-as-substitute).
    """
    rows = similarity_table.loc[similarity_table[PROTOCOL_A] == protocol_id]
    rows = rows.loc[rows[PROTOCOL_A] != rows[PROTOCOL_B]]
    if excluded:
        rows = rows.loc[~rows[PROTOCOL_B].isin(list(excluded))]
    return rows


def top_n_similar(similarities: pd.DataFrame, n: int = 5) -> list[int]:
    """Top-N most-similar protocol IDs in the given similarities slice."""
    return similarities.nlargest(n, SIMILARITY)[PROTOCOL_B].tolist()


def most_similar_within(
    candidates: list[int], similarities: pd.DataFrame
) -> int | None:
    """The single protocol in `candidates` with the highest similarity
    value in `similarities`. Returns `None` if no candidate is in the
    similarities table (e.g., the entire candidate list was excluded).

    Ties broken by first-row order in `similarities` — same as v0.3.1.
    """
    matched = similarities.loc[similarities[PROTOCOL_B].isin(candidates)]
    if matched.empty:
        return None
    peak = matched[SIMILARITY].max()
    return int(matched.loc[matched[SIMILARITY] == peak, PROTOCOL_B].iloc[0])
