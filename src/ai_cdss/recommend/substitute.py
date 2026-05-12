"""Substitute search — picks one protocol to replace a removed one.

The search is **two-tier**:

  Tier 1 (preferred):  pick the protocol the patient has NEVER used
                       (USAGE == 0) that is most similar to the removed
                       one. "Try something new but related."

  Tier 2 (fallback):   if every candidate has been used at least once,
                       pick the LEAST-used among the top-5 most-similar.
                       "Keep things related; ration the over-used."

If both tiers fail (the candidate pool is fully exhausted — typically
because every whitelist protocol is already excluded), the function
returns `None`. The caller decides what to do; the engine's v0.3.1
behavior is to keep the original protocol in place (see
`update.build_substitute_or_keep`).

Behavior preserved exactly from the v0.3.1 implementations in
`CDSS._get_substitute`, `_get_unused_candidates`,
`_get_least_used_candidates`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from ai_cdss.recommend import similarity as sim

logger = logging.getLogger(__name__)


@dataclass
class SubstituteResult:
    """The outcome of a substitute search.

    Carries the chosen protocol id (or None) plus enough context to
    populate a swap trace entry — which tier matched, which candidates
    were considered, why.
    """
    protocol_id: int | None
    tier:        str            # "unused" / "least_used_top_similar" / "exhausted"
    candidates:  list[int]      # the pool actually considered


def find_substitute(
    *,
    usage: pd.Series,
    similarities: pd.DataFrame,
    removed_protocol_id: int,
) -> SubstituteResult:
    """Run the two-tier search. See module docstring for the rules.

    Parameters
    ----------
    usage
        The patient's full usage Series (indexed by protocol_id).
    similarities
        Similarity rows for the removed protocol, already filtered to
        exclude self and excluded protocols. See `similarity.similarities_for`.
    removed_protocol_id
        The protocol being swapped out. Used only for logging.
    """
    unused = _protocols_never_used(usage)
    if unused:
        logger.info(
            "No usage for %s, selecting most similar from %s",
            removed_protocol_id, unused,
        )
        pick = sim.most_similar_within(unused, similarities)
        if pick is not None:
            return SubstituteResult(pick, "unused", unused)

    top5 = sim.top_n_similar(similarities, n=5)
    least_used = _least_used_among(usage, top5)
    if least_used:
        logger.info(
            "No unused protocols for %s, selecting least used from %s",
            removed_protocol_id, least_used,
        )
        pick = sim.most_similar_within(least_used, similarities)
        if pick is not None:
            return SubstituteResult(pick, "least_used_top_similar", least_used)

    return SubstituteResult(None, "exhausted", [])


# ---------------------------------------------------------------------------
# Internal helpers — small named functions so the tier logic above reads
# like prose.

def _protocols_never_used(usage: pd.Series) -> list[int]:
    """Protocol IDs with `usage == 0`."""
    return usage.loc[usage == 0].index.tolist()


def _least_used_among(usage: pd.Series, candidates: list[int]) -> list[int]:
    """Protocol IDs in `candidates` that share the minimum usage value.

    Returns empty list if `candidates` is empty or none of them appear
    in `usage`.
    """
    sub = usage.loc[usage.index.isin(candidates)]
    if sub.empty:
        return []
    floor = sub.min()
    return sub.loc[sub == floor].index.tolist()
