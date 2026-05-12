"""Marginal Value Theorem (MVT) — decide which prescribed protocols to swap.

Charnov 1976: a forager leaves a patch when its yield drops below the
environment's average. In CDSS terms: a prescribed protocol is a
candidate for swap when its SCORE is below the mean SCORE of the
currently-prescribed set.

The function is intentionally tiny so the criterion is one glance.

Behavior preserved exactly from the v0.3.1 implementation in
`CDSS._decide_prescription_swap`.

Note: this preserves the v0.3.1 "mean over prescribed only" semantics.
The alternative "mean over environment-wide whitelist" is being
explored on the `feat/env-wide-mvt-threshold` branch — see the original
repo at ../ai-cdss for that work-in-progress.
"""
from __future__ import annotations

import pandas as pd

from ai_cdss.constants import PROTOCOL_ID, SCORE


def below_mean_protocols(prescriptions: pd.DataFrame) -> list[int]:
    """Protocol IDs whose SCORE is strictly below the mean SCORE of the
    given prescriptions.

    Strict `<` — protocols at exactly the mean stay (no tie-breaking).

    Returns the list in DataFrame-row order. The caller is responsible
    for any further ordering (e.g. sort-by-score-ascending).
    """
    if prescriptions.empty:
        return []
    mean = prescriptions[SCORE].mean()
    below_mask = prescriptions[SCORE] < mean
    return prescriptions.loc[below_mask, PROTOCOL_ID].tolist()
