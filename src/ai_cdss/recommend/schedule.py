"""Day scheduling — distribute a list of protocols across the week.

Used by the bootstrap branch to lay out the initial schedule. Pure
function, no patient state, no engine context — just list arithmetic.

Behavior preserved exactly from the v0.3.1 implementation in
`CDSS._schedule_protocols`.
"""
from __future__ import annotations

import math


def round_robin_across_days(
    protocols: list[int],
    *,
    n_days: int,
    protocols_per_day: int,
) -> dict[int, list[int]]:
    """Distribute `protocols` across `n_days` round-robin, capped at
    `protocols_per_day` distinct protocols per day.

    Each protocol may appear on multiple days. Protocols are
    deduplicated WITHIN a day (no day gets the same protocol twice).

    Returns a `{day_index: [protocol_id, ...]}` mapping. Days are
    zero-indexed (0 = Monday in the AISN trial convention).
    """
    schedule: dict[int, list[int]] = {d: [] for d in range(n_days)}
    if not protocols:
        return schedule

    total_slots = n_days * protocols_per_day
    repeats = math.ceil(total_slots / len(protocols))
    sequence = (protocols * repeats)[:total_slots]

    for i, protocol in enumerate(sequence):
        day = i % n_days
        if protocol not in schedule[day]:
            schedule[day].append(protocol)
    return schedule
