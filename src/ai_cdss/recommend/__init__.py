"""Public API of the `recommend` package.

The top-level `CDSS` class (in `ai_cdss/cdss.py`) is a thin orchestrator;
the actual logic lives here, split by concern:

    state          — patient-scoped view of the scoring DataFrame
    mvt            — Marginal Value Theorem swap criterion
    similarity     — protocol-similarity queries
    substitute     — two-tier substitute search
    schedule       — round-robin day distribution
    bootstrap      — first-week (no prior) recommendation builder
    update         — swap branch (prior week exists, not skipped)
    repeat_week    — skipped-week branch (repeat prior unchanged)
    topup          — post-step grid coverage fill
    trace          — trace-dict construction helpers

External callers should import the top-level `CDSS` class; the modules
here are public for testability but their primary audience is the
`CDSS` orchestrator itself.
"""
