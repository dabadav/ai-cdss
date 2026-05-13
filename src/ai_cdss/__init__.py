"""Clinical Decision Support System for Rehabilitation Gaming System.

Public API after the functionality refactor:

    from ai_cdss import CDSS

That's the only entry point. Everything else is internal. DataFrame
column shapes are documented in `docs/schemas.md` rather than
exported as types — the recommender does not validate frames at
runtime.
"""

from .interface import CDSS

__all__ = ["CDSS"]
