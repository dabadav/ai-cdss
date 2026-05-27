"""Clinical Decision Support System for Rehabilitation Gaming System.

Public API after the functionality refactor:

    from ai_cdss import RecommendationService

That's the only entry point. `CDSS` remains as a back-compat alias for
`RecommendationService` (pre-existing callers); new code should use
`RecommendationService`. Everything else is internal. DataFrame column
shapes are documented in `docs/schemas.md` rather than exported as
types — the recommender does not validate frames at runtime.
"""

from .interface import CDSS, RecommendationService

__all__ = ["RecommendationService", "CDSS"]
