"""Clinical Decision Support System for Rehabilitation Gaming System.

Public API after the functionality refactor:

    from ai_cdss import CDSSInterface

That's the only entry point. Everything else is internal.

The pandera schemas are also re-exported because they're public types
that callers occasionally need to reference (e.g., for typed DataFrame
constructors in tests).
"""

from .interface import CDSSInterface
from .models import PPFSchema, ScoringSchema, SessionSchema

__all__ = [
    "CDSSInterface",
    "PPFSchema",
    "ScoringSchema",
    "SessionSchema",
]
