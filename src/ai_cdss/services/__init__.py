"""Back-compat re-export shim. Real code lives in `ai_cdss.service`."""
from ai_cdss.service import (
    PPFService,
    ProtocolSimilarityService,
    ProtocolWhitelistService,
    RecommendationDataService,
)

__all__ = [
    "PPFService",
    "ProtocolSimilarityService",
    "ProtocolWhitelistService",
    "RecommendationDataService",
]
