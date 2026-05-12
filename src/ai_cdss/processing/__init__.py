"""Back-compat re-export shim.

The `processing/` subdirectory was flattened in the Phase 3 refactor.
Everything now lives at `src/ai_cdss/` root:

    processing.pipeline         → ai_cdss.pipeline
    processing.processor        → ai_cdss.pipeline (DataProcessor)
    processing.feature_builder  → ai_cdss.feature  (FeatureBuilder)
    processing.features         → ai_cdss.feature
    processing.imputer          → ai_cdss.score    (Imputer)
    processing.scorer           → ai_cdss.score    (Scorer)
    processing.utils.get_nth    → ai_cdss.pipeline.get_nth
    processing.clinical         → ai_cdss.clinical

This shim re-exports the public surface so callers using
`from ai_cdss.processing import DataProcessor` (etc.) keep working.

Phase 4 will collapse this shim entirely.
"""
from ai_cdss.clinical import ClinicalSubscales, ProtocolToClinicalMapper
from ai_cdss.pipeline import DataPipeline, DataProcessor

__all__ = [
    "ClinicalSubscales",
    "DataPipeline",
    "DataProcessor",
    "ProtocolToClinicalMapper",
]
