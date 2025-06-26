"""
Service for computing protocol similarity between protocols using their mapped features.
"""

import logging
from typing import Any

import pandas as pd
from ai_cdss.constants import DEFAULT_OUTPUT_DIR, PROTOCOL_SIMILARITY_CSV
from ai_cdss.processing.clinical import ProtocolToClinicalMapper
from ai_cdss.processing.features import compute_protocol_similarity

logger = logging.getLogger(__name__)


class ProtocolSimilarityService:
    """
    Service for computing protocol similarity matrices.
    """

    def __init__(self, loader: Any):
        self.loader = loader

    def compute_protocol_similarity(self) -> pd.DataFrame:
        """
        Compute the protocol similarity matrix using mapped protocol attributes.
        Returns the protocol similarity DataFrame.
        """
        protocol = self.loader.load_protocol_attributes()
        if protocol is None or protocol.empty:
            raise ValueError("Protocol data could not be loaded.")
        protocol_map = ProtocolToClinicalMapper().map_protocol_features(protocol)
        if protocol_map is None or protocol_map.empty:
            raise ValueError("Mapped protocol features are empty.")
        similarity_df = compute_protocol_similarity(protocol_map)
        if similarity_df is None or similarity_df.empty:
            raise ValueError("Protocol similarity computation returned no data.")
        return similarity_df

    def persist_protocol_similarity(self, similarity_df: pd.DataFrame) -> str:
        """
        Persist the protocol similarity DataFrame to CSV in the output directory.
        Returns the file path.
        """
        try:
            output_path = DEFAULT_OUTPUT_DIR / PROTOCOL_SIMILARITY_CSV
            output_path.parent.mkdir(parents=True, exist_ok=True)
            similarity_df.to_csv(output_path, index=False)
            logger.info(
                "Protocol similarity persisted successfully to %s.", output_path
            )
            return str(output_path)
        except Exception as e:
            logger.error("Failed to save protocol similarity to CSV: %s", e)
            raise RuntimeError(f"Failed to save protocol similarity to CSV: {e}") from e

    def compute_and_persist_protocol_similarity(self) -> str:
        """
        Convenience method to compute and persist protocol similarity in one call.
        Returns the file path.
        """
        similarity_df = self.compute_protocol_similarity()
        return self.persist_protocol_similarity(similarity_df)
