# %%
import importlib.resources
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

from ai_cdss.constants import PROTOCOL_WHITELIST_YAML
from ai_cdss import config
from ai_cdss.processing.utils import load_yaml

logger = logging.getLogger(__name__)


class ProtocolWhitelistService:
    """
    Service for computing and persisting Patient-Protocol Fit (PPF) matrices.
    """
    def __init__(self, whitelist_yml_path: Optional[str] = None):
        """Initialize with an optional path to whitelist.yaml, defaulting to internal package resource."""
        # Retrieves max values for clinical subscales from config/scales.yaml
        if whitelist_yml_path:
            self.scales_path = Path(whitelist_yml_path)
        else:
            self.scales_path = importlib.resources.files(config) / Path(PROTOCOL_WHITELIST_YAML)
        if not self.scales_path.exists():
            raise FileNotFoundError(f"Scale YAML file not found at {self.scales_path}")

    def load_whitelist(self) -> Dict[str, Any]:
        """
        Load whitelist protocols from YAML file.

        Returns:
            Dict[str, Any]: Whitelist protocols.
        """
        # Load whitelist protocols
        whitelist = load_yaml(self.scales_path)['recommendations']['allowed_protocols']
        return whitelist

