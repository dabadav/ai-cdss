"""Back-compat re-export shim. Real code lives in `ai_cdss.loader`."""
from ai_cdss.loader import (
    DataLoader,
    DataLoaderBase,
    DataLoaderLocal,
    DataLoaderMock,
)

__all__ = [
    "DataLoader",
    "DataLoaderBase",
    "DataLoaderLocal",
    "DataLoaderMock",
]
