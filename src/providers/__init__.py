"""Provider registry and pluggable STT backends."""

from .registry import ProviderRegistry, ProviderNotFoundError, get_registry

__all__ = [
    "ProviderRegistry",
    "ProviderNotFoundError",
    "get_registry",
]
