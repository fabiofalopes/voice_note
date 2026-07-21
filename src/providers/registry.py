"""Central provider registry with two registration patterns.

Pattern 1 — Custom adapter class (Groq, MLX):
    registry.register("groq", GroqWhisperClient)

Pattern 2 — OpenAI-compatible config-only (zero provider code):
    registry.register_openai_compat(
        name="my-provider",
        base_url="https://api.example.com/v1",
        model="whisper-large-v3",
        api_key_env="MY_API_KEY",
        capabilities={...},
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from api.base_client import BaseSTTClient


class ProviderNotFoundError(LookupError):
    """Raised when a provider name is not registered."""


@dataclass(frozen=True, slots=True)
class ProviderInfo:
    """Metadata exposed without instantiating the provider."""

    name: str
    capabilities: dict[str, bool]
    models: list[str]
    pattern: str  # "custom" | "openai_compat"


@dataclass(frozen=True, slots=True)
class _OpenAICompatEntry:
    """Config-only provider definition for Pattern 2."""

    base_url: str
    model: str
    api_key_env: str
    capabilities: dict[str, bool] = field(default_factory=dict)


class ProviderRegistry:
    """Self-registering provider map with capability introspection."""

    def __init__(self) -> None:
        self._custom: dict[str, type[BaseSTTClient]] = {}
        self._openai_compat: dict[str, _OpenAICompatEntry] = {}

    def register(self, name: str, client_class: type[BaseSTTClient]) -> None:
        """Register a custom-adapter provider (Pattern 1)."""
        if name in self._custom or name in self._openai_compat:
            raise ValueError(f"Provider {name!r} already registered")
        self._custom[name] = client_class

    def register_openai_compat(
        self,
        name: str,
        *,
        base_url: str,
        model: str,
        api_key_env: str,
        capabilities: dict[str, bool],
    ) -> None:
        """Register an OpenAI-compatible endpoint with zero provider code (Pattern 2)."""
        if name in self._custom or name in self._openai_compat:
            raise ValueError(f"Provider {name!r} already registered")
        self._openai_compat[name] = _OpenAICompatEntry(
            base_url=base_url,
            model=model,
            api_key_env=api_key_env,
            capabilities=capabilities,
        )

    def get_class(self, name: str) -> type[BaseSTTClient]:
        """Resolve provider name to a BaseSTTClient subclass."""
        if name in self._custom:
            return self._custom[name]
        if name in self._openai_compat:
            return self._build_openai_compat_class(name)
        available = list(self._custom) + list(self._openai_compat)
        raise ProviderNotFoundError(
            f"Unknown provider {name!r}. Available: {available}"
        )

    def get_capabilities(self, name: str) -> dict[str, bool]:
        """Introspect capabilities without importing SDKs or hitting APIs."""
        if name in self._custom:
            return self._custom[name].CAPABILITIES
        if name in self._openai_compat:
            return self._openai_compat[name].capabilities
        raise ProviderNotFoundError(f"Unknown provider {name!r}")

    def list_providers(self) -> dict[str, ProviderInfo]:
        """Introspect all registered providers (for --list-providers)."""
        result: dict[str, ProviderInfo] = {}
        for name, cls in self._custom.items():
            result[name] = ProviderInfo(
                name=name,
                capabilities=cls.CAPABILITIES,
                models=list(cls.AVAILABLE_MODELS),
                pattern="custom",
            )
        for name, entry in self._openai_compat.items():
            result[name] = ProviderInfo(
                name=name,
                capabilities=entry.capabilities,
                models=[entry.model],
                pattern="openai_compat",
            )
        return result

    def _build_openai_compat_class(self, name: str) -> type[BaseSTTClient]:
        entry = self._openai_compat[name]
        from providers.openai_compat_client import make_openai_compat_client

        cls = make_openai_compat_client(
            provider_name=name,
            base_url=entry.base_url,
            default_model=entry.model,
            api_key_env=entry.api_key_env,
            capabilities=entry.capabilities,
        )
        self._custom[name] = cls
        del self._openai_compat[name]
        return cls


_registry = ProviderRegistry()


def get_registry() -> ProviderRegistry:
    """Return the process-wide provider registry singleton."""
    return _registry


def register_builtins() -> None:
    """Register the built-in providers. Called once at CLI startup."""
    from api.groq_client import GroqWhisperClient
    from api.modelos_client import ModelosSTTClient

    reg = get_registry()
    if "groq" not in reg._custom:
        reg.register("groq", GroqWhisperClient)
    if "modelos" not in reg._custom:
        reg.register("modelos", ModelosSTTClient)
