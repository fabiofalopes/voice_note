"""Provider registry tests — registration, lookup, capabilities, errors."""

import pytest

from providers.registry import (
    ProviderNotFoundError,
    ProviderRegistry,
)


@pytest.fixture
def registry():
    return ProviderRegistry()


class _FakeClient:
    PROVIDER_NAME = "fake"
    AVAILABLE_MODELS = ["fake-model-1", "fake-model-2"]
    CAPABILITIES = {
        "word_timestamps": True,
        "segment_timestamps": True,
        "language_detection": False,
        "quality_metrics": False,
        "speaker_diarization": False,
    }


class _OtherClient:
    PROVIDER_NAME = "other"
    AVAILABLE_MODELS = ["other-model"]
    CAPABILITIES = {
        "word_timestamps": False,
        "segment_timestamps": False,
        "language_detection": True,
        "quality_metrics": True,
        "speaker_diarization": False,
    }


def test_register_and_get_class(registry):
    registry.register("fake", _FakeClient)
    assert registry.get_class("fake") is _FakeClient


def test_register_duplicate_raises(registry):
    registry.register("fake", _FakeClient)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("fake", _OtherClient)


def test_get_class_unknown_raises(registry):
    with pytest.raises(ProviderNotFoundError, match="Unknown provider"):
        registry.get_class("nonexistent")


def test_get_capabilities(registry):
    registry.register("fake", _FakeClient)
    caps = registry.get_capabilities("fake")
    assert caps["word_timestamps"] is True
    assert caps["language_detection"] is False


def test_get_capabilities_unknown_raises(registry):
    with pytest.raises(ProviderNotFoundError):
        registry.get_capabilities("nonexistent")


def test_list_providers(registry):
    registry.register("fake", _FakeClient)
    registry.register("other", _OtherClient)
    providers = registry.list_providers()
    assert len(providers) == 2
    assert providers["fake"].pattern == "custom"
    assert providers["fake"].models == ["fake-model-1", "fake-model-2"]
    assert providers["other"].capabilities["language_detection"] is True


def test_register_openai_compat(registry):
    registry.register_openai_compat(
        name="compat",
        base_url="https://api.example.com/v1",
        model="whisper-1",
        api_key_env="FAKE_KEY",
        capabilities={"word_timestamps": False, "segment_timestamps": True},
    )
    caps = registry.get_capabilities("compat")
    assert caps["segment_timestamps"] is True

    providers = registry.list_providers()
    assert providers["compat"].pattern == "openai_compat"
    assert providers["compat"].models == ["whisper-1"]


def test_openai_compat_duplicate_raises(registry):
    registry.register_openai_compat(
        name="compat",
        base_url="https://api.example.com/v1",
        model="whisper-1",
        api_key_env="FAKE_KEY",
        capabilities={},
    )
    with pytest.raises(ValueError, match="already registered"):
        registry.register_openai_compat(
            name="compat",
            base_url="https://other.example.com/v1",
            model="whisper-2",
            api_key_env="OTHER_KEY",
            capabilities={},
        )


def test_cross_pattern_duplicate_raises(registry):
    registry.register("fake", _FakeClient)
    with pytest.raises(ValueError, match="already registered"):
        registry.register_openai_compat(
            name="fake",
            base_url="https://api.example.com/v1",
            model="whisper-1",
            api_key_env="FAKE_KEY",
            capabilities={},
        )


def test_builtin_registration():
    from providers.registry import get_registry, register_builtins

    register_builtins()
    reg = get_registry()
    assert "groq" in reg.list_providers()
    assert "modelos" in reg.list_providers()
