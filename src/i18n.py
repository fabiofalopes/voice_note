"""Language normalization for contract output."""

from langcodes import Language


def normalize_language(raw: str | None) -> str | None:
    """Normalize a provider language string to lowercase ISO 639-1.

    Args:
        raw: Provider language value such as ``English``, ``en``, or ``en-US``.

    Returns:
        The normalized language code, or ``None`` when no value was provided.
    """
    if raw is None:
        return None
    try:
        language = (
            Language.get(raw) if len(raw) <= 3 or "-" in raw else Language.find(raw)
        )
        return language.maximize().language.lower()
    except Exception:
        return raw.lower()
