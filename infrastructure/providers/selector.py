from __future__ import annotations

import json
from collections.abc import Callable
from urllib.error import URLError
from urllib.request import Request, urlopen

from shared.config import Settings

from .types import ProviderName


def _has_value(value: str) -> bool:
    return bool(value and value.strip())


def ollama_available(base_url: str, timeout_seconds: float = 1.0) -> bool:
    """Return True when Ollama is reachable and responds with JSON payload."""
    if not _has_value(base_url):
        return False

    endpoint = f"{base_url.rstrip('/')}/api/tags"
    request = Request(endpoint, method="GET")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read().decode("utf-8", errors="replace")
            data = json.loads(payload)
            return isinstance(data, dict)
    except (URLError, ValueError, OSError):
        return False


def resolve_chat_provider(
    settings: Settings,
    *,
    ollama_probe: Callable[[str], bool],
) -> ProviderName:
    if settings.chat_provider != "auto":
        return settings.chat_provider
    return _resolve_auto_provider(
        ollama_base_url=settings.ollama_base_url,
        openai_api_key=settings.openai_api_key,
        huggingface_api_key=settings.huggingface_api_key,
        ollama_probe=ollama_probe,
        provider_type_name="chat_provider",
    )


def resolve_embeddings_provider(
    settings: Settings,
    *,
    ollama_probe: Callable[[str], bool],
) -> ProviderName:
    if settings.embeddings_provider != "auto":
        return settings.embeddings_provider
    return _resolve_auto_provider(
        ollama_base_url=settings.ollama_base_url,
        openai_api_key=settings.openai_api_key,
        huggingface_api_key=settings.huggingface_api_key,
        ollama_probe=ollama_probe,
        provider_type_name="embeddings_provider",
    )


def _resolve_auto_provider(
    *,
    ollama_base_url: str,
    openai_api_key: str,
    huggingface_api_key: str,
    ollama_probe: Callable[[str], bool],
    provider_type_name: str,
) -> ProviderName:
    if ollama_probe(ollama_base_url):
        return "ollama"
    if _has_value(openai_api_key):
        return "openai"
    if _has_value(huggingface_api_key):
        return "huggingface"
    raise ValueError(
        f"{provider_type_name}=auto could not resolve a backend. "
        "Start Ollama or set OPENAI_API_KEY / HUGGINGFACE_API_KEY."
    )
