from __future__ import annotations

import logging

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from shared.config import Settings

from . import huggingface as huggingface_provider
from . import ollama as ollama_provider
from . import openai as openai_provider
from .contracts import ProviderBuilders
from .selector import ollama_available as _selector_ollama_available
from .selector import resolve_chat_provider as _resolve_chat_provider_internal
from .selector import resolve_embeddings_provider as _resolve_embeddings_provider_internal
from .types import ProviderName

logger = logging.getLogger(__name__)

_PROVIDER_BUILDERS: dict[ProviderName, ProviderBuilders] = {
    "openai": ProviderBuilders(
        build_chat_model=openai_provider.build_chat_model,
        build_embeddings=openai_provider.build_embeddings,
    ),
    "ollama": ProviderBuilders(
        build_chat_model=ollama_provider.build_chat_model,
        build_embeddings=ollama_provider.build_embeddings,
    ),
    "huggingface": ProviderBuilders(
        build_chat_model=huggingface_provider.build_chat_model,
        build_embeddings=huggingface_provider.build_embeddings,
    ),
}


def _ollama_available(base_url: str, timeout_seconds: float = 1.0) -> bool:
    """Backward-compatible Ollama availability hook (kept for tests)."""
    return _selector_ollama_available(base_url, timeout_seconds)


def resolve_chat_provider(settings: Settings) -> ProviderName:
    return _resolve_chat_provider_internal(
        settings,
        ollama_probe=lambda base_url: _ollama_available(base_url),
    )


def resolve_embeddings_provider(settings: Settings) -> ProviderName:
    return _resolve_embeddings_provider_internal(
        settings,
        ollama_probe=lambda base_url: _ollama_available(base_url),
    )


def build_chat_model(settings: Settings) -> BaseChatModel:
    provider = resolve_chat_provider(settings)
    logger.info("Resolved chat provider=%s", provider)
    return _PROVIDER_BUILDERS[provider].build_chat_model(settings)


def build_embeddings(settings: Settings) -> Embeddings:
    provider = resolve_embeddings_provider(settings)
    logger.info("Resolved embeddings provider=%s", provider)
    return _PROVIDER_BUILDERS[provider].build_embeddings(settings)


__all__ = [
    "ProviderName",
    "build_chat_model",
    "build_embeddings",
    "resolve_chat_provider",
    "resolve_embeddings_provider",
    "_ollama_available",
]
