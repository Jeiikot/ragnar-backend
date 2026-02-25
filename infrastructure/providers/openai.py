from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from shared.config import Settings


def _has_value(value: str) -> bool:
    return bool(value and value.strip())


def build_chat_model(settings: Settings) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    if not _has_value(settings.openai_api_key):
        raise ValueError("OPENAI_API_KEY is required for chat_provider=openai")
    return ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or None,
        temperature=settings.chat_temperature,
    )


def build_embeddings(settings: Settings) -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    if not _has_value(settings.openai_api_key):
        raise ValueError("OPENAI_API_KEY is required for embeddings_provider=openai")
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or None,
    )
