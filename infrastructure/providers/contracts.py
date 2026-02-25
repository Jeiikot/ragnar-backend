from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from shared.config import Settings


class ChatModelBuilder(Protocol):
    """Callable contract to build a chat model from settings."""

    def __call__(self, settings: Settings) -> BaseChatModel: ...


class EmbeddingsBuilder(Protocol):
    """Callable contract to build embeddings from settings."""

    def __call__(self, settings: Settings) -> Embeddings: ...


@dataclass(frozen=True)
class ProviderBuilders:
    """Grouped builders for one provider."""

    build_chat_model: ChatModelBuilder
    build_embeddings: EmbeddingsBuilder
