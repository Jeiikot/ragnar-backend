from __future__ import annotations

from unittest.mock import patch

import pytest

from infrastructure.providers import (
    build_chat_model,
    build_embeddings,
    resolve_chat_provider,
    resolve_embeddings_provider,
)
from shared.config import Settings


class TestProviderResolution:
    @patch("infrastructure.providers._ollama_available", return_value=True)
    def test_auto_prefers_ollama_when_available(self, _mock_ollama) -> None:
        settings = Settings(
            chat_provider="auto",
            embeddings_provider="auto",
            openai_api_key="test-key",
            huggingface_api_key="hf-key",
        )
        assert resolve_chat_provider(settings) == "ollama"
        assert resolve_embeddings_provider(settings) == "ollama"

    @patch("infrastructure.providers._ollama_available", return_value=False)
    def test_auto_falls_back_to_openai(self, _mock_ollama) -> None:
        settings = Settings(
            chat_provider="auto",
            embeddings_provider="auto",
            openai_api_key="test-key",
        )
        assert resolve_chat_provider(settings) == "openai"
        assert resolve_embeddings_provider(settings) == "openai"

    @patch("infrastructure.providers._ollama_available", return_value=False)
    def test_auto_falls_back_to_huggingface(self, _mock_ollama) -> None:
        settings = Settings(
            chat_provider="auto",
            embeddings_provider="auto",
            huggingface_api_key="hf-key",
        )
        assert resolve_chat_provider(settings) == "huggingface"
        assert resolve_embeddings_provider(settings) == "huggingface"

    @patch("infrastructure.providers._ollama_available", return_value=False)
    def test_auto_raises_when_no_provider_is_available(self, _mock_ollama) -> None:
        settings = Settings(chat_provider="auto", embeddings_provider="auto")
        with pytest.raises(ValueError):
            resolve_chat_provider(settings)
        with pytest.raises(ValueError):
            resolve_embeddings_provider(settings)


class TestProviderBuilders:
    def test_openai_chat_requires_api_key(self) -> None:
        settings = Settings(chat_provider="openai", openai_api_key="")
        with pytest.raises(ValueError):
            build_chat_model(settings)

    def test_huggingface_embeddings_requires_api_key(self) -> None:
        settings = Settings(
            embeddings_provider="huggingface",
            huggingface_api_key="",
        )
        with pytest.raises(ValueError):
            build_embeddings(settings)
