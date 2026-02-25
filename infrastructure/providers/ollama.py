from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from shared.config import Settings


class OllamaHTTPEmbeddings(Embeddings):
    """Embeddings client that calls Ollama without generation-only options."""

    def __init__(self, base_url: str, model: str, timeout_seconds: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def _post_json(self, endpoint: str, payload: dict) -> dict:
        request = Request(
            f"{self.base_url}{endpoint}",
            method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8", errors="replace"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ValueError(
                f"Error raised by Ollama API HTTP code: {exc.code}, {body}"
            ) from exc
        except (URLError, ValueError, OSError) as exc:
            raise ValueError(f"Error raised by Ollama API: {exc}") from exc

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        payload = {"model": self.model, "input": texts}
        response = self._post_json("/api/embed", payload)
        embeddings = response.get("embeddings")
        if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
            return embeddings

        # Fallback for older Ollama APIs that expose /api/embeddings only.
        result: list[list[float]] = []
        for text in texts:
            data = self._post_json("/api/embeddings", {"model": self.model, "prompt": text})
            embedding = data.get("embedding")
            if not isinstance(embedding, list):
                raise ValueError("Ollama embeddings response missing 'embedding' list")
            result.append(embedding)
        return result

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def build_chat_model(settings: Settings) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.ollama_chat_model,
        base_url=settings.ollama_base_url,
        temperature=settings.chat_temperature,
    )


def build_embeddings(settings: Settings) -> Embeddings:
    return OllamaHTTPEmbeddings(
        model=settings.ollama_embedding_model,
        base_url=settings.ollama_base_url,
        timeout_seconds=300.0,
    )
