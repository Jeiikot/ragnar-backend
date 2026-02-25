from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient, Response
from starlette.types import ASGIApp

from domain.chat.entities import ChatResponse as CoreChatResponse
from shared.config import Settings


@pytest.fixture
def test_settings() -> Settings:
    """Settings with test-safe defaults — no real API key."""
    return Settings(
        chat_provider="openai",
        embeddings_provider="openai",
        openai_api_key="test-key-fake",
        chroma_persist_dir="/tmp/ragnar_test_chroma",
        chroma_collection_name="test_collection",
        chunk_size=500,
        chunk_overlap=50,
    )


@pytest.fixture
def mock_retriever() -> MagicMock:
    retriever = MagicMock()
    retriever.invoke.return_value = []
    retriever.ainvoke = AsyncMock(return_value=[])
    return retriever


@pytest.fixture
def mock_chat_engine() -> MagicMock:
    engine = MagicMock()
    engine.aask = AsyncMock(
        return_value=CoreChatResponse(answer="Test answer", sources=["test.py:1"])
    )
    return engine


class SyncASGIClient:
    """Synchronous wrapper around httpx.AsyncClient for unit tests."""

    def __init__(self, app: ASGIApp, base_url: str = "http://testserver") -> None:
        self._app = app
        self._base_url = base_url
        # Keep one event loop for the fixture lifetime. Creating/closing a loop
        # per request hangs on executor shutdown in this runtime when uploads are used.
        self._loop = asyncio.new_event_loop()

    async def _request(self, method: str, url: str, **kwargs: object) -> Response:
        transport = ASGITransport(app=self._app)
        async with AsyncClient(transport=transport, base_url=self._base_url) as client:
            return await client.request(method, url, **kwargs)

    def request(self, method: str, url: str, **kwargs: object) -> Response:
        return self._loop.run_until_complete(self._request(method, url, **kwargs))

    def get(self, url: str, **kwargs: object) -> Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: object) -> Response:
        return self.request("POST", url, **kwargs)

    def close(self) -> None:
        self._loop.run_until_complete(self._loop.shutdown_asyncgens())
        executor = getattr(self._loop, "_default_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        self._loop.close()


@pytest.fixture
def client(mock_chat_engine: MagicMock, test_settings: Settings) -> SyncASGIClient:
    """HTTP client for FastAPI tests using an in-process ASGI transport."""
    from api.dependencies import get_app_settings, get_chat_engine_dep
    from api.main import create_app

    application = create_app()

    async def _chat_engine_override() -> MagicMock:
        return mock_chat_engine

    async def _settings_override() -> Settings:
        return test_settings

    application.dependency_overrides[get_chat_engine_dep] = _chat_engine_override
    application.dependency_overrides[get_app_settings] = _settings_override

    client_instance = SyncASGIClient(application)
    yield client_instance

    client_instance.close()
    application.dependency_overrides.clear()
