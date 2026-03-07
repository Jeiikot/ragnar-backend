from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from domain.chat.entities import ChatResponse as CoreChatResponse
from infrastructure.chat.engine import ChatEngine
from shared.config import Settings


@pytest.fixture
def integration_client(test_settings: Settings) -> TestClient:
    """TestClient with a ChatEngine that has a mocked LLM but real chain wiring."""
    from api.dependencies import get_app_settings, get_chat_engine_dep
    from api.main import create_app

    mock_engine = MagicMock(spec=ChatEngine)
    mock_engine.aask = AsyncMock(
        return_value=CoreChatResponse(
            answer="The auth module handles JWT tokens.",
            sources=["auth.py:15", "auth.py:42"],
        )
    )

    app = create_app()
    app.dependency_overrides[get_chat_engine_dep] = lambda: mock_engine
    app.dependency_overrides[get_app_settings] = lambda: test_settings
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.mark.integration
class TestChatEndpointIntegration:
    def test_chat_returns_answer_with_sources(self, integration_client: TestClient) -> None:
        resp = integration_client.post(
            "/api/v1/chat",
            json={"message": "how does auth work?", "session_id": "int-1"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "auth" in data["data"]["answer"].lower() or "JWT" in data["data"]["answer"]
        assert len(data["data"]["sources"]) >= 1

    def test_health_endpoint(self, integration_client: TestClient) -> None:
        resp = integration_client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
