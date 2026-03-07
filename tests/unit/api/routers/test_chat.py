from __future__ import annotations

from unittest.mock import AsyncMock


class TestChatEndpoint:
    def test_chat_success(self, client, mock_chat_engine) -> None:  # type: ignore[no-untyped-def]
        resp = client.post(
            "/api/v1/chat",
            json={"message": "what does main do?", "session_id": "s1"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["data"]["answer"] == "Test answer"
        assert data["data"]["sources"] == ["test.py:1"]
        mock_chat_engine.aask.assert_called_once_with(
            question="what does main do?", session_id="s1"
        )

    def test_chat_default_session(self, client) -> None:  # type: ignore[no-untyped-def]
        resp = client.post("/api/v1/chat", json={"message": "hello"})
        assert resp.status_code == 200

    def test_chat_engine_failure(self, client, mock_chat_engine) -> None:  # type: ignore[no-untyped-def]
        mock_chat_engine.aask = AsyncMock(side_effect=RuntimeError("LLM down"))
        resp = client.post(
            "/api/v1/chat",
            json={"message": "test", "session_id": "fail"},
        )
        assert resp.status_code == 500
        body = resp.json()
        assert body["error_code"] == "CHAT_FAILED"

    def test_chat_empty_message_rejected(self, client) -> None:  # type: ignore[no-untyped-def]
        resp = client.post("/api/v1/chat", json={"message": ""})
        assert resp.status_code == 422
