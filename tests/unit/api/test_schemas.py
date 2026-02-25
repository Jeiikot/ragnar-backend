from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.schemas import ChatRequest, ChatResponse, IndexResponse


class TestIndexResponse:
    def test_serialization(self) -> None:
        resp = IndexResponse(documents_indexed=42)
        data = resp.model_dump()
        assert data == {"status": "ok", "documents_indexed": 42}


class TestChatRequest:
    def test_valid_request(self) -> None:
        req = ChatRequest(message="what does main do?", session_id="abc")
        assert req.message == "what does main do?"
        assert req.session_id == "abc"

    def test_default_session_id(self) -> None:
        req = ChatRequest(message="hi")
        assert req.session_id == "default"

    def test_rejects_empty_message(self) -> None:
        with pytest.raises(ValidationError):
            ChatRequest(message="")

    def test_rejects_too_long_message(self) -> None:
        with pytest.raises(ValidationError):
            ChatRequest(message="x" * 4001)


class TestChatResponse:
    def test_serialization(self) -> None:
        resp = ChatResponse(answer="hello", sources=["a.py:1"])
        data = resp.model_dump()
        assert data == {"answer": "hello", "sources": ["a.py:1"]}

    def test_default_sources(self) -> None:
        resp = ChatResponse(answer="hi")
        assert resp.sources == []
