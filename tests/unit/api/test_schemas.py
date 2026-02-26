from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.schemas import (
    ApiErrorResponse,
    ApiResponse,
    ChatRequest,
    ChatResponse,
    ClearResponse,
    ErrorCode,
    IndexResponse,
)


class TestIndexResponse:
    def test_serialization(self) -> None:
        resp = IndexResponse(documents_indexed=42)
        data = resp.model_dump()
        assert data == {"documents_indexed": 42}


class TestClearResponse:
    def test_default_cleared(self) -> None:
        resp = ClearResponse()
        assert resp.cleared is True

    def test_serialization(self) -> None:
        resp = ClearResponse()
        assert resp.model_dump() == {"cleared": True}


class TestApiResponse:
    def test_wraps_data(self) -> None:
        inner = IndexResponse(documents_indexed=5)
        resp = ApiResponse(data=inner)
        dumped = resp.model_dump()
        assert dumped == {"data": {"documents_indexed": 5}}


class TestApiErrorResponse:
    def test_error_code_serializes_as_string(self) -> None:
        resp = ApiErrorResponse(
            detail="Bad file type",
            error_code=ErrorCode.INVALID_FILE_TYPE,
        )
        dumped = resp.model_dump()
        assert dumped["error_code"] == "INVALID_FILE_TYPE"
        assert isinstance(dumped["error_code"], str)

    def test_details_defaults_to_none(self) -> None:
        resp = ApiErrorResponse(detail="err", error_code=ErrorCode.INTERNAL_ERROR)
        assert resp.details is None

    def test_details_can_be_set(self) -> None:
        resp = ApiErrorResponse(
            detail="Validation error",
            error_code=ErrorCode.VALIDATION_ERROR,
            details={"field": ["required"]},
        )
        assert resp.details == {"field": ["required"]}


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
