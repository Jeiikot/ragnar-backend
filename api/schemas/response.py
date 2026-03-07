from __future__ import annotations

from enum import StrEnum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class ErrorCode(StrEnum):
    INVALID_FILE_TYPE = "INVALID_FILE_TYPE"
    INDEXING_FAILED = "INDEXING_FAILED"
    CHAT_FAILED = "CHAT_FAILED"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ApiResponse(BaseModel, Generic[T]):
    data: T


class IndexStatusMeta(BaseModel):
    total_items: int
    total_chunks: int


class ApiListResponse(BaseModel, Generic[T]):
    data: list[T]
    meta: IndexStatusMeta


class ApiErrorResponse(BaseModel):
    detail: str
    error_code: ErrorCode
    details: dict[str, Any] | None = None
