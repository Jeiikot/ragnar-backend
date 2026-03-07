from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base exception for Ragnar application errors."""

    error_code: str = "INTERNAL_ERROR"
    status_code: int = 500

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        return self.message


class InvalidFileTypeError(AppError):
    error_code = "INVALID_FILE_TYPE"
    status_code = 400


class IndexingError(AppError):
    error_code = "INDEXING_FAILED"
    status_code = 500


class ChatError(AppError):
    error_code = "CHAT_FAILED"
    status_code = 500
