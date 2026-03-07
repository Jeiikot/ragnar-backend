from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routers import chat as chat_router
from api.routers import index as index_router
from api.schemas import ApiErrorResponse, ErrorCode, HealthResponse
from infrastructure.indexing.file_discovery import load_local_ignore_spec
from shared.config import get_settings

logger = logging.getLogger(__name__)


class JsonFormatter(logging.Formatter):
    """Minimal JSON formatter for structured logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


def _setup_logging(level: str, log_format: str) -> None:
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler()
    if log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    root_logger.addHandler(handler)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    # Fail fast: global indexing policy file is mandatory.
    load_local_ignore_spec()
    _setup_logging(settings.log_level, settings.log_format)
    logger.info("Ragnar starting on %s:%d", settings.host, settings.port)
    yield


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title="Ragnar Backend",
        description="Index source code and ask natural language questions about it.",
        version="0.1.0",
        lifespan=lifespan,
    )

    settings = get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict) and "error_code" in exc.detail:
            error_code = exc.detail["error_code"]
            message = exc.detail.get("detail", "Request failed")
        else:
            message = exc.detail if isinstance(exc.detail, str) else "Request failed"
            error_code = (
                ErrorCode.INVALID_FILE_TYPE
                if exc.status_code == 400
                else ErrorCode.VALIDATION_ERROR
                if exc.status_code == 422
                else ErrorCode.INTERNAL_ERROR
            )
        return JSONResponse(
            status_code=exc.status_code,
            content=ApiErrorResponse(
                detail=message,
                error_code=error_code,
            ).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.warning("Request validation failed: %s", exc.errors())
        field_details: dict[str, list[str]] = {}
        for err in exc.errors():
            loc = ".".join(str(part) for part in err.get("loc", []) if part != "body")
            field_details.setdefault(loc, []).append(err.get("msg", "invalid"))
        return JSONResponse(
            status_code=422,
            content=ApiErrorResponse(
                detail="Validation error",
                error_code=ErrorCode.VALIDATION_ERROR,
                details=field_details or None,
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def _unhandled_exception(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception")
        return JSONResponse(
            status_code=500,
            content=ApiErrorResponse(
                detail="Internal server error",
                error_code=ErrorCode.INTERNAL_ERROR,
            ).model_dump(),
        )

    app.include_router(index_router.router)
    app.include_router(chat_router.router)

    @app.get("/api/v1/health", response_model=HealthResponse, tags=["health"])
    async def health() -> HealthResponse:
        return HealthResponse()

    return app


app = create_app()
