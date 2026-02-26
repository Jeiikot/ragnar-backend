from __future__ import annotations

import asyncio
import logging
from contextlib import suppress

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from api.dependencies import get_app_settings, get_document_ports, get_indexing_ports
from api.schemas import (
    ApiErrorResponse,
    ApiListResponse,
    ApiResponse,
    ClearResponse,
    ErrorCode,
    IndexResponse,
    IndexSourceInfo,
    IndexStatusMeta,
)
from application.indexing.service import index_documents as _index_documents
from application.indexing.service import index_zip_bytes
from domain.indexing.ports.bundles import DocumentIndexingPorts, IndexingPorts
from shared.config import Settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["indexing"])

_ERROR_RESPONSES = {
    400: {"model": ApiErrorResponse},
    500: {"model": ApiErrorResponse},
}


def _is_zip_file(file: UploadFile) -> bool:
    filename = (file.filename or "").lower()
    if filename.endswith(".zip"):
        return True
    return file.content_type in {"application/zip", "application/x-zip-compressed"}


@router.post(
    "/index/code",
    response_model=ApiResponse[IndexResponse],
    responses=_ERROR_RESPONSES,
    summary="Index a source code project from a zip archive",
)
async def index_code(
    file: UploadFile = File(...),
    ports: IndexingPorts = Depends(get_indexing_ports),
    settings: Settings = Depends(get_app_settings),
) -> ApiResponse[IndexResponse]:
    """Receive a .zip of source code, extract it safely, and index all text files."""
    if not _is_zip_file(file):
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": ErrorCode.INVALID_FILE_TYPE,
                "detail": "Uploaded file must be a .zip archive",
            },
        )

    try:
        zip_bytes = await file.read()
        count = await asyncio.to_thread(index_zip_bytes, zip_bytes, ports, settings)
    except Exception as exc:
        logger.exception("Code indexing failed filename=%s", file.filename)
        raise HTTPException(
            status_code=500,
            detail={"error_code": ErrorCode.INDEXING_FAILED, "detail": "Indexing failed"},
        ) from exc
    finally:
        with suppress(Exception):
            await file.close()

    return ApiResponse(data=IndexResponse(documents_indexed=count))


@router.post(
    "/index/documents",
    response_model=ApiResponse[IndexResponse],
    responses=_ERROR_RESPONSES,
    summary="Index PDF documents (single PDF or a zip of PDFs)",
)
async def index_documents(
    file: UploadFile = File(...),
    ports: DocumentIndexingPorts = Depends(get_document_ports),
    settings: Settings = Depends(get_app_settings),
) -> ApiResponse[IndexResponse]:
    """Receive a .pdf or a .zip of PDFs, extract text, and index it for RAG."""
    filename = (file.filename or "").lower()
    if not (filename.endswith(".pdf") or filename.endswith(".zip")):
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": ErrorCode.INVALID_FILE_TYPE,
                "detail": "File must be a .pdf or a .zip of PDFs",
            },
        )

    try:
        file_bytes = await file.read()
        count = await asyncio.to_thread(
            _index_documents, file_bytes, file.filename or "document", ports, settings
        )
    except Exception as exc:
        logger.exception("Document indexing failed filename=%s", file.filename)
        raise HTTPException(
            status_code=500,
            detail={"error_code": ErrorCode.INDEXING_FAILED, "detail": "Indexing failed"},
        ) from exc
    finally:
        with suppress(Exception):
            await file.close()

    return ApiResponse(data=IndexResponse(documents_indexed=count))


@router.get("/index/status", response_model=ApiListResponse[IndexSourceInfo])
async def index_status(
    session_id: str = "default",
    settings: Settings = Depends(get_app_settings),
) -> ApiListResponse[IndexSourceInfo]:
    from infrastructure.indexing.storage import get_collection_info

    sources_raw = await asyncio.to_thread(get_collection_info, settings, session_id)
    sources = [IndexSourceInfo(name=s["name"], chunks=s["chunks"]) for s in sources_raw]
    return ApiListResponse(
        data=sources,
        meta=IndexStatusMeta(
            total_items=len(sources),
            total_chunks=sum(s.chunks for s in sources),
        ),
    )


@router.post("/index/clear", response_model=ApiResponse[ClearResponse])
async def index_clear(
    session_id: str = Form("default"),
    settings: Settings = Depends(get_app_settings),
) -> ApiResponse[ClearResponse]:
    from infrastructure.indexing.storage import clear_collection

    await asyncio.to_thread(clear_collection, settings, session_id)
    return ApiResponse(data=ClearResponse())
