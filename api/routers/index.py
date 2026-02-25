from __future__ import annotations

import asyncio
import logging
from contextlib import suppress

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from api.dependencies import get_app_settings, get_document_ports, get_indexing_ports
from api.schemas import ErrorResponse, IndexResponse, IndexSourceInfo, IndexStatusResponse
from application.indexing.service import index_documents as _index_documents
from application.indexing.service import index_zip_bytes
from domain.indexing.ports.bundles import DocumentIndexingPorts, IndexingPorts
from shared.config import Settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["indexing"])


def _is_zip_file(file: UploadFile) -> bool:
    filename = (file.filename or "").lower()
    if filename.endswith(".zip"):
        return True
    return file.content_type in {"application/zip", "application/x-zip-compressed"}


@router.post(
    "/index/code",
    response_model=IndexResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Index a source code project from a zip archive",
)
async def index_code(
    file: UploadFile = File(...),
    ports: IndexingPorts = Depends(get_indexing_ports),
    settings: Settings = Depends(get_app_settings),
) -> IndexResponse:
    """Receive a .zip of source code, extract it safely, and index all text files."""
    if not _is_zip_file(file):
        raise HTTPException(status_code=400, detail="Uploaded file must be a .zip archive")

    try:
        zip_bytes = await file.read()
        count = await asyncio.to_thread(index_zip_bytes, zip_bytes, ports, settings)
    except Exception as exc:
        logger.exception("Code indexing failed filename=%s", file.filename)
        raise HTTPException(status_code=500, detail="Indexing failed") from exc
    finally:
        with suppress(Exception):
            await file.close()

    return IndexResponse(documents_indexed=count)


@router.post(
    "/index/documents",
    response_model=IndexResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Index PDF documents (single PDF or a zip of PDFs)",
)
async def index_documents(
    file: UploadFile = File(...),
    ports: DocumentIndexingPorts = Depends(get_document_ports),
    settings: Settings = Depends(get_app_settings),
) -> IndexResponse:
    """Receive a .pdf or a .zip of PDFs, extract text, and index it for RAG."""
    filename = (file.filename or "").lower()
    if not (filename.endswith(".pdf") or filename.endswith(".zip")):
        raise HTTPException(status_code=400, detail="File must be a .pdf or a .zip of PDFs")

    try:
        file_bytes = await file.read()
        count = await asyncio.to_thread(
            _index_documents, file_bytes, file.filename or "document", ports, settings
        )
    except Exception as exc:
        logger.exception("Document indexing failed filename=%s", file.filename)
        raise HTTPException(status_code=500, detail="Indexing failed") from exc
    finally:
        with suppress(Exception):
            await file.close()

    return IndexResponse(documents_indexed=count)


@router.get("/index/status", response_model=IndexStatusResponse)
async def index_status(
    session_id: str = "default",
    settings: Settings = Depends(get_app_settings),
) -> IndexStatusResponse:
    from infrastructure.indexing.storage import get_collection_info

    sources_raw = await asyncio.to_thread(get_collection_info, settings, session_id)
    sources = [IndexSourceInfo(name=s["name"], chunks=s["chunks"]) for s in sources_raw]
    return IndexStatusResponse(sources=sources, total_chunks=sum(s.chunks for s in sources))


@router.post("/index/clear", response_model=dict)
async def index_clear(
    session_id: str = Form("default"),
    settings: Settings = Depends(get_app_settings),
) -> dict:
    from infrastructure.indexing.storage import clear_collection

    await asyncio.to_thread(clear_collection, settings, session_id)
    return {"status": "ok"}
