from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_chat_engine_dep
from api.schemas import ChatRequest, ChatResponse, ErrorResponse
from infrastructure.chat.engine import ChatEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={500: {"model": ErrorResponse}},
    summary="Ask a question about the indexed codebase",
)
async def chat(
    request: ChatRequest,
    engine: ChatEngine = Depends(get_chat_engine_dep),
) -> ChatResponse:
    """Send a question through the RAG pipeline and return an answer
    with source citations."""
    try:
        result = await engine.aask(
            question=request.message,
            session_id=request.session_id,
        )
    except Exception as exc:
        logger.exception("Chat failed for session=%s", request.session_id)
        raise HTTPException(status_code=500, detail="Chat failed") from exc
    return ChatResponse(answer=result.answer, sources=result.sources)
