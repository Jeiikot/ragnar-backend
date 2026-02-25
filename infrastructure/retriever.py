from __future__ import annotations

import logging
from typing import Any

from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from infrastructure.providers import build_embeddings
from shared.config import Settings

logger = logging.getLogger(__name__)


def get_retriever(settings: Settings, session_id: str | None = None) -> VectorStoreRetriever:
    """Build a read-only retriever over the persisted Chroma collection."""
    collection_name = session_id or settings.chroma_collection_name
    embeddings = build_embeddings(settings)
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )
    search_kwargs: dict[str, Any] = {"k": settings.retriever_k}
    if settings.retriever_search_type == "mmr":
        search_kwargs["fetch_k"] = max(settings.retriever_fetch_k, settings.retriever_k)

    retriever = vectorstore.as_retriever(
        search_type=settings.retriever_search_type,
        search_kwargs=search_kwargs,
    )
    logger.info(
        "Retriever ready collection=%s search_type=%s k=%d fetch_k=%s",
        collection_name,
        settings.retriever_search_type,
        settings.retriever_k,
        search_kwargs.get("fetch_k"),
    )
    return retriever
