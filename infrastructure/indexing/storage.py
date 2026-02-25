from __future__ import annotations

import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document

from infrastructure.providers import build_embeddings
from shared.config import Settings

logger = logging.getLogger(__name__)


def build_vectorstore(settings: Settings, collection_name: str | None = None) -> Chroma:
    """Create or open the persisted Chroma vectorstore."""
    embeddings = build_embeddings(settings)
    return Chroma(
        collection_name=collection_name or settings.chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )


def append_documents(
    settings: Settings, documents: list[Document], collection_name: str | None = None
) -> None:
    """Add documents to the collection without replacing existing content."""
    vectorstore = build_vectorstore(settings, collection_name)
    vectorstore.add_documents(documents)


def clear_collection(settings: Settings, collection_name: str | None = None) -> None:
    """Delete the entire collection for a session."""
    vectorstore = build_vectorstore(settings, collection_name)
    try:
        vectorstore.delete_collection()
    except Exception:
        logger.debug("No collection to clear", exc_info=True)


def get_collection_info(
    settings: Settings, collection_name: str | None = None
) -> list[dict[str, int | str]]:
    """Return a list of indexed sources with their chunk counts."""
    vectorstore = build_vectorstore(settings, collection_name)
    try:
        data = vectorstore.get(include=["metadatas"])
        metadatas: list[dict] = data.get("metadatas") or []
        source_counts: dict[str, int] = {}
        for meta in metadatas:
            source = (meta or {}).get("source", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
        return [{"name": s, "chunks": c} for s, c in sorted(source_counts.items())]
    except Exception:
        logger.debug("Could not read collection info", exc_info=True)
        return []
