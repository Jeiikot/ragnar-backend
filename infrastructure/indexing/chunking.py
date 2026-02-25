from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from infrastructure.indexing.constants import EXTENSION_LANGUAGE_MAP

logger = logging.getLogger(__name__)


def _resolve_language(extension: str) -> Language | None:
    return EXTENSION_LANGUAGE_MAP.get(extension.strip().lower())


def get_splitter(
    extension: str,
    chunk_size: int,
    chunk_overlap: int,
) -> RecursiveCharacterTextSplitter:
    """
    Return a language-aware splitter when possible, generic otherwise.
    """
    splitter_kwargs = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "add_start_index": True,
    }
    language = _resolve_language(extension)
    if language is None:
        return RecursiveCharacterTextSplitter(**splitter_kwargs)

    return RecursiveCharacterTextSplitter.from_language(
        language=language,
        **splitter_kwargs,
    )


def _add_metadata(
    chunks: list[Document],
    content: str,
    source: str,
    language_name: str,
    extension: str,
) -> list[Document]:
    """Attach source and position metadata to a single chunk."""
    for index, chunk in enumerate(chunks):
        start_index = chunk.metadata.get("start_index")
        if isinstance(start_index, int) and start_index >= 0:
            start_line = content.count("\n", 0, start_index) + 1
        else:
            preview = chunk.page_content[:80]
            char_offset = content.find(preview) if preview else -1
            start_line = content[: max(char_offset, 0)].count("\n") + 1

        chunk.metadata = {
            "source": source,
            "language": language_name,
            "file_extension": extension,
            "chunk_index": index,
            "start_line": start_line,
        }
    return chunks


def load_and_split(
    file_path: Path,
    extension: str,
    root: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """
    Read a single file, split it, and attach metadata to each chunk.
    """
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Skipping %s: %s", file_path, exc)
        return []

    if not content.strip():
        return []

    language = _resolve_language(extension)
    splitter = get_splitter(extension, chunk_size, chunk_overlap)
    chunks = splitter.create_documents([content])
    source = str(file_path.relative_to(root))
    language_name = language.value if language is not None else "text"

    chunks = _add_metadata(
        chunks=chunks,
        content=content,
        source=source,
        language_name=language_name,
        extension=extension,
    )
    return chunks
