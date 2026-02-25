from __future__ import annotations

import io
import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

logger = logging.getLogger(__name__)


def read_and_chunk_pdf(
    pdf_bytes: bytes,
    filename: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Read PDF bytes, extract text page by page, and split into chunks."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as exc:
        logger.warning("Could not parse PDF %s: %s", filename, exc)
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    all_chunks: list[Document] = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        chunks = splitter.create_documents([text])
        for i, chunk in enumerate(chunks):
            chunk.metadata = {
                "source": filename,
                "page": page_num,
                "chunk_index": i,
            }
        all_chunks.extend(chunks)

    return all_chunks
