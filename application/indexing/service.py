from __future__ import annotations

import io
import logging
import tempfile
import zipfile
from pathlib import Path

from langchain_core.documents import Document

from domain.indexing.ports.bundles import DocumentIndexingPorts, IndexingPorts
from shared.config import Settings

logger = logging.getLogger(__name__)


def index_directory(path: str, ports: IndexingPorts, settings: Settings) -> int:
    """Index all allowed files under ``path`` and return indexed chunk count."""
    extracted_zip_root = Path(path).resolve()
    if not extracted_zip_root.is_dir():
        raise FileNotFoundError(f"Directory not found: {extracted_zip_root}")

    files = ports.collect_files(extracted_zip_root)
    logger.info("Found %d files to index under %s", len(files), extracted_zip_root)

    all_chunks: list[Document] = []
    for file_path, ext in files:
        chunks = ports.split_file(
            file_path,
            ext,
            extracted_zip_root,
            settings.chunk_size,
            settings.chunk_overlap,
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.warning("No document chunks produced from %s", extracted_zip_root)
        return 0

    ports.write_documents(all_chunks)
    logger.info("Indexed %d chunks into Chroma", len(all_chunks))
    return len(all_chunks)


def index_zip_bytes(zip_bytes: bytes, ports: IndexingPorts, settings: Settings) -> int:
    """Extract zip bytes into a temp directory and index the extracted files."""
    if not zip_bytes:
        raise ValueError("Zip file is empty")

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zip_archive:
            for info in zip_archive.infolist():
                member = Path(info.filename)
                if member.is_absolute() or ".." in member.parts:
                    raise ValueError("Zip file contains unsafe paths")
            with tempfile.TemporaryDirectory(prefix="ragnar_zip_") as tmp_dir:
                extracted_zip_root = Path(tmp_dir)
                ports.extract_zip(zip_archive, extracted_zip_root)
                return index_directory(str(extracted_zip_root), ports, settings)
    except zipfile.BadZipFile as exc:
        raise ValueError("Invalid zip file") from exc


def index_documents(
    file_bytes: bytes, filename: str, ports: DocumentIndexingPorts, settings: Settings
) -> int:
    """Index a PDF or a ZIP of PDFs."""
    if not file_bytes:
        raise ValueError("File is empty")

    name_lower = filename.lower()

    if name_lower.endswith(".pdf"):
        chunks = ports.read_pdf(file_bytes, filename, settings.chunk_size, settings.chunk_overlap)
    elif name_lower.endswith(".zip"):
        chunks = _index_pdf_zip(file_bytes, ports, settings)
    else:
        raise ValueError("File must be a .pdf or a .zip of PDFs")

    if not chunks:
        logger.warning("No content extracted from %s", filename)
        return 0

    ports.write_documents(chunks)
    logger.info("Indexed %d chunks from %s", len(chunks), filename)
    return len(chunks)


def _index_pdf_zip(
    file_bytes: bytes, ports: DocumentIndexingPorts, settings: Settings
) -> list[Document]:
    """Extract a ZIP archive and chunk every PDF found inside."""
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zip_archive:
            with tempfile.TemporaryDirectory(prefix="ragnar_docs_") as tmp_dir:
                extracted = Path(tmp_dir)
                ports.extract_zip(zip_archive, extracted)
                all_chunks: list[Document] = []
                for pdf_path in sorted(extracted.rglob("*.pdf")):
                    pdf_bytes = pdf_path.read_bytes()
                    rel_name = str(pdf_path.relative_to(extracted))
                    chunks = ports.read_pdf(
                        pdf_bytes, rel_name, settings.chunk_size, settings.chunk_overlap
                    )
                    all_chunks.extend(chunks)
                return all_chunks
    except zipfile.BadZipFile as exc:
        raise ValueError("Invalid zip file") from exc
