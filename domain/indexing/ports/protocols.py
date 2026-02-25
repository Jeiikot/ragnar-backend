from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Protocol

from langchain_core.documents import Document


class FileCollectorProtocol(Protocol):
    """Collect indexable files from a root directory."""

    def __call__(self, root: Path) -> list[tuple[Path, str]]: ...


class FileChunkerProtocol(Protocol):
    """Split one file into chunk documents."""

    def __call__(
        self,
        file_path: Path,
        extension: str,
        root: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[Document]: ...


class VectorStoreWriterProtocol(Protocol):
    """Persist chunks into the vector store."""

    def __call__(self, documents: list[Document]) -> None: ...


class ZipExtractorProtocol(Protocol):
    """Extract zip archive into destination directory."""

    def __call__(self, zip_archive: zipfile.ZipFile, destination: Path) -> None: ...


class PdfChunkerProtocol(Protocol):
    """Read PDF bytes and split into chunk documents."""

    def __call__(
        self,
        pdf_bytes: bytes,
        filename: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[Document]: ...
