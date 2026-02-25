from __future__ import annotations

from dataclasses import dataclass

from .protocols import (
    FileChunkerProtocol,
    FileCollectorProtocol,
    PdfChunkerProtocol,
    VectorStoreWriterProtocol,
    ZipExtractorProtocol,
)


@dataclass(frozen=True)
class IndexingPorts:
    collect_files: FileCollectorProtocol
    split_file: FileChunkerProtocol
    write_documents: VectorStoreWriterProtocol
    extract_zip: ZipExtractorProtocol


@dataclass(frozen=True)
class DocumentIndexingPorts:
    read_pdf: PdfChunkerProtocol
    write_documents: VectorStoreWriterProtocol
    extract_zip: ZipExtractorProtocol
