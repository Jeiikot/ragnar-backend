from __future__ import annotations

from functools import partial

from domain.indexing.ports import DocumentIndexingPorts, IndexingPorts
from infrastructure.indexing.chunking import load_and_split as split_file_impl
from infrastructure.indexing.file_discovery import collect_all_files as collect_files_impl
from infrastructure.indexing.pdf_reader import read_and_chunk_pdf as read_pdf_impl
from infrastructure.indexing.storage import append_documents as append_documents_impl
from infrastructure.indexing.zip_utils import extract_zip_safely as extract_zip_impl
from shared.config import Settings


def build_indexing_ports(settings: Settings, session_id: str | None = None) -> IndexingPorts:
    """Wire concrete infrastructure implementations to the domain ports."""
    collection_name = session_id or settings.chroma_collection_name
    return IndexingPorts(
        collect_files=collect_files_impl,
        split_file=split_file_impl,
        write_documents=partial(
            append_documents_impl,
            settings,
            collection_name=collection_name,
        ),
        extract_zip=extract_zip_impl,
    )


def build_document_ports(
    settings: Settings, session_id: str | None = None
) -> DocumentIndexingPorts:
    """Wire PDF reader and writer implementations to the document indexing ports."""
    collection_name = session_id or settings.chroma_collection_name
    return DocumentIndexingPorts(
        read_pdf=read_pdf_impl,
        write_documents=partial(
            append_documents_impl,
            settings,
            collection_name=collection_name,
        ),
        extract_zip=extract_zip_impl,
    )
