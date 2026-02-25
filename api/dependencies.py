from __future__ import annotations

from fastapi import Depends, Form

from domain.indexing.ports.bundles import DocumentIndexingPorts, IndexingPorts
from infrastructure.chat.engine import ChatEngine, build_chat_engine, reset_session_store
from infrastructure.indexing.adapters import build_document_ports, build_indexing_ports
from shared.config import Settings, get_settings, reset_settings

# ---------------------------------------------------------------------------
# Module-level singletons with explicit reset for testing
# ---------------------------------------------------------------------------
_chat_engine_instance: ChatEngine | None = None


def get_app_settings() -> Settings:
    """FastAPI dependency for Settings."""
    return get_settings()


def get_chat_engine_dep() -> ChatEngine:
    """FastAPI dependency for ChatEngine (lazy singleton)."""
    global _chat_engine_instance
    if _chat_engine_instance is None:
        _chat_engine_instance = build_chat_engine(settings=get_app_settings())
    return _chat_engine_instance


def get_indexing_ports(
    session_id: str = Form("default"),
    settings: Settings = Depends(get_app_settings),
) -> IndexingPorts:
    """FastAPI dependency for IndexingPorts (per-request, session-scoped)."""
    return build_indexing_ports(settings, session_id)


def get_document_ports(
    session_id: str = Form("default"),
    settings: Settings = Depends(get_app_settings),
) -> DocumentIndexingPorts:
    """FastAPI dependency for DocumentIndexingPorts (per-request, session-scoped)."""
    return build_document_ports(settings, session_id)


def reset_singletons() -> None:
    """Reset cached instances — call in test teardown."""
    global _chat_engine_instance
    _chat_engine_instance = None
    reset_session_store()
    reset_settings()
