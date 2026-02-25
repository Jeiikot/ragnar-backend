from __future__ import annotations

import logging

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from domain.chat.entities import ChatResponse
from infrastructure.providers import build_chat_model
from infrastructure.retriever import get_retriever
from shared.config import Settings

logger = logging.getLogger(__name__)

NO_CONTEXT_SENTINEL = "[NO_RELEVANT_CONTEXT]"

RAG_SYSTEM_PROMPT = """\
You are Ragnar, an expert assistant. Answer questions about the \
indexed content using ONLY the provided context. If the context does not contain \
enough information, say so honestly.

By default, give a concise answer (2-4 sentences or a short code snippet). \
Only elaborate with full details if the user explicitly asks for more, \
such as "explain more", "give details", or "elaborate".

If the provided context is partial or missing key details needed for an exact answer, \
or if context is exactly [NO_RELEVANT_CONTEXT], follow this format in the user's language:
1) Start with one explicit limitation sentence.
2) Add 3-5 suggested questions as a numbered list.
3) Suggestions must stay close to the user's intent and be concrete.
4) Only suggest questions you can answer from the provided context.
5) If context is exactly [NO_RELEVANT_CONTEXT], suggest scope-narrowing questions \
that help retrieve relevant context next (module, file, class, endpoint, test name).
6) Do not invent missing facts, counts, or coverage.

When the context supports only part of the answer, provide the known part first, \
then the suggested questions.

Assume retrieved context may be incomplete by default. Do not claim global certainty \
unless completeness is explicitly proven by the provided context.

When referencing code, cite the source file and approximate line number \
in the format `file/path.py:line`.

Context:
{context}"""

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


# ---------------------------------------------------------------------------
# Session store — in-memory, unbounded (same semantics as
# ConversationBufferMemory).  Keyed by session_id.
# ---------------------------------------------------------------------------
_session_store: dict[str, ChatMessageHistory] = {}


def _get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]


def reset_session_store() -> None:
    """Clear all in-memory chat sessions."""
    _session_store.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into a context string for the prompt."""
    if not docs:
        return NO_CONTEXT_SENTINEL

    parts: list[str] = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        line = doc.metadata.get("start_line", "?")
        parts.append(f"--- {source}:{line} ---\n{doc.page_content}")
    return "\n\n".join(parts)


def _extract_sources(docs: list[Document]) -> list[str]:
    """Extract unique ``source:line`` references from retrieved documents."""
    seen: set[str] = set()
    sources: list[str] = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        line = doc.metadata.get("start_line", "?")
        ref = f"{source}:{line}"
        if ref not in seen:
            seen.add(ref)
            sources.append(ref)
    return sources


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ChatEngine:
    """RAG + conversational memory engine.

    Instantiated once and reused across requests. Builds a per-session
    retriever on each call so each session searches its own Chroma collection.
    """

    def __init__(self, settings: Settings, llm: BaseChatModel) -> None:
        self._settings = settings
        self._llm = llm

        # LCEL chain: prompt | llm | parser
        rag_chain = _PROMPT | self._llm | StrOutputParser()

        # Wrap with message history (auto-injects chat_history per session)
        self._chain = RunnableWithMessageHistory(
            rag_chain,
            _get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )

    def _prepare_context(self, docs: list[Document]) -> tuple[str, list[str]]:
        return _format_docs(docs), _extract_sources(docs)

    def ask(self, question: str, session_id: str) -> ChatResponse:
        """Synchronous RAG query."""
        retriever = get_retriever(self._settings, session_id)
        docs: list[Document] = retriever.invoke(question)
        context, sources = self._prepare_context(docs)

        answer: str = self._chain.invoke(
            {"question": question, "context": context},
            config={"configurable": {"session_id": session_id}},
        )
        logger.info(
            "Chat session=%s question_len=%d sources=%d",
            session_id,
            len(question),
            len(sources),
        )
        return ChatResponse(answer=answer, sources=sources)

    async def aask(self, question: str, session_id: str) -> ChatResponse:
        """Async RAG query."""
        retriever = get_retriever(self._settings, session_id)
        docs: list[Document] = await retriever.ainvoke(question)
        context, sources = self._prepare_context(docs)

        answer: str = await self._chain.ainvoke(
            {"question": question, "context": context},
            config={"configurable": {"session_id": session_id}},
        )
        logger.info(
            "Chat session=%s question_len=%d sources=%d (async)",
            session_id,
            len(question),
            len(sources),
        )
        return ChatResponse(answer=answer, sources=sources)


def build_chat_engine(settings: Settings) -> ChatEngine:
    """Factory — build the RAG chat engine."""
    llm = build_chat_model(settings)
    return ChatEngine(settings=settings, llm=llm)
