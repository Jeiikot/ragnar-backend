from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.documents import Document

from infrastructure.chat.engine import (
    ChatEngine,
    NO_CONTEXT_SENTINEL,
    RAG_SYSTEM_PROMPT,
    _extract_sources,
    _format_docs,
    _get_session_history,
)


class TestFormatDocs:
    def test_formats_single_doc(self) -> None:
        doc = Document(
            page_content="def hello(): pass",
            metadata={"source": "main.py", "start_line": 1},
        )
        result = _format_docs([doc])
        assert "main.py:1" in result
        assert "def hello(): pass" in result

    def test_formats_multiple_docs(self) -> None:
        docs = [
            Document(page_content="a", metadata={"source": "a.py", "start_line": 1}),
            Document(page_content="b", metadata={"source": "b.py", "start_line": 10}),
        ]
        result = _format_docs(docs)
        assert "a.py:1" in result
        assert "b.py:10" in result

    def test_handles_missing_metadata(self) -> None:
        doc = Document(page_content="code", metadata={})
        result = _format_docs([doc])
        assert "unknown:?" in result

    def test_returns_sentinel_when_no_docs(self) -> None:
        assert _format_docs([]) == NO_CONTEXT_SENTINEL


class TestExtractSources:
    def test_extracts_unique_sources(self) -> None:
        docs = [
            Document(page_content="a", metadata={"source": "a.py", "start_line": 1}),
            Document(page_content="b", metadata={"source": "a.py", "start_line": 1}),
            Document(page_content="c", metadata={"source": "b.py", "start_line": 5}),
        ]
        sources = _extract_sources(docs)
        assert sources == ["a.py:1", "b.py:5"]

    def test_empty_docs(self) -> None:
        assert _extract_sources([]) == []


class TestGetSessionHistory:
    def test_creates_new_session(self) -> None:
        history = _get_session_history("new_session_xyz")
        assert history is not None
        assert len(history.messages) == 0

    def test_reuses_existing_session(self) -> None:
        h1 = _get_session_history("reuse_test_session")
        h2 = _get_session_history("reuse_test_session")
        assert h1 is h2


class TestChatEngine:
    def test_ask_calls_retriever_and_chain(
        self,
        test_settings,  # type: ignore[no-untyped-def]
    ) -> None:
        with (
            patch("infrastructure.chat.engine.RunnableWithMessageHistory") as mock_history_cls,
            patch("infrastructure.chat.engine.get_retriever") as mock_get_retriever,
        ):
            mock_retriever = MagicMock()
            mock_retriever.invoke.return_value = [
                Document(
                    page_content="def foo(): pass",
                    metadata={"source": "foo.py", "start_line": 1},
                )
            ]
            mock_get_retriever.return_value = mock_retriever

            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "The foo function does X"
            mock_history_cls.return_value = mock_chain
            mock_llm = MagicMock()

            engine = ChatEngine(settings=test_settings, llm=mock_llm)
            result = engine.ask("what does foo do?", session_id="s1")

            mock_get_retriever.assert_called_once_with(test_settings, "s1")
            mock_retriever.invoke.assert_called_once_with("what does foo do?")
            mock_chain.invoke.assert_called_once()
            assert result.answer == "The foo function does X"
            assert "foo.py:1" in result.sources

    def test_ask_returns_fallback_when_no_docs(
        self,
        test_settings,  # type: ignore[no-untyped-def]
    ) -> None:
        with (
            patch("infrastructure.chat.engine.RunnableWithMessageHistory") as mock_history_cls,
            patch("infrastructure.chat.engine.get_retriever") as mock_get_retriever,
        ):
            mock_retriever = MagicMock()
            mock_retriever.invoke.return_value = []
            mock_get_retriever.return_value = mock_retriever

            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "No relevant context found. Try these questions..."
            mock_history_cls.return_value = mock_chain
            mock_llm = MagicMock()

            engine = ChatEngine(settings=test_settings, llm=mock_llm)
            result = engine.ask("what does foo do?", session_id="s1")

            assert result.answer == "No relevant context found. Try these questions..."
            assert result.sources == []
            mock_chain.invoke.assert_called_once_with(
                {"question": "what does foo do?", "context": NO_CONTEXT_SENTINEL},
                config={"configurable": {"session_id": "s1"}},
            )

    async def test_aask_returns_fallback_when_no_docs(
        self,
        test_settings,  # type: ignore[no-untyped-def]
    ) -> None:
        with (
            patch("infrastructure.chat.engine.RunnableWithMessageHistory") as mock_history_cls,
            patch("infrastructure.chat.engine.get_retriever") as mock_get_retriever,
        ):
            mock_retriever = MagicMock()
            mock_retriever.ainvoke = AsyncMock(return_value=[])
            mock_get_retriever.return_value = mock_retriever

            mock_chain = MagicMock()
            mock_chain.ainvoke = AsyncMock(
                return_value="No relevant context found. Try these questions..."
            )
            mock_history_cls.return_value = mock_chain
            mock_llm = MagicMock()

            engine = ChatEngine(settings=test_settings, llm=mock_llm)
            result = await engine.aask("what does foo do?", session_id="s1")

            assert result.answer == "No relevant context found. Try these questions..."
            assert result.sources == []
            mock_chain.ainvoke.assert_called_once_with(
                {"question": "what does foo do?", "context": NO_CONTEXT_SENTINEL},
                config={"configurable": {"session_id": "s1"}},
            )


class TestPromptInstructions:
    def test_requires_explicit_insufficient_context_format(self) -> None:
        assert "If the provided context is partial or missing key details" in RAG_SYSTEM_PROMPT
        assert "Start with one explicit limitation sentence." in RAG_SYSTEM_PROMPT
        assert "Add 3-5 suggested questions as a numbered list." in RAG_SYSTEM_PROMPT
        assert "Do not invent missing facts, counts, or coverage." in RAG_SYSTEM_PROMPT
        assert "Only suggest questions you can answer from the provided context." in RAG_SYSTEM_PROMPT
        assert "Assume retrieved context may be incomplete by default." in RAG_SYSTEM_PROMPT
