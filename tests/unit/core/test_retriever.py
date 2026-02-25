from __future__ import annotations

from unittest.mock import MagicMock, patch

from infrastructure.retriever import get_retriever


class TestGetRetriever:
    @patch("infrastructure.retriever.Chroma")
    @patch("infrastructure.retriever.build_embeddings")
    def test_creates_retriever_with_correct_params(
        self,
        mock_build_embeddings: MagicMock,
        mock_chroma_cls: MagicMock,
        test_settings,  # type: ignore[no-untyped-def]
    ) -> None:
        mock_vs = MagicMock()
        mock_chroma_cls.return_value = mock_vs
        mock_build_embeddings.return_value = MagicMock()
        mock_retriever = MagicMock()
        mock_vs.as_retriever.return_value = mock_retriever

        result = get_retriever(test_settings)

        mock_build_embeddings.assert_called_once_with(test_settings)
        mock_chroma_cls.assert_called_once()
        mock_vs.as_retriever.assert_called_once_with(
            search_type=test_settings.retriever_search_type,
            search_kwargs={
                "k": test_settings.retriever_k,
                "fetch_k": test_settings.retriever_fetch_k,
            },
        )
        assert result is mock_retriever

    @patch("infrastructure.retriever.Chroma")
    @patch("infrastructure.retriever.build_embeddings")
    def test_similarity_search_omits_fetch_k(
        self,
        mock_build_embeddings: MagicMock,
        mock_chroma_cls: MagicMock,
        test_settings,  # type: ignore[no-untyped-def]
    ) -> None:
        test_settings.retriever_search_type = "similarity"
        mock_vs = MagicMock()
        mock_chroma_cls.return_value = mock_vs
        mock_build_embeddings.return_value = MagicMock()
        mock_vs.as_retriever.return_value = MagicMock()

        get_retriever(test_settings)

        call_kwargs = mock_vs.as_retriever.call_args
        assert "fetch_k" not in call_kwargs.kwargs.get("search_kwargs", {})
