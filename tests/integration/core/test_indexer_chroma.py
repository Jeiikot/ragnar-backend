from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from application.indexing.service import index_directory
from infrastructure.indexing.adapters import build_indexing_ports
from shared.config import Settings


def _configure_embedding_mock(mock_emb: MagicMock) -> None:
    instance = MagicMock()
    instance.embed_documents.side_effect = lambda texts: [[0.1] * 10 for _ in texts]
    instance.embed_query.return_value = [0.1] * 10
    mock_emb.return_value = instance


@pytest.fixture
def integration_settings(tmp_path: Path) -> Settings:
    return Settings(
        embeddings_provider="openai",
        openai_api_key="fake-key",
        chroma_persist_dir=str(tmp_path / "chroma"),
        chroma_collection_name="integration_test",
        chunk_size=500,
        chunk_overlap=50,
    )


@pytest.fixture
def sample_project(tmp_path: Path) -> Path:
    project = tmp_path / "sample_project"
    project.mkdir()
    (project / "main.py").write_text(
        'def main():\n    print("hello world")\n\nif __name__ == "__main__":\n    main()\n'
    )
    (project / "utils.py").write_text(
        "def add(a: int, b: int) -> int:\n    return a + b\n\n"
        "def multiply(a: int, b: int) -> int:\n    return a * b\n"
    )
    sub = project / "sub"
    sub.mkdir()
    (sub / "helper.js").write_text(
        "function greet(name) {\n  return `Hello, ${name}!`;\n}\nmodule.exports = { greet };\n"
    )
    return project


@pytest.mark.integration
class TestIndexerWithChroma:
    def test_indexes_all_text_files(
        self, sample_project: Path, integration_settings: Settings
    ) -> None:
        # Use fake embeddings to avoid real API calls.
        with patch("infrastructure.indexing.storage.build_embeddings") as mock_emb:
            _configure_embedding_mock(mock_emb)
            ports = build_indexing_ports(integration_settings)
            count = index_directory(str(sample_project), ports, integration_settings)

        assert count >= 3  # chunks from all text files in the project

    def test_reindex_replaces_data(
        self, sample_project: Path, integration_settings: Settings
    ) -> None:
        with patch("infrastructure.indexing.storage.build_embeddings") as mock_emb:
            _configure_embedding_mock(mock_emb)
            ports = build_indexing_ports(integration_settings)
            count1 = index_directory(str(sample_project), ports, integration_settings)
            count2 = index_directory(str(sample_project), ports, integration_settings)

        assert count1 == count2
