from __future__ import annotations

import io
import textwrap
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_text_splitters import Language

from application.indexing.service import index_directory, index_zip_bytes
from domain.indexing.ports.bundles import IndexingPorts
from infrastructure.indexing.chunking import get_splitter, load_and_split
from infrastructure.indexing.constants import EXTENSION_LANGUAGE_MAP
from infrastructure.indexing.file_discovery import collect_all_files


def _make_mock_ports(chunks: list[Document] | None = None) -> IndexingPorts:
    """Build a mock IndexingPorts with sensible defaults."""
    return IndexingPorts(
        collect_files=MagicMock(return_value=[]),
        split_file=MagicMock(return_value=chunks or []),
        write_documents=MagicMock(return_value=None),
        extract_zip=MagicMock(return_value=None),
    )


class TestExtensionLanguageMap:
    def test_python_mapped(self) -> None:
        assert EXTENSION_LANGUAGE_MAP[".py"] is Language.PYTHON

    def test_js_mapped(self) -> None:
        assert EXTENSION_LANGUAGE_MAP[".js"] is Language.JS

    def test_typescript_mapped(self) -> None:
        assert EXTENSION_LANGUAGE_MAP[".ts"] is Language.TS

    def test_go_mapped(self) -> None:
        assert EXTENSION_LANGUAGE_MAP[".go"] is Language.GO


class TestGetSplitter:
    def test_returns_language_splitter_for_python(self) -> None:
        splitter = get_splitter(".py", 1000, 100)
        assert splitter is not None

    def test_returns_generic_splitter_for_unknown(self) -> None:
        splitter = get_splitter(".xyz", 1000, 100)
        assert splitter is not None

    def test_respects_chunk_size(self) -> None:
        splitter = get_splitter(".txt", 500, 50)
        assert splitter._chunk_size == 500
        assert splitter._chunk_overlap == 50


class TestLoadAndSplit:
    def test_produces_chunks_with_metadata(self, tmp_path: Path) -> None:
        code = textwrap.dedent(
            """\
            def hello():
                return \"world\"

            def foo():
                return \"bar\"
        """
        )
        f = tmp_path / "example.py"
        f.write_text(code)
        chunks = load_and_split(f, ".py", tmp_path, 500, 50)
        assert len(chunks) >= 1
        assert chunks[0].metadata["source"] == "example.py"
        assert chunks[0].metadata["language"] == "python"
        assert chunks[0].metadata["file_extension"] == ".py"
        assert "start_line" in chunks[0].metadata

    def test_skips_empty_files(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.py"
        f.write_text("")
        chunks = load_and_split(f, ".py", tmp_path, 500, 50)
        assert chunks == []

    def test_handles_unreadable_file(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.py"
        f.write_text("code")
        f.chmod(0o000)
        chunks = load_and_split(f, ".py", tmp_path, 500, 50)
        assert chunks == []
        f.chmod(0o644)  # cleanup


class TestIndexDirectory:
    def test_raises_on_nonexistent_path(self, test_settings) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(FileNotFoundError):
            index_directory("/nonexistent/path", _make_mock_ports(), test_settings)

    def test_indexes_files_and_returns_count(
        self,
        tmp_path: Path,
        test_settings,  # type: ignore[no-untyped-def]
    ) -> None:
        (tmp_path / "a.py").write_text("def a(): pass")
        (tmp_path / "b.py").write_text("def b(): pass")

        fake_chunk = Document(page_content="x", metadata={})
        ports = _make_mock_ports()
        ports.collect_files.return_value = [
            (tmp_path / "a.py", ".py"),
            (tmp_path / "b.py", ".py"),
        ]
        ports.split_file.return_value = [fake_chunk]

        count = index_directory(str(tmp_path), ports, test_settings)
        assert count == 2
        ports.write_documents.assert_called_once()

    def test_returns_zero_for_no_indexable_files(
        self,
        tmp_path: Path,
        test_settings,  # type: ignore[no-untyped-def]
    ) -> None:
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        ports = _make_mock_ports()
        ports.collect_files.return_value = []

        count = index_directory(str(tmp_path), ports, test_settings)
        assert count == 0
        ports.write_documents.assert_not_called()


def _make_zip_bytes(files: dict[str, str]) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, mode="w") as archive:
        for filename, content in files.items():
            archive.writestr(filename, content)
    return payload.getvalue()


class TestIndexZipBytes:
    def test_indexes_zip_contents(
        self,
        test_settings,  # type: ignore[no-untyped-def]
    ) -> None:
        zip_payload = _make_zip_bytes({"repo/main.py": "def hello():\n    return 1\n"})
        fake_chunk = Document(page_content="x", metadata={})
        ports = _make_mock_ports()
        ports.collect_files.return_value = [(Path("repo/main.py"), ".py")]
        ports.split_file.return_value = [fake_chunk]

        count = index_zip_bytes(zip_payload, ports, test_settings)

        assert count >= 1
        ports.extract_zip.assert_called_once()
        ports.write_documents.assert_called_once()

    def test_rejects_invalid_zip(self, test_settings) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="Invalid zip file"):
            index_zip_bytes(b"not-a-valid-zip", _make_mock_ports(), test_settings)

    def test_rejects_zip_with_unsafe_paths(
        self,
        test_settings,  # type: ignore[no-untyped-def]
    ) -> None:
        zip_payload = _make_zip_bytes({"../escape.py": "print('bad')"})
        ports = _make_mock_ports()

        with pytest.raises(ValueError, match="unsafe paths"):
            index_zip_bytes(zip_payload, ports, test_settings)


class TestCollectAllFiles:
    def test_finds_all_text_files(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("print('hello')")
        (tmp_path / "app.js").write_text("console.log('hi')")
        (tmp_path / "style.css").write_text("body {}")
        result = collect_all_files(tmp_path)
        assert len(result) == 3

    def test_skips_binary_files(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("code")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        result = collect_all_files(tmp_path)
        assert len(result) == 1

    def test_skips_sensitive_files(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("code")
        (tmp_path / ".env").write_text("SECRET=abc")
        (tmp_path / "key.pem").write_text("-----BEGIN-----")
        result = collect_all_files(tmp_path)
        assert len(result) == 1

    def test_skips_lock_files(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("code")
        (tmp_path / "package-lock.json").write_text("{}")
        (tmp_path / "yarn.lock").write_text("# lock")
        result = collect_all_files(tmp_path)
        assert len(result) == 1

    def test_skips_files_without_extension(self, tmp_path: Path) -> None:
        (tmp_path / "README").write_text("text")
        (tmp_path / "main.py").write_text("code")
        result = collect_all_files(tmp_path)
        rel_paths = {str(path.relative_to(tmp_path)) for path, _ in result}
        assert "main.py" in rel_paths
        assert "README" not in rel_paths

    def test_ignores_project_gitignore_rules(self, tmp_path: Path) -> None:
        (tmp_path / ".gitignore").write_text("generated/\n*.log\n")
        (tmp_path / "main.py").write_text("code")
        gen_dir = tmp_path / "generated"
        gen_dir.mkdir()
        (gen_dir / "output.py").write_text("generated code")
        (tmp_path / "debug.log").write_text("log data")
        result = collect_all_files(tmp_path)
        rel_paths = {str(path.relative_to(tmp_path)) for path, _ in result}
        assert "main.py" in rel_paths
        assert "generated/output.py" in rel_paths
        assert "debug.log" in rel_paths
