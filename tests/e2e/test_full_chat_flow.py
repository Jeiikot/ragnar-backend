"""End-to-end test: index a sample project and ask questions.

Requires a real OPENAI_API_KEY. Skipped unless the ``e2e`` marker is
selected explicitly (``pytest -m e2e``).
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.e2e


@pytest.fixture
def e2e_settings(tmp_path: Path):
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    from shared.config import Settings

    return Settings(
        chat_provider="openai",
        embeddings_provider="openai",
        openai_api_key=api_key,
        chroma_persist_dir=str(tmp_path / "chroma_e2e"),
        chroma_collection_name="e2e_test",
        chunk_size=800,
        chunk_overlap=100,
    )


@pytest.fixture
def sample_project(tmp_path: Path) -> Path:
    project = tmp_path / "myapp"
    project.mkdir()
    (project / "auth.py").write_text(
        "import jwt\n\n"
        "SECRET = 'supersecret'\n\n"
        "def create_token(user_id: int) -> str:\n"
        '    return jwt.encode({"user_id": user_id}, SECRET, algorithm="HS256")\n\n'
        "def verify_token(token: str) -> dict:\n"
        '    return jwt.decode(token, SECRET, algorithms=["HS256"])\n'
    )
    (project / "main.py").write_text(
        "from auth import create_token, verify_token\n\n"
        "def login(user_id: int) -> str:\n"
        "    return create_token(user_id)\n"
    )
    return project


@pytest.fixture
def e2e_client(e2e_settings):
    from api.dependencies import get_app_settings
    from api.main import create_app

    app = create_app()
    app.dependency_overrides[get_app_settings] = lambda: e2e_settings
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


class TestFullChatFlow:
    @staticmethod
    def _sample_project_zip_bytes(sample_project: Path) -> bytes:
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, mode="w") as archive:
            for file_path in sample_project.rglob("*"):
                if file_path.is_file():
                    archive.write(
                        file_path,
                        arcname=file_path.relative_to(sample_project),
                    )
        return payload.getvalue()

    def test_index_then_ask(self, e2e_client: TestClient, sample_project: Path) -> None:
        # Step 1: Index
        zip_payload = self._sample_project_zip_bytes(sample_project)
        resp = e2e_client.post(
            "/api/v1/index/code",
            files={"file": ("repo.zip", zip_payload, "application/zip")},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["documents_indexed"] > 0

        # Step 2: Ask a question
        resp = e2e_client.post(
            "/api/v1/chat",
            json={
                "message": "How does the authentication module work?",
                "session_id": "e2e-1",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]["answer"]) > 10
        assert len(data["data"]["sources"]) >= 1

    def test_session_continuity(self, e2e_client: TestClient, sample_project: Path) -> None:
        # Index first
        zip_payload = self._sample_project_zip_bytes(sample_project)
        e2e_client.post(
            "/api/v1/index/code",
            files={"file": ("repo.zip", zip_payload, "application/zip")},
        )

        # First question
        e2e_client.post(
            "/api/v1/chat",
            json={
                "message": "What does auth.py do?",
                "session_id": "e2e-continuity",
            },
        )

        # Follow-up in same session
        resp = e2e_client.post(
            "/api/v1/chat",
            json={
                "message": "Tell me more about the token functions",
                "session_id": "e2e-continuity",
            },
        )
        assert resp.status_code == 200
        assert len(resp.json()["data"]["answer"]) > 10
