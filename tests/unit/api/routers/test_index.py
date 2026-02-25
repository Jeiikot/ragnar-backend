from __future__ import annotations

from unittest.mock import patch


class TestIndexEndpoint:
    def test_directory_index_route_removed(self, client) -> None:  # type: ignore[no-untyped-def]
        resp = client.post(
            "/api/v1/index",
            json={"path": "/tmp", "extensions": [".py"]},
        )
        assert resp.status_code == 404

    def test_zip_index_success(self, client) -> None:  # type: ignore[no-untyped-def]
        with patch("api.routers.index.index_zip_bytes", return_value=12) as mock_index:
            resp = client.post(
                "/api/v1/index/code",
                files={"file": ("repo.zip", b"zip-content", "application/zip")},
            )

        assert resp.status_code == 200
        assert resp.json()["documents_indexed"] == 12
        mock_index.assert_called_once()

    def test_zip_index_rejects_non_zip_file(self, client) -> None:  # type: ignore[no-untyped-def]
        resp = client.post(
            "/api/v1/index/code",
            files={"file": ("repo.txt", b"text", "text/plain")},
        )

        assert resp.status_code == 400
        assert "zip" in resp.json()["detail"].lower()

    def test_zip_index_validation_error(self, client) -> None:  # type: ignore[no-untyped-def]
        with patch(
            "api.routers.index.index_zip_bytes",
            side_effect=ValueError("Invalid zip file"),
        ):
            resp = client.post(
                "/api/v1/index/code",
                files={"file": ("repo.zip", b"bad", "application/zip")},
            )

        assert resp.status_code == 500
        assert "indexing failed" in resp.json()["detail"].lower()

    def test_zip_index_internal_error(self, client) -> None:  # type: ignore[no-untyped-def]
        with patch(
            "api.routers.index.index_zip_bytes",
            side_effect=RuntimeError("boom"),
        ):
            resp = client.post(
                "/api/v1/index/code",
                files={"file": ("repo.zip", b"bad", "application/zip")},
            )

        assert resp.status_code == 500
