from __future__ import annotations

import logging
from pathlib import Path

import pathspec

logger = logging.getLogger(__name__)
BACKEND_ROOT = Path(__file__).resolve().parents[2]
LOCAL_IGNORE_FILE = BACKEND_ROOT / ".ignore"


def load_local_ignore_spec() -> pathspec.PathSpec:
    """
    Load required global ignore rules from backend/.ignore.
    """

    if not LOCAL_IGNORE_FILE.is_file():
        raise FileNotFoundError(f"Required ignore file not found: {LOCAL_IGNORE_FILE}")

    try:
        lines = LOCAL_IGNORE_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise OSError(f"Could not read required ignore file: {LOCAL_IGNORE_FILE}") from exc
    return pathspec.PathSpec.from_lines("gitignore", lines)


def collect_all_files(extracted_zip_root: Path) -> list[tuple[Path, str]]:
    """
    Walk extracted zip root and return files with extension, respecting backend/.ignore rules.
    """
    results: list[tuple[Path, str]] = []
    local_ignore_spec = load_local_ignore_spec()

    for file_path in extracted_zip_root.rglob("*"):
        if not file_path.is_file():
            continue

        relative = str(file_path.relative_to(extracted_zip_root))
        ext = file_path.suffix.lower()

        if local_ignore_spec.match_file(relative) or not ext:
            logger.debug("Skipping ignored file: %s", relative)
            continue

        results.append((file_path, ext))

    results.sort(key=lambda item: str(item[0]))
    return results
