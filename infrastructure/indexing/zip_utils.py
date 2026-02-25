from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


def extract_zip_safely(zip_archive: zipfile.ZipFile, destination: Path) -> None:
    """Extract files from *archive* into *destination* with path traversal protection."""
    for archive in zip_archive.infolist():
        member_path = Path(archive.filename)

        # Check if the file is a directory
        if archive.is_dir():
            continue

        # Check for absolute paths or path traversal
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError("Zip file contains unsafe paths")

        # Skip empty or directory paths
        if not archive.filename or archive.filename.endswith("/"):
            continue

        # Create the target path and ensure parent directories exist
        target_path = destination / member_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Open the archive member and write it to the target path
        with zip_archive.open(archive, "r") as source, target_path.open("wb") as target:
            shutil.copyfileobj(source, target)
