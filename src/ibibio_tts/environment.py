"""Read-only environment checks for the data preparation pipeline."""

from __future__ import annotations

import hashlib
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_checksum(checksum_file: Path) -> str:
    fields = checksum_file.read_text(encoding="utf-8").strip().split()
    if not fields or len(fields[0]) != 64:
        raise ValueError(f"invalid SHA-256 record: {checksum_file}")
    return fields[0].lower()


def run_checks(
    project_root: Path,
    *,
    python_version: tuple[int, int] | None = None,
    minimum_free_mib: int = 1024,
) -> list[Check]:
    version = python_version or (sys.version_info.major, sys.version_info.minor)
    archive = project_root / (
        "data/raw/common_voice_ibb/"
        "common-voice-scripted-speech-26-0-ibibio-7ed511b3.tar.gz"
    )
    checksum_file = project_root / "reports/common_voice_ibb_archive.sha256"
    free_mib = shutil.disk_usage(project_root).free // (1024 * 1024)

    checks = [
        Check(
            "python",
            (3, 11) <= version < (3, 13),
            f"{version[0]}.{version[1]} (requires >=3.11,<3.13)",
        ),
        Check("ffmpeg", shutil.which("ffmpeg") is not None, "required for audio decoding"),
        Check("ffprobe", shutil.which("ffprobe") is not None, "required for audio inspection"),
        Check(
            "disk_space",
            free_mib >= minimum_free_mib,
            f"{free_mib} MiB free (minimum preflight threshold: {minimum_free_mib} MiB)",
        ),
        Check("archive", archive.is_file(), str(archive.relative_to(project_root))),
    ]

    if archive.is_file():
        try:
            expected = expected_checksum(checksum_file)
            actual = sha256(archive)
            checks.append(Check("archive_sha256", actual == expected, f"expected {expected}"))
        except (OSError, ValueError) as error:
            checks.append(Check("archive_sha256", False, str(error)))

    return checks
