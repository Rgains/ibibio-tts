#!/usr/bin/env python3
"""Download, extract, and import the Common Voice Ibibio validated split.

Required order (see PROTOCOL.md): environment must already be validated.
This script downloads the archive via the Mozilla Data Collective API,
verifies its checksum, extracts it, and imports `validated.tsv`, tracking
every rejection. It never overwrites raw audio and never touches speaker
identity beyond the anonymous `client_id` already provided by Common Voice.
"""

from __future__ import annotations

import shutil
import tarfile
import time
from pathlib import Path

from ibibio_tts import mdc_client
from ibibio_tts.config import load_dotenv, require_env
from ibibio_tts.environment import expected_checksum, sha256
from ibibio_tts.importer import import_validated_tsv

ARCHIVE_RELATIVE = Path(
    "data/raw/common_voice_ibb/common-voice-scripted-speech-26-0-ibibio-7ed511b3.tar.gz"
)
EXTRACTED_RELATIVE = Path(
    "data/raw/common_voice_ibb/cv-corpus-26.0-2026-06-12/ibb"
)


def download(project_root: Path, api_key: str, dataset_id: str) -> Path:
    archive = project_root / ARCHIVE_RELATIVE
    checksum_file = project_root / "reports/common_voice_ibb_archive.sha256"
    expected = expected_checksum(checksum_file)

    if archive.is_file() and sha256(archive) == expected:
        print(f"[skip] archive already present and verified: {archive}")
        return archive

    metadata = mdc_client.fetch_metadata(dataset_id, api_key)
    if metadata.checksum != expected:
        raise ValueError(
            f"dataset metadata checksum {metadata.checksum} does not match "
            f"pinned checksum {expected} in {checksum_file}"
        )

    print(f"[download] {metadata.filename} ({metadata.size_bytes / 1e6:.1f} MB)")
    started = time.monotonic()
    session = mdc_client.request_download(dataset_id, api_key)
    mdc_client.download_to(session, archive)
    elapsed = time.monotonic() - started
    print(f"[download] finished in {elapsed:.1f}s")

    actual = sha256(archive)
    if actual != expected:
        archive.unlink()
        raise ValueError(f"checksum mismatch: expected {expected}, got {actual}")
    print("[download] checksum verified")
    return archive


def extract(project_root: Path, archive: Path) -> Path:
    extracted_root = project_root / EXTRACTED_RELATIVE
    if extracted_root.is_dir():
        print(f"[skip] already extracted: {extracted_root}")
        return extracted_root

    print(f"[extract] {archive} -> {archive.parent}")
    started = time.monotonic()
    with tarfile.open(archive) as tar:
        tar.extractall(archive.parent, filter="data")
    print(f"[extract] finished in {time.monotonic() - started:.1f}s")
    return extracted_root


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root)
    api_key = require_env("YOUR_API_KEY")
    dataset_id = require_env("DATASET_ID")

    free_mib = shutil.disk_usage(project_root).free // (1024 * 1024)
    if free_mib < 2048:
        raise SystemExit(f"only {free_mib} MiB free; need headroom for download + extraction")

    archive = download(project_root, api_key, dataset_id)
    extracted_root = extract(project_root, archive)

    validated_tsv = extracted_root / "validated.tsv"
    clips_dir = extracted_root / "clips"
    manifest_out = project_root / "data/interim/common_voice_ibb/imported.parquet"
    rejections_out = project_root / "reports/import_rejections.csv"

    print(f"[import] {validated_tsv}")
    started = time.monotonic()
    summary = import_validated_tsv(validated_tsv, clips_dir, manifest_out, rejections_out)
    print(
        f"[import] {summary.imported} imported, {summary.rejected} rejected "
        f"in {time.monotonic() - started:.1f}s"
    )
    print(f"[import] manifest: {manifest_out}")
    print(f"[import] rejections: {rejections_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
