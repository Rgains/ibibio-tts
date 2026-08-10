"""Minimal client for the Mozilla Data Collective dataset download API.

Reference: https://mozilladatacollective.com/api-reference/docs
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASE_URL = "https://mozilladatacollective.com/api"


@dataclass(frozen=True)
class DatasetMetadata:
    id: str
    filename: str
    checksum: str
    size_bytes: int
    locale: str
    license: str


@dataclass(frozen=True)
class DownloadSession:
    download_url: str
    filename: str
    checksum: str
    size_bytes: int


def _get_json(url: str, api_key: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(request) as response:
        return json.load(response)  # type: ignore[no-any-return]


def _post_json(url: str, api_key: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        method="POST",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)  # type: ignore[no-any-return]


def fetch_metadata(dataset_id: str, api_key: str) -> DatasetMetadata:
    payload = _get_json(f"{BASE_URL}/datasets/{dataset_id}", api_key)
    return DatasetMetadata(
        id=payload["id"],
        filename=payload["filename"],
        checksum=payload["checksum"],
        size_bytes=payload["sizeBytes"],
        locale=payload["locale"],
        license=payload["licenseAbbreviation"],
    )


def request_download(dataset_id: str, api_key: str) -> DownloadSession:
    payload = _post_json(f"{BASE_URL}/datasets/{dataset_id}/download", api_key)
    return DownloadSession(
        download_url=payload["downloadUrl"],
        filename=payload["filename"],
        checksum=payload["checksum"],
        size_bytes=payload["sizeBytes"],
    )


def download_to(session: DownloadSession, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(session.download_url) as response, tmp.open("wb") as out:
        while chunk := response.read(1024 * 1024):
            out.write(chunk)
    tmp.rename(dest)
    return dest
