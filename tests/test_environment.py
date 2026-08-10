from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

from pytest import MonkeyPatch

import ibibio_tts.environment as environment


def test_expected_checksum_and_sha256(tmp_path: Path) -> None:
    archive = tmp_path / "archive.tar.gz"
    archive.write_bytes(b"Ibibio")
    digest = hashlib.sha256(b"Ibibio").hexdigest()
    record = tmp_path / "archive.sha256"
    record.write_text(f"{digest}  archive.tar.gz\n", encoding="utf-8")

    assert environment.expected_checksum(record) == digest
    assert environment.sha256(archive) == digest


def test_run_checks_reports_missing_archive(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "common_voice_ibb_archive.sha256").write_text("0" * 64, encoding="utf-8")
    monkeypatch.setattr(environment.shutil, "which", lambda _name: "/usr/bin/tool")
    monkeypatch.setattr(
        environment.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=9),
    )

    checks = environment.run_checks(tmp_path, python_version=(3, 11), minimum_free_mib=0)
    results = {check.name: check.passed for check in checks}

    assert results["python"] is True
    assert results["archive"] is False
    assert "archive_sha256" not in results
