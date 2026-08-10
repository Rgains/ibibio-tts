from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd

from ibibio_tts.importer import import_validated_tsv


def _write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def test_import_validated_tsv_tracks_rejections(tmp_path: Path) -> None:
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    (clips_dir / "a.mp3").write_bytes(b"audio")

    validated_tsv = tmp_path / "validated.tsv"
    _write_tsv(
        validated_tsv,
        [
            {"client_id": "c1", "path": "a.mp3", "sentence": "Ndinamke mmo"},
            {"client_id": "c1", "path": "a.mp3", "sentence": "Ndinamke mmo"},  # duplicate
            {"client_id": "c2", "path": "missing.mp3", "sentence": "hello"},  # missing audio
            {"client_id": "c3", "path": "b.mp3", "sentence": ""},  # empty sentence
            {"client_id": "c4", "path": "", "sentence": "hello"},  # empty path
        ],
    )

    manifest_out = tmp_path / "out" / "imported.parquet"
    rejections_out = tmp_path / "out" / "rejections.csv"

    summary = import_validated_tsv(validated_tsv, clips_dir, manifest_out, rejections_out)

    assert summary.imported == 1
    assert summary.rejected == 4

    manifest = pd.read_parquet(manifest_out)
    assert manifest.loc[0, "client_id"] == "c1"
    assert manifest.loc[0, "normalized_text"] == "Ndinamke mmo"
    assert manifest.loc[0, "text_corrected"] is False or not manifest.loc[0, "text_corrected"]

    rejections = pd.read_csv(rejections_out)
    assert set(rejections["reason"]) == {
        "duplicate_row",
        "missing_audio_file",
        "empty_sentence",
        "empty_path",
    }


def test_import_validated_tsv_applies_reviewed_correction(tmp_path: Path) -> None:
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    (clips_dir / "a.mp3").write_bytes(b"audio")

    cyrillic_schwa = unicodedata.lookup("CYRILLIC SMALL LETTER SCHWA")
    latin_schwa = unicodedata.lookup("LATIN SMALL LETTER SCHWA")

    validated_tsv = tmp_path / "validated.tsv"
    _write_tsv(
        validated_tsv,
        [{"client_id": "c1", "path": "a.mp3", "sentence": "et" + cyrillic_schwa + "k"}],
    )

    manifest_out = tmp_path / "out" / "imported.parquet"
    rejections_out = tmp_path / "out" / "rejections.csv"
    summary = import_validated_tsv(validated_tsv, clips_dir, manifest_out, rejections_out)

    assert summary.imported == 1
    manifest = pd.read_parquet(manifest_out)
    assert manifest.loc[0, "normalized_text"] == "et" + latin_schwa + "k"
    assert bool(manifest.loc[0, "text_corrected"]) is True
    assert manifest.loc[0, "flags"] == ""


def test_import_validated_tsv_requires_columns(tmp_path: Path) -> None:
    validated_tsv = tmp_path / "validated.tsv"
    _write_tsv(validated_tsv, [{"client_id": "c1"}])

    try:
        import_validated_tsv(
            validated_tsv, tmp_path, tmp_path / "m.parquet", tmp_path / "r.csv"
        )
        raised = False
    except ValueError:
        raised = True
    assert raised
