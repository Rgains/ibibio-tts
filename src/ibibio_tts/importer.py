"""Import Common Voice `validated.tsv`, tracking every rejection.

Per `PROTOCOL.md`: use only `validated.tsv`, preserve raw and normalized text,
retain anonymous client IDs, never identify speakers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ibibio_tts.text import apply_reviewed_corrections, flag_text, normalize_text

REQUIRED_COLUMNS = ("client_id", "path", "sentence")


@dataclass(frozen=True)
class ImportSummary:
    imported: int
    rejected: int


def _reject(reasons: list[dict[str, object]], row_index: int, path: object, reason: str) -> None:
    reasons.append({"row_index": row_index, "path": path, "reason": reason})


def import_validated_tsv(
    validated_tsv: Path,
    clips_dir: Path,
    manifest_out: Path,
    rejections_out: Path,
) -> ImportSummary:
    frame = pd.read_csv(validated_tsv, sep="\t", dtype=str, keep_default_na=False)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"validated.tsv missing required columns: {missing}")

    rejections: list[dict[str, object]] = []
    seen_pairs: set[tuple[str, str]] = set()
    records: list[dict[str, object]] = []

    for row_index, row in enumerate(frame.to_dict(orient="records")):
        client_id = str(row["client_id"]).strip()
        clip_path = str(row["path"]).strip()
        raw_text = str(row["sentence"])

        if not clip_path:
            _reject(rejections, row_index, clip_path, "empty_path")
            continue
        if not raw_text.strip():
            _reject(rejections, row_index, clip_path, "empty_sentence")
            continue

        pair = (client_id, clip_path)
        if pair in seen_pairs:
            _reject(rejections, row_index, clip_path, "duplicate_row")
            continue
        seen_pairs.add(pair)

        audio_path = clips_dir / clip_path
        if not audio_path.is_file():
            _reject(rejections, row_index, clip_path, "missing_audio_file")
            continue

        normalized_text = normalize_text(raw_text)
        reviewed_text = apply_reviewed_corrections(normalized_text)
        flags = flag_text(reviewed_text)
        records.append(
            {
                "client_id": client_id,
                "clip_path": clip_path,
                "raw_text": raw_text,
                "normalized_text": reviewed_text,
                "text_corrected": reviewed_text != normalized_text,
                "flags": ";".join(flags),
            }
        )

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_parquet(manifest_out, index=False)

    rejections_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(rejections).to_csv(rejections_out, index=False)

    return ImportSummary(imported=len(records), rejected=len(rejections))
