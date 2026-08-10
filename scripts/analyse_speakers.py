#!/usr/bin/env python3
"""Anonymous-speaker audit. Never attempts to identify Common Voice speakers.

Aggregates by the anonymous `client_id` already provided by Common Voice --
no re-identification, demographic inference, or voice fingerprinting.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

IMPORTED_IN = Path("data/interim/common_voice_ibb/imported.parquet")
AUDITED_IN = Path("data/interim/common_voice_ibb/audited.parquet")
REPORT_JSON = Path("reports/speaker_audit.json")
REPORT_MD = Path("reports/speaker_audit.md")


def analyse(project_root: Path) -> dict[str, Any]:
    imported = pd.read_parquet(project_root / IMPORTED_IN)
    audited = pd.read_parquet(project_root / AUDITED_IN)
    merged = imported.merge(audited, on="clip_path", how="left")

    usable = merged[merged["usable"] == True]  # noqa: E712

    per_speaker = (
        usable.groupby("client_id")
        .agg(
            usable_clips=("clip_path", "count"),
            usable_duration_ms=("duration_ms", "sum"),
            unique_sentences=("normalized_text", pd.Series.nunique),
        )
        .reset_index()
    )
    per_speaker["usable_hours"] = (per_speaker["usable_duration_ms"] / 3_600_000).round(6)
    per_speaker = per_speaker.sort_values("usable_hours", ascending=False)

    ranking = [
        {
            "speaker_id": row["client_id"],
            "usable_clips": int(row["usable_clips"]),
            "usable_hours": float(row["usable_hours"]),
            "unique_sentences": int(row["unique_sentences"]),
        }
        for row in per_speaker.to_dict(orient="records")
    ]

    total_speakers = merged["client_id"].nunique()
    strongest_hours = ranking[0]["usable_hours"] if ranking else 0.0

    report: dict[str, Any] = {
        "total_speakers": int(total_speakers),
        "speakers_with_usable_audio": len(ranking),
        "strongest_speaker_usable_hours": strongest_hours,
        "total_usable_hours": round(sum(entry["usable_hours"] for entry in ranking), 6),
        "ranking": ranking,
        "limitations": [
            "Speaker IDs are the anonymous Common Voice client_id; no re-identification "
            "was attempted or is supported by this audit.",
            "Ranking reflects usable recorded duration only, not recording quality or "
            "consent for any form of identity- or voice-specific use.",
        ],
    }

    report_json = project_root / REPORT_JSON
    report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    md_lines = [
        "# Ibibio Common Voice anonymous-speaker audit",
        "",
        f"- Total speakers (anonymous client IDs): {report['total_speakers']}",
        f"- Speakers with usable audio: {report['speakers_with_usable_audio']}",
        f"- Total usable hours: {report['total_usable_hours']:.3f}",
        "",
        "| Rank | Anonymous speaker | Usable clips | Usable hours | Unique sentences |",
        "|---:|---|---:|---:|---:|",
    ]
    for rank, entry in enumerate(ranking, start=1):
        md_lines.append(
            f"| {rank} | `{entry['speaker_id']}` | {entry['usable_clips']} | "
            f"{entry['usable_hours']:.3f} | {entry['unique_sentences']} |"
        )
    (project_root / REPORT_MD).write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return report


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    started = time.monotonic()
    report = analyse(project_root)
    elapsed = time.monotonic() - started
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[analyse_speakers] finished in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
