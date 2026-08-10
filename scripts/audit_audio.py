#!/usr/bin/env python3
"""Audio audit: decode every imported clip once, resample, and report.

Per `PROTOCOL.md`: decode source audio once to mono 24 kHz PCM16 WAV, accept
1-15 second clips, never overwrite raw audio or automatically denoise it.
Writes `data/processed/common_voice_ibb/wav/` and `reports/audio_audit.*`.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from ibibio_tts.audio import decode_to_mono_pcm16, is_usable_duration, sha256_file, write_wav_pcm16

MANIFEST_IN = Path("data/interim/common_voice_ibb/imported.parquet")
CLIPS_DIR = Path("data/raw/common_voice_ibb/cv-corpus-26.0-2026-06-12/ibb/clips")
WAV_DIR = Path("data/processed/common_voice_ibb/wav")
AUDITED_OUT = Path("data/interim/common_voice_ibb/audited.parquet")
REPORT_JSON = Path("reports/audio_audit.json")
REPORT_MD = Path("reports/audio_audit.md")


def audit(project_root: Path) -> dict[str, Any]:
    manifest = pd.read_parquet(project_root / MANIFEST_IN)
    records: list[dict[str, Any]] = []

    for row in manifest.to_dict(orient="records"):
        clip_path = str(row["clip_path"])
        source = project_root / CLIPS_DIR / clip_path
        result, samples = decode_to_mono_pcm16(source)

        record: dict[str, Any] = {
            "clip_path": clip_path,
            "decode_status": result.decode_status,
            "error": result.error,
            "source_sample_rate": result.source_sample_rate,
            "source_channels": result.source_channels,
            "duration_ms": result.duration_ms,
            "audio_sha256": sha256_file(source),
            "usable": False,
            "processed_wav_path": "",
        }

        if result.decode_status == "ok" and samples is not None:
            usable = is_usable_duration(result.duration_ms)
            record["usable"] = usable
            if usable:
                wav_path = project_root / WAV_DIR / clip_path.replace(".mp3", ".wav")
                write_wav_pcm16(wav_path, samples)
                record["processed_wav_path"] = str(wav_path.relative_to(project_root))

        records.append(record)

    audited = pd.DataFrame.from_records(records)
    audited_out = project_root / AUDITED_OUT
    audited_out.parent.mkdir(parents=True, exist_ok=True)
    audited.to_parquet(audited_out, index=False)

    ok = audited[audited["decode_status"] == "ok"]
    usable = audited[audited["usable"]]
    durations = ok["duration_ms"].dropna().tolist()

    report: dict[str, Any] = {
        "clips_total": len(audited),
        "decode_ok": len(ok),
        "decode_failed": len(audited) - len(ok),
        "usable_clips": len(usable),
        "usable_definition": "decode ok and 1-15 second duration",
        "usable_hours": round(usable["duration_ms"].sum() / 3_600_000, 6) if len(usable) else 0.0,
        "source_sample_rate_distribution": dict(
            sorted(Counter(int(v) for v in ok["source_sample_rate"].dropna()).items())
        ),
        "source_channel_distribution": dict(
            sorted(Counter(int(v) for v in ok["source_channels"].dropna()).items())
        ),
        "duration_ms_distribution": {
            "minimum": min(durations) if durations else None,
            "median": statistics.median(durations) if durations else None,
            "maximum": max(durations) if durations else None,
        },
        "duplicate_audio_content": int((audited["audio_sha256"].value_counts() > 1).sum()),
        "decode_failures": audited.loc[audited["decode_status"] != "ok", ["clip_path", "error"]]
        .to_dict(orient="records"),
    }

    report_json = project_root / REPORT_JSON
    report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    md_lines = [
        "# Ibibio Common Voice audio audit",
        "",
        f"- Clips: {report['clips_total']}",
        f"- Decode ok: {report['decode_ok']} / decode failed: {report['decode_failed']}",
        f"- Usable ({report['usable_definition']}): {report['usable_clips']} "
        f"({report['usable_hours']:.3f} hours)",
        f"- Source sample rates: {report['source_sample_rate_distribution']}",
        f"- Source channels: {report['source_channel_distribution']}",
        f"- Duration ms (min/median/max): {report['duration_ms_distribution']}",
        f"- Clips sharing identical audio content (sha256): {report['duplicate_audio_content']}",
    ]
    (project_root / REPORT_MD).write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return report


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    started = time.monotonic()
    report = audit(project_root)
    elapsed = time.monotonic() - started
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[audit_audio] finished in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
