#!/usr/bin/env python3
"""Bundle a trained VITS checkpoint into the flat layout the demo expects.

Produces `model.pth`, `config.json`, `speakers.pth`, `speaker_labels.json`
(and a copy of `metrics.json` for reference) in one directory -- this is
both what `scripts/train_vits_full.py`'s run leaves scattered under a
timestamped run folder, and what `scripts/push_model_to_hf.py` uploads
as-is. Speaker labels are anonymous ("Speaker 1", ...), sorted by training
clip count, never exposing the raw client-id hash -- see
`src/ibibio_tts/inference.py`.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def build_speaker_labels(manifest_dir: Path, speaker_names: set[str]) -> list[dict]:
    import pandas as pd

    train = pd.read_parquet(manifest_dir / "train.parquet")
    train = train[train["flags"] == ""]
    counts = train["client_id"].value_counts()
    ranked = sorted(
        speaker_names, key=lambda name: counts.get(name, 0), reverse=True
    )
    return [
        {"label": f"Speaker {i + 1}", "speaker_name": name, "clip_count": int(counts.get(name, 0))}
        for i, name in enumerate(ranked)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="Timestamped training run directory"
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/processed/common_voice_ibb/manifests",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "model_export",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = run_dir / "best_model.pth"
    if not checkpoint.exists():
        checkpoint = max(run_dir.glob("checkpoint_*.pth"), key=lambda p: p.stat().st_mtime)
    shutil.copy(checkpoint, output_dir / "model.pth")
    shutil.copy(run_dir / "config.json", output_dir / "config.json")
    shutil.copy(run_dir / "speakers.pth", output_dir / "speakers.pth")
    if (run_dir / "metrics.json").exists():
        shutil.copy(run_dir / "metrics.json", output_dir / "metrics.json")

    import torch

    speakers_blob = torch.load(run_dir / "speakers.pth", weights_only=False)
    speaker_names = set(speakers_blob.get("name_to_id", speakers_blob).keys())
    labels = build_speaker_labels(args.manifest_dir, speaker_names)
    (output_dir / "speaker_labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Exported demo model bundle to {output_dir}")
    print(f"  checkpoint: {checkpoint.name}")
    print(f"  speakers:   {len(labels)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
