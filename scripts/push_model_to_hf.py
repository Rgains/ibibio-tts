#!/usr/bin/env python3
"""Push an exported demo model bundle (see export_model_for_demo.py) to a
Hugging Face Hub model repo, so `app.py` can download it on Streamlit Cloud
or any machine without the training checkpoint locally.

Requires an HF token with write access, either via `--token` or the
`HF_TOKEN` env var (`huggingface-cli login` also works). Creates the repo
if it does not exist yet.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ibibio_tts.inference import MODEL_FILES

MODEL_CARD = """---
license: cc0-1.0
language: ibb
tags:
  - text-to-speech
  - vits
  - coqui-tts
  - ibibio
---

# Ibibio Common Voice TTS -- research baseline

Multi-speaker VITS model trained from scratch on 7.68 hours of Mozilla
Common Voice Scripted Speech 26.0 (Ibibio locale, CC0-1.0). See the
`ibibio-tts` project repository's `papers.md` for the full training log
and `PROTOCOL.md` for the data pipeline this model came from.

**This is a research baseline, not a production voice.** A native Ibibio
speaker evaluated this model's output and found intonation plausible but
individual words often unclear -- see `papers.md`'s "root cause confirmed
and fixed" and native-speaker-evaluation entries for the full
investigation and why (from-scratch character-level input with no
Ibibio phonemizer available, 7.68h total data, imbalanced per-speaker
clip counts).

Speaker identities are anonymized: the 17 speakers are only ever labeled
by training-clip volume ("Speaker 1" = most training data), never by the
original Common Voice speaker ID.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="e.g. your-username/ibibio-tts-vits")
    parser.add_argument(
        "--model-dir", type=Path, default=Path(__file__).resolve().parents[1] / "model_export"
    )
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    missing = [name for name in MODEL_FILES if not (args.model_dir / name).exists()]
    if missing:
        raise SystemExit(
            f"Missing {missing} in {args.model_dir} -- run scripts/export_model_for_demo.py first."
        )
    if not args.token:
        raise SystemExit("No Hugging Face token: pass --token or set HF_TOKEN.")

    from huggingface_hub import HfApi

    api = HfApi(token=args.token)
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        folder_path=str(args.model_dir),
        allow_patterns=[*MODEL_FILES, "metrics.json"],
    )
    (args.model_dir / "README.md").write_text(MODEL_CARD, encoding="utf-8")
    api.upload_file(
        repo_id=args.repo_id,
        path_or_fileobj=str(args.model_dir / "README.md"),
        path_in_repo="README.md",
    )
    print(f"Pushed {args.model_dir} to https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
