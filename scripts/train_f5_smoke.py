#!/usr/bin/env python3
"""F5-TTS (flow matching, DiT backbone) smoke experiment for Ibibio Common Voice.

Optional research config (`configs/model/f5_research.yaml`) -- NOT part of
PROTOCOL.md's required order, unlike the VITS smoke experiment. Trained from
scratch with a custom Ibibio character vocabulary, not fine-tuned from the
public F5TTS_v1_Base checkpoint: that checkpoint's vocab is Chinese-pinyin /
English only and does not cover Ibibio graphemes (e.g. ọ, ị, ñ), so a real
fine-tune would first need embedding-table surgery to extend it. Training
from scratch matches this project's VITS smoke test in spirit: it proves
the pipeline works end to end, and makes no naturalness or quality claim.
The DiT backbone is deliberately shrunk from the paper's 1024/22/16 config
so a from-scratch smoke run finishes in minutes on one GPU; it exercises
the same code path (CFM + DiT + mel spec + Arrow dataset + Trainer), not a
scaled attempt at real quality.

The vocoder (Vocos) is used pretrained/frozen, as F5-TTS is designed to be
used -- only the text-to-mel flow-matching model is trained from scratch
here, not the vocoder.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from datasets import Dataset as ArrowDataset
from datasets.arrow_writer import ArrowWriter
from f5_tts.infer.utils_infer import infer_process, load_vocoder
from f5_tts.model import CFM, DiT
from f5_tts.model.dataset import CustomDataset
from f5_tts.model.trainer import Trainer

SEED = 20260804
LANGUAGE = "ibb"
TARGET_SAMPLE_RATE = 24_000
HOP_LENGTH = 256
N_MEL_CHANNELS = 100


def load_split(manifest_dir: Path, split: str) -> pd.DataFrame:
    frame = pd.read_parquet(manifest_dir / f"{split}.parquet")
    return frame[frame["flags"] == ""].reset_index(drop=True)


def build_vocab(*text_frames: pd.Series) -> tuple[dict[str, int], list[str]]:
    chars: set[str] = set()
    for series in text_frames:
        for text in series:
            chars.update(text)
    ordered = sorted(chars)
    vocab_char_map = {char: index for index, char in enumerate(ordered)}
    return vocab_char_map, ordered


def write_arrow_dataset(data_dir: Path, rows: pd.DataFrame, project_root: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    durations: list[float] = []
    with ArrowWriter(path=str(data_dir / "raw.arrow")) as writer:
        for row in rows.to_dict(orient="records"):
            duration_s = float(row["duration_ms"]) / 1000.0
            writer.write(
                {
                    "audio_path": str(project_root / row["processed_wav_path"]),
                    "text": row["normalized_text"],
                    "duration": duration_s,
                }
            )
            durations.append(duration_s)
        writer.finalize()
    (data_dir / "duration.json").write_text(
        json.dumps({"duration": durations}, ensure_ascii=False), encoding="utf-8"
    )


def run(args: Any) -> dict[str, Any]:
    run_started = time.perf_counter()
    torch.manual_seed(SEED)
    rng = random.Random(SEED)

    project_root = args.project_root.resolve()
    manifest_dir = project_root / args.manifest_dir

    train_all = load_split(manifest_dir, "train")
    eval_all = load_split(manifest_dir, "validation")

    train_indices = sorted(rng.sample(range(len(train_all)), min(args.clips, len(train_all))))
    train_rows = train_all.iloc[train_indices].reset_index(drop=True)

    # Held-out synthesis sentences and a reference clip come from validation,
    # never from the training subset -- keeps the smoke-test synthesis
    # honest even though F5 has no built-in eval loop to check against.
    gen_indices = sorted(
        rng.sample(range(len(eval_all)), min(args.generated_samples + 1, len(eval_all)))
    )
    gen_rows = eval_all.iloc[gen_indices].reset_index(drop=True)
    ref_row = gen_rows.iloc[0]
    synth_rows = gen_rows.iloc[1:].reset_index(drop=True)

    vocab_char_map, vocab_list = build_vocab(
        train_rows["normalized_text"], gen_rows["normalized_text"]
    )

    data_dir = project_root / args.data_dir
    write_arrow_dataset(data_dir, train_rows, project_root)
    (data_dir / "vocab.txt").write_text("\n".join(vocab_list) + "\n", encoding="utf-8")

    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    mel_spec_kwargs = dict(
        n_fft=1024,
        hop_length=HOP_LENGTH,
        win_length=1024,
        n_mel_channels=N_MEL_CHANNELS,
        target_sample_rate=TARGET_SAMPLE_RATE,
        mel_spec_type="vocos",
    )

    # Deliberately shrunk DiT (paper/F5TTS_v1_Base uses dim=1024, depth=22,
    # heads=16) -- this is a from-scratch smoke test on 200 clips, not an
    # attempt at the paper's scale or quality.
    model_cfg = dict(dim=256, depth=4, heads=4, ff_mult=2, text_dim=128, conv_layers=2)
    model = CFM(
        transformer=DiT(
            **model_cfg,
            mel_dim=N_MEL_CHANNELS,
            text_num_embeds=len(vocab_char_map),
        ),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )
    parameter_count = sum(p.numel() for p in model.parameters())

    trainer = Trainer(
        model,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_warmup_updates=args.warmup_updates,
        save_per_updates=args.save_per_updates,
        last_per_updates=args.save_per_updates,
        keep_last_n_checkpoints=1,
        checkpoint_path=str(output_root),
        batch_size_per_gpu=args.batch_size,
        batch_size_type="sample",
        grad_accumulation_steps=1,
        logger=None,
        log_samples=False,
        mel_spec_type="vocos",
        model_cfg_dict=model_cfg,
    )

    # f5_tts.model.dataset.load_dataset()'s "CustomDatasetPath" branch has an
    # upstream bug in f5-tts==1.1.22 (references `preprocessed_mel` before
    # assignment), so build the CustomDataset directly instead -- this is
    # exactly what that branch does when the bug is fixed.
    raw_dataset = ArrowDataset.from_file(str(data_dir / "raw.arrow"))
    durations = json.loads((data_dir / "duration.json").read_text(encoding="utf-8"))["duration"]
    train_dataset = CustomDataset(
        raw_dataset, durations=durations, preprocessed_mel=False, **mel_spec_kwargs
    )

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer.train(train_dataset, num_workers=2, resumable_with_seed=SEED)
    training_elapsed = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated() / (1024**2) if cuda_available else None
    peak_reserved = torch.cuda.max_memory_reserved() / (1024**2) if cuda_available else None

    # Strict reload check into a fresh model instance, mirroring the VITS
    # smoke test's checkpoint-integrity check.
    checkpoint_path = output_root / "model_last.pt"
    reloaded = CFM(
        transformer=DiT(**model_cfg, mel_dim=N_MEL_CHANNELS, text_num_embeds=len(vocab_char_map)),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    reloaded.load_state_dict(checkpoint["model_state_dict"], strict=True)
    total_updates = int(checkpoint["update"])
    device = "cuda" if cuda_available else "cpu"
    reloaded = reloaded.to(device)
    reloaded.eval()

    vocoder = load_vocoder()
    ref_audio_path = str(project_root / ref_row["processed_wav_path"])
    generated_samples: list[dict[str, Any]] = []
    for index, row in enumerate(synth_rows.to_dict(orient="records")):
        waveform, sample_rate, _ = infer_process(
            ref_audio_path,
            ref_row["normalized_text"],
            row["normalized_text"],
            reloaded,
            vocoder,
            mel_spec_type="vocos",
            nfe_step=16,  # fewer ODE steps than the default 32 -- smoke speed, not quality
            device=device,
        )
        audio_path = output_root / f"sample_{index}.wav"
        import soundfile as sf

        sf.write(audio_path, waveform, sample_rate)
        generated_samples.append(
            {
                "ref_text": ref_row["normalized_text"],
                "gen_text": row["normalized_text"],
                "audio_path": str(audio_path),
                "duration_seconds": len(waveform) / sample_rate,
                "sample_rate": sample_rate,
            }
        )

    result: dict[str, Any] = {
        "purpose": (
            "Ibibio Common Voice F5-TTS (flow matching) research smoke test, trained from "
            "scratch with a shrunk DiT and custom vocab; not a production voice, no "
            "naturalness/quality claim, not part of PROTOCOL.md's required order."
        ),
        "device": device,
        "torch_version": torch.__version__,
        "seed": SEED,
        "language": LANGUAGE,
        "model_cfg": model_cfg,
        "parameter_count": parameter_count,
        "train_clips": len(train_rows),
        "vocabulary_size": len(vocab_char_map),
        "batch_size": args.batch_size,
        "epochs_configured": args.epochs,
        "updates": total_updates,
        "training_elapsed_seconds": training_elapsed,
        "total_elapsed_seconds": time.perf_counter() - run_started,
        "seconds_per_update": training_elapsed / max(total_updates, 1),
        "peak_vram_allocated_mib": peak_allocated,
        "peak_vram_reserved_mib": peak_reserved,
        "checkpoint_reload": True,
        "checkpoint": str(checkpoint_path),
        "vocoder": "vocos (pretrained, frozen -- not part of the from-scratch training)",
        "reference_audio": ref_audio_path,
        "generated_samples": generated_samples,
        "raw_dataset_modified": False,
    }
    metrics_path = output_root / "metrics.json"
    metrics_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--manifest-dir", type=Path, default=Path("data/processed/common_voice_ibb/manifests")
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/processed/common_voice_ibb/f5_smoke")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/home/ubuntu/ibibio-tts/checkpoints/f5-smoke")
    )
    parser.add_argument("--clips", type=int, default=200)
    parser.add_argument("--generated-samples", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--warmup-updates", type=int, default=10)
    parser.add_argument("--save-per-updates", type=int, default=100)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
