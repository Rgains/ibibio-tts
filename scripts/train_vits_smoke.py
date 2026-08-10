#!/usr/bin/env python3
"""VITS multi-speaker GPU smoke experiment for Ibibio Common Voice.

Per `PROTOCOL.md`: this is the first VITS smoke experiment, run only after
the environment, import, audits, and manifests exist. It exercises real
multi-speaker conditioning (anonymous Common Voice `client_id` speaker
embeddings) end to end -- data loading, BF16 training, checkpoint
save/strict-reload, and synthesis -- and captures run metadata (seed,
precision, peak VRAM, step time, generated samples). It does not claim
anything about voice quality, naturalness, or production readiness; see
`PROTOCOL.md`: "The first result must be described as an Ibibio Common
Voice multi-speaker TTS research baseline, not a production voice."
"""

from __future__ import annotations

import argparse
import json
import random
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models import vits as vits_module
from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig
from TTS.tts.utils.text import cleaners as coqui_cleaners
from TTS.vocoder.utils.generic_utils import plot_results as _plot_results

SEED = 20260804
LANGUAGE = "ibb"


def ibb_character_cleaner(text: str) -> str:
    """No-op beyond whitespace collapse -- text.py already NFC-normalized it."""
    return " ".join(text.split())


coqui_cleaners.ibb_character_cleaner = ibb_character_cleaner  # type: ignore[attr-defined]


def _bf16_safe_create_logs(self: Any, batch: Any, outputs: Any) -> Any:
    """Coqui's Vits._create_logs calls .numpy() on raw BF16 tensors for
    tensorboard previews; numpy has no bfloat16 dtype. Cast to float32
    first. Only affects preview images/audio, not training math.
    """
    y_hat = outputs[1]["model_outputs"].float()
    y = outputs[1]["waveform_seg"].float()
    figures = _plot_results(y_hat, y, self.ap)
    sample_voice = y_hat[0].squeeze(0).detach().cpu().numpy()
    audios = {"audio": sample_voice}
    alignments = outputs[1]["alignments"]
    align_img = alignments[0].float().data.cpu().numpy().T
    figures.update({"alignment": vits_module.plot_alignment(align_img, output_fig=False)})
    return figures, audios


vits_module.Vits._create_logs = _bf16_safe_create_logs


def character_config(vocab: set[str]) -> CharactersConfig:
    """Split the observed vocabulary into letters/marks vs punctuation by
    Unicode general category, so Ibibio graphemes (including combining
    tone marks) are treated as characters and everything else as
    punctuation -- no fixed, language-specific punctuation list needed.
    """
    letters = "".join(sorted(c for c in vocab if unicodedata.category(c)[0] in ("L", "M")))
    punctuation = "".join(sorted(c for c in vocab if unicodedata.category(c)[0] not in ("L", "M")))
    return CharactersConfig(
        pad="<PAD>",
        eos=None,
        bos=None,
        blank="<BLNK>",
        characters=letters,
        punctuations=punctuation,
        is_unique=False,
        is_sorted=False,
    )


def load_split(manifest_dir: Path, split: str) -> pd.DataFrame:
    frame = pd.read_parquet(manifest_dir / f"{split}.parquet")
    return frame[frame["flags"] == ""].reset_index(drop=True)


def to_samples(project_root: Path, frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "text": row["normalized_text"],
            "audio_file": str(project_root / row["processed_wav_path"]),
            "audio_unique_name": row["clip_path"],
            "speaker_name": row["client_id"],
            "language": LANGUAGE,
            "root_path": "",
        }
        for row in frame.to_dict(orient="records")
    ]


def save_waveform(path: Path, waveform: torch.Tensor, sample_rate: int) -> tuple[float, int]:
    import soundfile as sf

    array = waveform.detach().cpu().squeeze().numpy().astype(np.float32)
    peak = float(np.max(np.abs(array))) if array.size else 0.0
    if peak > 0.98:
        array *= 0.98 / peak
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, array, sample_rate)
    info = sf.info(path)
    return info.duration, info.samplerate


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_started = time.perf_counter()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = random.Random(SEED)

    project_root = args.project_root.resolve()
    manifest_dir = project_root / args.manifest_dir

    train_all = load_split(manifest_dir, "train")
    eval_all = load_split(manifest_dir, "validation")
    test_all = load_split(manifest_dir, "test")

    train_indices = sorted(rng.sample(range(len(train_all)), min(args.clips, len(train_all))))
    eval_indices = sorted(rng.sample(range(len(eval_all)), min(args.eval_clips, len(eval_all))))
    train_rows = train_all.iloc[train_indices].reset_index(drop=True)
    eval_rows = eval_all.iloc[eval_indices].reset_index(drop=True)

    speakers = sorted(set(train_rows["client_id"]) | set(eval_rows["client_id"]))
    # Synthesis needs a speaker embedding the model actually has, i.e. a
    # speaker present in this run's sampled train/eval subset -- not just
    # anywhere in the (speaker-overlapping-by-design) test split.
    test_candidates = test_all[test_all["client_id"].isin(speakers)]
    if test_candidates.empty:
        test_candidates = test_all
    test_rows = test_candidates.iloc[: args.generated_samples].reset_index(drop=True)
    vocab: set[str] = set()
    for frame in (train_rows, eval_rows, test_rows):
        for text in frame["normalized_text"]:
            vocab.update(text)

    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_dir = args.log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    config = VitsConfig(
        output_path=str(output_root),
        run_name=args.run_name,
        run_description=(
            "Ibibio Common Voice multi-speaker TTS research baseline smoke test -- "
            "not a production voice; no naturalness or quality claim."
        ),
        dashboard_logger="tensorboard",
        audio=VitsAudioConfig(
            sample_rate=24_000,
            fft_size=1_024,
            win_length=1_024,
            hop_length=256,
            num_mels=80,
            mel_fmin=0,
            mel_fmax=8_000,
        ),
        characters=character_config(vocab),
        use_phonemes=False,
        text_cleaner="ibb_character_cleaner",
        add_blank=True,
        enable_eos_bos_chars=False,
        # get_from_config_or_model_args_with_default() checks model_args first,
        # so use_speaker_embedding must be set there, not only top-level.
        model_args=VitsArgs(use_speaker_embedding=True, speaker_embedding_channels=128),
        use_speaker_embedding=True,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_loader_workers=2,
        num_eval_loader_workers=2,
        epochs=args.epochs,
        run_eval=True,
        test_delay_epochs=999,
        print_step=10,
        plot_step=100_000,
        save_step=args.save_step,
        save_n_checkpoints=1,
        save_all_best=False,
        log_model_step=1_000_000,
        save_checkpoints=True,
        model_param_stats=False,
        mixed_precision=True,
        precision="bf16",
        training_seed=SEED,
        test_sentences=[],
        # See scripts/train_vits_full.py and papers.md: cudnn_benchmark only
        # pays off when padded shapes repeat, which is unreliable with
        # variable-length audio and no length bucketing.
        cudnn_benchmark=False,
        cudnn_deterministic=False,
    )

    train_samples = to_samples(project_root, train_rows)
    eval_samples = to_samples(project_root, eval_rows)
    model = Vits.init_from_config(config, samples=train_samples + eval_samples)
    if model.speaker_manager is None:
        raise RuntimeError("Multi-speaker model initialized without a speaker manager")
    speaker_name_to_id = dict(model.speaker_manager.name_to_id)

    unknown_before: set[str] = set()
    test_samples = to_samples(project_root, test_rows)
    for sample in train_samples + eval_samples + test_samples:
        model.tokenizer.text_to_ids(sample["text"], language=sample["language"])
        unknown_before.update(model.tokenizer.not_found_characters)
    if unknown_before:
        raise RuntimeError(f"Unknown characters found before training: {sorted(unknown_before)}")

    parameter_count = sum(p.numel() for p in model.parameters())
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    trainer = Trainer(
        TrainerArgs(),
        config,
        str(output_root),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        parse_command_line_args=False,
    )
    device = "cuda" if trainer.use_cuda else "cpu"
    trainer.fit()
    training_elapsed = time.perf_counter() - started
    trainer.save_checkpoint()
    checkpoint_path = max(
        Path(trainer.output_path).glob("checkpoint_*.pth"),
        key=lambda path: path.stat().st_mtime,
    )
    peak_allocated = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else None
    peak_reserved = torch.cuda.max_memory_reserved() / (1024**2) if device == "cuda" else None

    reloaded = Vits.init_from_config(config, samples=train_samples + eval_samples)
    reloaded.load_checkpoint(config, checkpoint_path, eval=True, strict=True)
    if reloaded.speaker_manager is None:
        reloaded.speaker_manager = model.speaker_manager
    if device == "cuda":
        reloaded = reloaded.to("cuda")

    generated_samples: list[dict[str, Any]] = []
    unknown_at_inference: set[str] = set()
    for index, row in enumerate(test_rows.to_dict(orient="records")):
        speaker_id = speaker_name_to_id.get(row["client_id"])
        if speaker_id is None:
            # This model has no embedding for this speaker (not in the
            # sampled train/eval subset); skip rather than synthesize with
            # an undefined speaker embedding.
            continue
        ids = reloaded.tokenizer.text_to_ids(row["normalized_text"], language=LANGUAGE)
        unknown_at_inference.update(reloaded.tokenizer.not_found_characters)
        tokens = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        speaker_ids = torch.tensor([speaker_id], dtype=torch.long)
        aux_input: dict[str, torch.Tensor] = {"speaker_ids": speaker_ids}
        if device == "cuda":
            tokens = tokens.to("cuda")
            aux_input = {key: value.to("cuda") for key, value in aux_input.items()}
        with torch.inference_mode():
            generated = reloaded.inference(tokens, aux_input=aux_input)["model_outputs"]
        audio_path = Path(trainer.output_path) / f"sample_{index}.wav"
        duration, sample_rate = save_waveform(audio_path, generated, config.audio.sample_rate)
        generated_samples.append(
            {
                "text": row["normalized_text"],
                "speaker_id": row["client_id"],
                "in_training_speaker_set": speaker_id is not None,
                "audio_path": str(audio_path),
                "duration_seconds": duration,
                "sample_rate": sample_rate,
            }
        )

    result: dict[str, Any] = {
        "purpose": (
            "Ibibio Common Voice multi-speaker TTS research baseline smoke test; "
            "not a production voice, no naturalness/quality claim."
        ),
        "device": device,
        "torch_version": torch.__version__,
        "precision": config.precision,
        "mixed_precision": config.mixed_precision,
        "seed": SEED,
        "language": LANGUAGE,
        "num_speakers": len(speakers),
        "batch_size": args.batch_size,
        "epochs_configured": args.epochs,
        "steps": trainer.total_steps_done,
        "parameter_count": parameter_count,
        "train_clips": len(train_samples),
        "eval_clips": len(eval_samples),
        "test_clips": len(test_rows),
        "vocabulary_size": len(vocab),
        "training_elapsed_seconds": training_elapsed,
        "total_elapsed_seconds": time.perf_counter() - run_started,
        "seconds_per_step": training_elapsed / max(trainer.total_steps_done, 1),
        "peak_vram_allocated_mib": peak_allocated,
        "peak_vram_reserved_mib": peak_reserved,
        "unknown_characters_before_training": sorted(unknown_before),
        "unknown_characters_at_inference": sorted(unknown_at_inference),
        "checkpoint_reload": True,
        "checkpoint": str(checkpoint_path),
        "generated_samples": generated_samples,
        "raw_dataset_modified": False,
        "mms_baseline_available": False,
    }
    if unknown_at_inference:
        raise RuntimeError(f"Unknown characters at inference: {sorted(unknown_at_inference)}")

    metrics_path = Path(trainer.output_path) / "metrics.json"
    metrics_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    import shutil

    shutil.copy(metrics_path, log_dir / args.metrics_name)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--manifest-dir", type=Path, default=Path("data/processed/common_voice_ibb/manifests")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/home/ubuntu/ibibio-tts/checkpoints/vits-smoke")
    )
    parser.add_argument("--log-dir", type=Path, default=Path("/home/ubuntu/ibibio-tts/logs"))
    parser.add_argument("--clips", type=int, default=200)
    parser.add_argument("--eval-clips", type=int, default=20)
    parser.add_argument("--generated-samples", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--save-step", type=int, default=100)
    parser.add_argument("--metrics-name", default="vits_smoke_metrics.json")
    parser.add_argument("--run-name", default="ibb_common_voice_smoke_multispeaker")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
