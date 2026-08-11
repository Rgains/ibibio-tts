"""Inference-time model loading and synthesis for the Streamlit demo.

Speakers are anonymous client-id hashes. This module only ever surfaces the
friendly labels from `speaker_labels.json` ("Speaker 1", ...), never the raw
hash, consistent with `PROTOCOL.md`'s "never identify speakers" rule.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_FILES = ("model.pth", "config.json", "speakers.pth", "speaker_labels.json")


@dataclass(frozen=True)
class SpeakerOption:
    label: str
    speaker_name: str
    clip_count: int


def resolve_model_dir(local_dir: Path, hf_repo_id: str | None, hf_token: str | None = None) -> Path:
    """Return a directory containing all `MODEL_FILES`.

    Prefers `local_dir` if it already has every file (fast path for local
    testing on the training box); otherwise downloads them from the given
    Hugging Face Hub model repo into `local_dir`.
    """
    if all((local_dir / name).exists() for name in MODEL_FILES):
        return local_dir
    if not hf_repo_id:
        raise FileNotFoundError(
            f"Model files not found in {local_dir} and no Hugging Face repo configured "
            "(set HF_MODEL_REPO)."
        )
    from huggingface_hub import hf_hub_download

    local_dir.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        downloaded = hf_hub_download(repo_id=hf_repo_id, filename=name, token=hf_token)
        shutil.copy(downloaded, local_dir / name)
    return local_dir


def load_speaker_options(model_dir: Path) -> list[SpeakerOption]:
    entries = json.loads((model_dir / "speaker_labels.json").read_text(encoding="utf-8"))
    return [
        SpeakerOption(
            label=entry["label"],
            speaker_name=entry["speaker_name"],
            clip_count=entry["clip_count"],
        )
        for entry in entries
    ]


def allowed_characters(model_dir: Path) -> set[str]:
    """Characters the model was trained on -- input outside this set gets
    silently dropped by the tokenizer, so the UI warns about it up front.
    """
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    chars = config.get("characters", {}) or {}
    allowed = set(chars.get("characters", "")) | set(chars.get("punctuations", ""))
    allowed.add(" ")
    return allowed


def load_synthesizer(model_dir: Path) -> Any:
    """Load the trained VITS checkpoint for CPU inference.

    Imports `TTS`/`torch` lazily so the rest of this module stays importable
    (e.g. for `load_speaker_options`) without the heavy training deps.
    """
    from TTS.api import TTS
    from TTS.tts.utils.text import cleaners as coqui_cleaners

    def ibb_character_cleaner(text: str) -> str:
        return " ".join(text.split())

    # Must match the name trained into config.json's "text_cleaner" field --
    # see scripts/train_vits_full.py's identical registration.
    coqui_cleaners.ibb_character_cleaner = ibb_character_cleaner

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    # The trained config.json has the training box's absolute speaker-file
    # paths baked in; rewrite them to wherever this copy actually lives.
    # Coqui reads these through get_from_config_or_model_args_with_default,
    # which returns config["model_args"][key] in preference to config[key], so
    # patching only the top level leaves the stale training path in effect.
    speakers_path = str(model_dir / "speakers.pth")
    for container in (config, config.get("model_args")):
        if not isinstance(container, dict):
            continue
        for key in ("speakers_file", "speaker_file"):
            if container.get(key):
                container[key] = speakers_path
    config["speakers_file"] = speakers_path
    runtime_config_path = model_dir / "config.runtime.json"
    runtime_config_path.write_text(json.dumps(config), encoding="utf-8")

    return TTS(
        model_path=str(model_dir / "model.pth"),
        config_path=str(runtime_config_path),
        gpu=False,
    )


def synthesize(tts: Any, text: str, speaker_name: str) -> tuple[list[float], int]:
    waveform = tts.tts(text=text, speaker=speaker_name)
    sample_rate = int(tts.synthesizer.output_sample_rate)
    return waveform, sample_rate
