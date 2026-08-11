"""Streamlit demo for the Ibibio Common Voice multi-speaker VITS baseline.

Research demo, not a production voice -- see the banner below and
`PROTOCOL.md` / `papers.md` in this repo for why. Loads model weights from
`IBB_TTS_LOCAL_MODEL_DIR` if present (fast path when running on the training
box), otherwise downloads them from the Hugging Face Hub repo named by
`HF_MODEL_REPO` (set via Streamlit secrets or an env var).
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st

# Prefer this checkout's src/ over any copy already installed into
# site-packages: a host that pulls new code without reinstalling would
# otherwise keep running a stale build of ibibio_tts.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ibibio_tts.inference import (  # noqa: E402
    allowed_characters,
    load_speaker_options,
    load_synthesizer,
    resolve_model_dir,
    synthesize,
)

st.set_page_config(page_title="Ibibio TTS -- research demo", page_icon="\U0001f5e3️")


def _config(name: str, default: str = "") -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except FileNotFoundError:
        pass
    return os.environ.get(name, default)


LOCAL_MODEL_DIR = Path(_config("IBB_TTS_LOCAL_MODEL_DIR", "model_export"))
HF_MODEL_REPO = _config("HF_MODEL_REPO") or None
HF_TOKEN = _config("HF_TOKEN") or None


@st.cache_resource(show_spinner="Loading model...")
def _load_model_dir() -> Path:
    return resolve_model_dir(LOCAL_MODEL_DIR, HF_MODEL_REPO, HF_TOKEN)


@st.cache_resource(show_spinner="Loading Ibibio VITS checkpoint (CPU inference)...")
def _load_tts(model_dir_str: str):
    return load_synthesizer(Path(model_dir_str))


st.title("Ibibio Common Voice TTS -- research demo")

st.warning(
    "**Research baseline, not a production voice.** Trained from scratch on "
    "7.68 hours of Mozilla Common Voice Ibibio audio (17 speakers, no "
    "phonemizer available for Ibibio). A native Ibibio speaker evaluated "
    "this model's output and found intonation plausible but individual "
    "words often unclear. See `papers.md` in the project repo for the full "
    "investigation. Speaker identities are anonymized -- selectable "
    "speakers are labeled by training-data volume only, never by the "
    "original Common Voice speaker ID.",
    icon="⚠️",
)

try:
    model_dir = _load_model_dir()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

speakers = load_speaker_options(model_dir)
vocab = allowed_characters(model_dir)

with st.form("synthesize"):
    speaker_choice = st.selectbox(
        "Speaker",
        options=speakers,
        format_func=lambda opt: f"{opt.label} ({opt.clip_count} training clips)",
    )
    text = st.text_area(
        "Ibibio text",
        value="Idịp ete ke imaidia se idiimaaha ke esie inaanam uyom nte esop mbakara.",
        height=100,
    )
    submitted = st.form_submit_button("Synthesize")

if submitted:
    unsupported = sorted({ch for ch in text if ch not in vocab and not ch.isspace()})
    if unsupported:
        st.info(
            "These characters were not in the training vocabulary and will "
            f"be dropped by the model: {' '.join(unsupported)}"
        )
    if not text.strip():
        st.error("Enter some Ibibio text first.")
    else:
        tts = _load_tts(str(model_dir))
        with st.spinner("Synthesizing..."):
            waveform, sample_rate = synthesize(tts, text, speaker_choice.speaker_name)
        buffer = io.BytesIO()
        sf.write(buffer, np.asarray(waveform, dtype=np.float32), sample_rate, format="WAV")
        st.audio(buffer.getvalue(), format="audio/wav")
        st.download_button(
            "Download WAV",
            data=buffer.getvalue(),
            file_name="ibibio_tts_sample.wav",
            mime="audio/wav",
        )

st.caption(
    "Source: Mozilla Common Voice Scripted Speech 26.0, Ibibio (CC0-1.0). "
    "Model: VITS multi-speaker, trained from scratch, character-level input."
)
