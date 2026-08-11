# ibibio-tts-lab

Reproducible data preparation and research tooling for an Ibibio (`ibb`)
Common Voice multi-speaker TTS baseline.

## Quick start

```bash
UV_CACHE_DIR=.uv-cache uv sync --python 3.11
UV_CACHE_DIR=.uv-cache uv run python scripts/validate_environment.py
UV_CACHE_DIR=.uv-cache uv run pytest
UV_CACHE_DIR=.uv-cache uv run ruff check .
UV_CACHE_DIR=.uv-cache uv run mypy src scripts
```

The validator checks the supported Python version, FFmpeg tools, free disk
space, and (when present) the Common Voice archive checksum. Run it before
downloading or importing data. A non-zero exit means the environment is not
ready for the next pipeline stage.

## Configuration

Copy `.env.example` to `.env` and fill in real values; `.env` is git-ignored
and must never be committed. `.env.example` must only ever hold placeholders.

- `YOUR_API_KEY` — Mozilla Data Collective API key used to fetch the Common
  Voice dataset archive.
- `DATASET_ID` — Common Voice Scripted Speech 26.0 (Ibibio) dataset ID,
  `cmqiohial00zgnq07ybhhu7ni`. Leave this at the pinned value; see
  `PROTOCOL.md` before changing it.

With `.env` configured, run the pipeline in order (`import_common_voice.py`
downloads the archive itself via the Mozilla Data Collective API, verifies
its checksum against `reports/common_voice_ibb_archive.sha256`, and extracts
it into `data/raw/common_voice_ibb/` — no manual placement needed):

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/import_common_voice.py
UV_CACHE_DIR=.uv-cache uv run python scripts/audit_audio.py
UV_CACHE_DIR=.uv-cache uv run python scripts/analyse_speakers.py
UV_CACHE_DIR=.uv-cache uv run python scripts/build_manifests.py
```

This produces, in order: an import manifest with every rejection tracked
(`data/interim/.../imported.parquet`), decoded 24 kHz mono PCM16 WAV clips
plus an audio audit (`reports/audio_audit.*`), an anonymous per-speaker
audit (`reports/speaker_audit.*`), and leakage-safe 90/5/5 train/validation/
test manifests with a leakage check (`reports/manifest_summary.json`).

## Training

The first model is a multi-speaker VITS baseline (`PROTOCOL.md`'s required
order). Training deps are a separate `uv` dependency group, kept out of the
default `sync` so the data pipeline stays light:

```bash
UV_CACHE_DIR=.uv-cache uv sync --python 3.11 --group train
UV_CACHE_DIR=.uv-cache uv run python scripts/train_vits_smoke.py
```

This is a smoke experiment on a small clip subset — it proves the pipeline
works end to end (data loading, BF16 training, checkpoint save/strict-reload,
per-speaker synthesis) and makes no naturalness or quality claim. See
`papers.md` for run results and hardware notes.

`scripts/train_vits_full.py` runs the same architecture on the complete
leakage-safe train/validation splits (150 epochs, ~34.6k steps, ~4.2h on a
single GPU). Its result is still a **research baseline**: a native Ibibio
speaker who evaluated its output found prosody plausible but individual
words often unclear, consistent with training from scratch on 7.68h total
audio with no Ibibio phonemizer available. Full investigation in `papers.md`.

An optional, non-required F5-TTS (flow matching) research experiment also
exists (`configs/model/f5_research.yaml`), trained from scratch with a
project-built character vocab rather than fine-tuned from the public
`F5TTS_v1_Base` checkpoint (that checkpoint's vocab doesn't cover Ibibio
graphemes):

```bash
UV_CACHE_DIR=.uv-cache uv sync --python 3.11 --group train-f5
UV_CACHE_DIR=.uv-cache uv run python scripts/train_f5_smoke.py
```

Raw/interim/processed audio and model artifacts are intentionally ignored by
Git; trained checkpoints live outside the repo entirely
(`/home/ubuntu/ibibio-tts/checkpoints/`, `/home/ubuntu/ibibio-tts/logs/`),
matching my other TTS projects. See `PROTOCOL.md` and `CONTRIBUTING.md`
before making changes.

## Demo

`app.py` is a Streamlit demo (text in, synthesized speech out, anonymized
speaker picker) built on the full-run VITS checkpoint. It carries the same
research-baseline disclaimer as everywhere else in this project — the demo
is for listening to and understanding the current state of the model, not
a claim that it's production-ready.

Run it locally against a checkpoint already on this machine:

```bash
UV_CACHE_DIR=.uv-cache uv sync --python 3.11 --group train --group demo
UV_CACHE_DIR=.uv-cache uv run python scripts/export_model_for_demo.py \
  --run-dir /home/ubuntu/ibibio-tts/checkpoints/vits-full/<run-name>
UV_CACHE_DIR=.uv-cache uv run streamlit run app.py
```

`export_model_for_demo.py` flattens a timestamped training run into
`model_export/` (`model.pth`, `config.json`, `speakers.pth`,
`speaker_labels.json`) — the exact bundle `app.py` loads, and what gets
pushed to Hugging Face for deployment elsewhere (`model_export/` is
git-ignored; it does not belong in this repo).

To deploy on Streamlit Community Cloud (which has no GPU and doesn't share
this machine's filesystem), push the exported bundle to a Hugging Face Hub
model repo first:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/push_model_to_hf.py \
  --repo-id <your-hf-username>/ibibio-tts-vits --token <hf-write-token>
```

Then on Streamlit Cloud, point the app at `app.py` and set these secrets
(Settings → Secrets):

```toml
HF_MODEL_REPO = "<your-hf-username>/ibibio-tts-vits"
# HF_TOKEN only needed if the HF repo is private
```

`app.py` downloads the model from `HF_MODEL_REPO` on first load (cached
after that) whenever `model_export/` isn't already present locally — the
same code path works unchanged on this GPU box, a Mac, or Streamlit Cloud.

## Licensing

Project code and original documentation are licensed under the MIT License.
The Mozilla Common Voice source dataset remains CC0-1.0 and is not included in
this repository. Model checkpoints will be licensed separately after their
dependencies and evaluation status are reviewed. See `DATA_LICENSES.md`.
