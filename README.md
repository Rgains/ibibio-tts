# ibibio-tts-lab

Reproducible data preparation and research tooling for an Ibibio (`ibb`)
Common Voice multi-speaker TTS baseline.

## Quick start

```bash
UV_CACHE_DIR=.uv-cache uv sync --python 3.11
UV_CACHE_DIR=.uv-cache uv run pytest
UV_CACHE_DIR=.uv-cache uv run ruff check .
UV_CACHE_DIR=.uv-cache uv run mypy src scripts
```

Place the Common Voice archive in `data/raw/common_voice_ibb/`, then run:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/import_common_voice.py
UV_CACHE_DIR=.uv-cache uv run python scripts/audit_audio.py
UV_CACHE_DIR=.uv-cache uv run python scripts/analyse_speakers.py
UV_CACHE_DIR=.uv-cache uv run python scripts/build_manifests.py
```

Raw/interim/processed audio and model artifacts are intentionally ignored by
Git. See `PROTOCOL.md` and `CONTRIBUTING.md` before making changes.

## Licensing

Project code and original documentation are licensed under the MIT License.
The Mozilla Common Voice source dataset remains CC0-1.0 and is not included in
this repository. Model checkpoints will be licensed separately after their
dependencies and evaluation status are reviewed. See `DATA_LICENSES.md`.
