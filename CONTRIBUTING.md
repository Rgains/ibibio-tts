# Project rules

- Read `PROTOCOL.md` before modifying this repository.
- This repository is for Ibibio (`ibb`) TTS.
- Do not add Fulfulde, Tiv, or another language unless explicitly requested.
- Preserve all valid Ibibio Unicode characters.
- Never transliterate text to ASCII silently.
- Never attempt to identify Common Voice speakers.
- Never commit source dataset audio.
- Never put real secrets in `.env.example`; it is a committed template and
  must only contain placeholders. Real values belong in `.env` (git-ignored).
- Use validated Common Voice clips first.
- Preserve dataset version, licence, and provenance.
- Prevent sentence and audio leakage across data splits.
- Do not begin full training before the audit report and manifests exist.
- Run tests after every meaningful implementation stage.
- Record configuration, data, and model lineage for every run.
- Do not claim production readiness without native-speaker evaluation.
- Do not add translation, warning generation, or medical advice to the TTS endpoint.
- The TTS endpoint may only synthesize supplied Ibibio text.
