# Ibibio TTS project context

`ibibio-tts-lab` builds a reproducible Ibibio (`ibb`) text-to-speech research
pipeline from Mozilla Common Voice Scripted Speech 26.0. The first model is a
multi-speaker, character-level VITS baseline with anonymous speaker embeddings;
target-speaker adaptation is optional and depends on the audit.

## Dataset

- Dataset ID: `cmqiohial00zgnq07ybhhu7ni`
- Release: Common Voice Scripted Speech 26.0, Ibibio
- Locale: `ibb`
- Licence: CC0-1.0
- Expected archive: `common-voice-scripted-speech-26-0-ibibio-7ed511b3.tar.gz`
- Use only `validated.tsv` for the first baseline.
- Never identify speakers or redistribute source audio. Retain anonymous client IDs.

## Data rules

Normalize text with Unicode NFC, collapse repeated whitespace, and apply only
safe punctuation spacing. Preserve raw and normalized text. Never translate,
paraphrase, ASCII-transliterate, or silently remove unknown characters. Flag
numbers, dates, abbreviations, and unsupported scripts for review.

Decode source audio once to mono, 24 kHz, signed 16-bit PCM WAV. Accept clips
from 1 to 15 seconds. Do not overwrite raw audio or automatically denoise it.

Split by sentence identity/normalized text, never by row. The main test set is
sentence-disjoint. Audio hashes must also be disjoint. Use ratios 90/5/5 and
seed `20260804`.

## Required order

1. Validate the environment.
2. Import validated data and track every rejection.
3. Produce audio, text, and anonymous-speaker audits.
4. Produce leakage-safe manifests.
5. Only then verify MMS and run a small VITS smoke experiment.

The first result must be described as an **Ibibio Common Voice multi-speaker TTS
research baseline**, not a production voice.
