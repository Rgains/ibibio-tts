# Lab notebook — Ibibio Common Voice multi-speaker TTS baseline

Running research log for the `ibibio-tts-lab` pipeline. Written as the work
happens; entries are append-only and dated. `PROTOCOL.md` holds the fixed
experimental protocol and `CONTRIBUTING.md` the project rules this log follows.

## Environment

| Item | Value |
|---|---|
| GPU | 1x NVIDIA L4, 23034 MiB VRAM, driver 595.84, CUDA 13.2 |
| CPU | 8 vCPU |
| RAM | 30 GiB |
| Disk (`/`) | 49 GB total, shared with sibling `tiv-tts` / `fulfulde-tts` projects |
| Python | 3.11 (uv-managed venv) |
| ffmpeg/ffprobe | 4.4.2 |

Disk is a shared, capacity-constrained resource on this host — check free
space before any download or training run and record it below.

## Dataset under study

- Common Voice Scripted Speech 26.0 — Ibibio (`ibb`), dataset ID
  `cmqiohial00zgnq07ybhhu7ni`, licence CC0-1.0, distributed via Mozilla Data
  Collective (`mozilladatacollective.com/api`).
- Archive: `common-voice-scripted-speech-26-0-ibibio-7ed511b3.tar.gz`,
  379,706,094 bytes, SHA-256 `02487b9ccc6e99aac21e2a51dd929c229d258724c83a116ac19fccf9d7af3cdb`.
- First model: multi-speaker, character-level VITS baseline with anonymous
  speaker embeddings. Not a production voice.

---

## Log

### 2026-08-10 — environment audit and disk triage

- `nvidia-smi`: 1x L4, idle, 0 MiB used — no contending job on the GPU.
- Root disk was at **100% (42 MiB free)** before any Ibibio work started,
  entirely due to sibling projects `tiv-tts` (27G) and `fulfulde-tts` (12G)
  sharing this host. 380 MB archive + extraction would not have fit.
- Confirmed no active process held those projects' venvs/caches
  (`ps aux`, `nvidia-smi --query-compute-apps`), so it was safe to reclaim
  regenerable build artifacts. With user approval, removed:
  - `tiv-tts/repo/tiv-tts/.venv` (7.8G), `tiv-tts/F5-TTS/.venv` (6.5G),
    `tiv-tts/repo/tiv-tts/.cache/uv` (7.5G),
    `fulfulde-tts/repo/fulfulde-tts/.venv` (7.4G) — all `uv sync`-regenerable.
  - Duplicate/superseded checkpoints (verified byte-identical via `md5sum`
    before deleting): `tiv-tts` `best_model_29463.pth` (dupe of
    `best_model.pth`) and 9 of 10 intermediate `checkpoint_*.pth` steps
    (kept latest, `checkpoint_103500.pth`); `fulfulde-tts`
    `best_model_24963.pth` (dupe of `best_model.pth`).
  - Net result: 42 MiB → 25 GB free. No source data or unique model weights
    were touched.
- `make sync` (uv, Python 3.11): 22 packages, ~2s.
- `scripts/validate_environment.py`: PASS on python/ffmpeg/ffprobe/disk,
  FAIL on `archive` (expected — not downloaded yet). Runtime 0.2s.
- Verified Mozilla Data Collective credentials against
  `GET /api/datasets/{id}`: response matches the pinned dataset exactly
  (name, locale `ibb`, licence CC0-1.0, checksum, size) — confirms
  `YOUR_API_KEY`/`DATASET_ID` in `.env` are correct before spending the
  download quota (org limit: 30 dataset downloads/day).

### 2026-08-10 — import, audio audit, speaker audit, manifests

- `scripts/import_common_voice.py`: download 379.7 MB in 11.6s, extract in
  3.7s, import `validated.tsv` in 0.2s (16.6s total). 3,106 rows imported,
  0 rejected. 17 anonymous speakers (`client_id`).
- Text audit found 198 rows (6.4%) flagged `unsupported_script`: Common
  Voice contributors typed Cyrillic `ә` (U+04D9 CYRILLIC SMALL LETTER
  SCHWA) instead of Latin `ə` (U+0259 LATIN SMALL LETTER SCHWA) — a
  visually-identical mixed-script substitution. Per `PROTOCOL.md` this is
  flagged for review, not silently corrected or transliterated.
- `scripts/audit_audio.py`: decoded and resampled all 3,106 clips (32 kHz
  mono source → 24 kHz mono PCM16 WAV) in 38.6s. 0 decode failures. 3,071
  clips (98.9%) usable (1-15s duration gate); 35 clips exceeded 15s and
  were excluded, none too short. 0 duplicate audio content by SHA-256.
  Usable total: 7.68 hours.
- `scripts/analyse_speakers.py`: 17/17 speakers have usable audio.
  Strongest speaker: 1.29 usable hours / 504 clips. Weakest: 0.024 hours /
  10 clips. This is a small, imbalanced multi-speaker corpus — expect the
  baseline to lean on the top few speakers.
- `scripts/build_manifests.py`: clustered the 3,071 usable rows by
  (normalized_text ∪ audio_sha256) into 996 leakage-safe groups (many
  scripted prompts are read by multiple speakers), then assigned clusters
  to splits with `random.Random(20260804)` targeting 90/5/5 by row count.
  Result: train 2,765 (90.03%, 6.92h), validation 153 (4.98%, 0.38h), test
  153 (4.98%, 0.37h). Verified zero normalized-text or audio-SHA256 overlap
  across all three splits.
- Disk after this stage: 27G used / 22G free (`data/processed` = 1.3G).

### 2026-08-10 — MMS verification (PROTOCOL.md required-order step 5, part 1)

- No dedicated `facebook/mms-tts-ibb` checkpoint is publicly available.
  Hugging Face returns HTTP 401 uniformly for both gated-and-real and
  nonexistent repos when unauthenticated, so a raw status check alone is
  not conclusive; controlled against known-real public repos
  (`facebook/mms-tts-eng`, `-yor`, `-hau` → 200) and a fabricated repo name
  (`facebook/mms-tts-zzznonexistent999` → 401), `facebook/mms-tts-ibb`
  matches the fabricated-negative pattern (401), not the real-positive one.
  `facebook/mms-tts-ibo` (Igbo) also returned 401 under the same test,
  i.e. MMS's per-language TTS coverage is not universal even among larger
  Nigerian languages.
- `configs/model/mms_baseline.yaml` already pins `checkpoint: null` and
  `require_exact_language_checkpoint: true` — per that config, no MMS
  baseline is available for this language, so the project proceeds
  straight to the VITS smoke experiment as PROTOCOL.md specifies. This is
  a documented negative result, not a blocker.

### 2026-08-10 — VITS multi-speaker smoke experiment (result)

Added `torch==2.8.0` / `torchaudio==2.8.0` / `coqui-tts==0.27.5` /
`transformers==4.57.6` as a `train` uv dependency group (kept out of the
default `sync` so the data pipeline stays light); `transformers` had to be
pinned down from the resolver's default 5.14.1 -- coqui-tts 0.27.5 imports
`transformers.pytorch_utils.isin_mps_friendly`, removed upstream.

`scripts/train_vits_smoke.py`: multi-speaker VITS (Coqui `TTS`/`trainer`),
`configs/training/smoke.yaml` (200 clips, bf16, seed 20260804, 1 rolling
checkpoint) and `configs/model/vits_multispeaker.yaml` (128-channel speaker
embeddings). Text rows carrying a review flag (e.g. the Cyrillic-schwa
substitution) were excluded from the smoke subset.

**First attempt failed at inference** (not training): `hifigan_generator`
crashed on `cond_layer(g)` with `g=None`. Root cause -- with only 200/2765
train rows and 20/153 eval rows sampled (~7%), some of the 17 speakers
had zero representation in the sampled subset, so `speaker_name_to_id`
had no embedding for a couple of the 3 test-split speakers the script
picked blindly. Fixed by restricting synthesis-sentence selection to
speakers actually present in the sampled train/eval subset (speaker
overlap across splits is expected/fine -- only sentence text and audio
content must stay disjoint, which the manifests already guarantee).

**Result** (`ibb_common_voice_smoke_multispeaker-August-10-2026_01+44AM`):

| Metric | Value |
|---|---|
| Device | NVIDIA L4 (cuda) |
| Precision | bf16, mixed precision |
| Seed | 20260804 |
| Speakers in run | 16 (of 17 total) |
| Train / eval / test clips | 200 / 20 / 3 |
| Vocabulary size | 40 characters |
| Steps | 325 (25 epochs, batch 16) |
| Parameters | 84,732,396 |
| Training time | 415.8s (1.28s/step) |
| Total wall time | 423.5s (~7m) |
| Peak VRAM | 21,275 MiB allocated / 21,918 MiB reserved (of 23,034 MiB) |
| avg_loss (discriminator+generator, epoch 24) | 33.8 (down from 56.5 at epoch 0) |
| Unknown characters (train or inference) | 0 |
| Checkpoint strict reload | passed |
| Generated samples | 3, all non-silent (RMS 0.006-0.014, no clipping) |

This confirms the full pipeline works end to end on real data: import →
audit → manifests → multi-speaker VITS training → checkpoint reload →
synthesis with per-speaker conditioning. Per `PROTOCOL.md`, this is an
**Ibibio Common Voice multi-speaker TTS research baseline**, not a
production voice -- 325 steps on 200 clips makes no naturalness or
quality claim; native-speaker evaluation has not been performed.
Peak VRAM (21.9/23 GiB reserved) leaves little headroom on this GPU for a
larger batch size or the full 2,765-clip train set without gradient
accumulation or a batch-size reduction.

Artifacts: checkpoint + samples under
`/home/ubuntu/ibibio-tts/checkpoints/vits-smoke/`, metrics copied to
`/home/ubuntu/ibibio-tts/logs/vits_smoke_metrics.json` (both outside the
git repo, matching the convention in my other TTS projects).

### 2026-08-10 — F5-TTS (flow matching) smoke experiment, optional/research

`configs/model/f5_research.yaml` is not part of PROTOCOL.md's required
order (VITS is) -- I explored it separately after the VITS
result. Added a third uv dependency group, `train-f5` (`torch==2.8.0`,
`torchaudio==2.8.0`, `f5-tts==1.1.22`), kept apart from `train` (coqui)
since the two pull incompatible dependency sets into the same venv; also
had to pin `datasets>=3.0` -- f5-tts's unpinned `datasets` requirement
resolved to 2.14.4, which calls `pyarrow.PyExtensionType`, removed from
the newer pyarrow this project's base dependencies already pull in.

**Fine-tuning the public `F5TTS_v1_Base` checkpoint was ruled out**: its
tokenizer is a fixed Chinese-pinyin/English vocab (`prepare_csv_wavs.py`
hard-copies it for any `--finetune` run) that doesn't cover Ibibio
graphemes (ọ, ị, ñ, etc.); using it would need embedding-table surgery
first. I trained from scratch with a project-built character
vocab instead -- same "prove the pipeline, no quality claim" spirit as
the VITS smoke test. Deliberately shrunk the DiT backbone from the
paper's `dim=1024, depth=22, heads=16` (~300M+ params) to `dim=256,
depth=4, heads=4, text_dim=128, conv_layers=2` (4.45M params) so a
from-scratch run finishes in under a minute; this exercises the same
CFM+DiT+Arrow-dataset+Trainer code path, not an attempt at the paper's
scale. The Vocos vocoder (mel -> waveform) is used pretrained/frozen, as
F5-TTS is designed to be used -- only the text-to-mel flow-matching model
was trained from scratch here.

**Found and worked around a real upstream bug**: `f5_tts.model.dataset
.load_dataset()`'s `"CustomDatasetPath"` branch (for pointing at a
project-local prepared dataset instead of one under the installed
package's `data/` directory) references `preprocessed_mel` before it is
ever assigned -- `UnboundLocalError` on the first run. Bypassed by
constructing `CustomDataset` directly from `Dataset.from_file(raw.arrow)`
+ `duration.json`, which is exactly what that branch does once the bug is
fixed. Not present in the VITS/Coqui stack; specific to f5-tts 1.1.22.

**Result** (`/home/ubuntu/ibibio-tts/checkpoints/f5-smoke/`):

| Metric | Value |
|---|---|
| Device | NVIDIA L4 (cuda) |
| Seed | 20260804 |
| Train clips | 200 (same seeded subsample source as VITS, disjoint reference/synthesis text from validation) |
| Vocabulary size | 40 characters |
| Model | shrunk DiT, 4,451,556 parameters |
| Updates | 250 (10 epochs, batch 8) |
| Training time | 20.8s (0.083s/update) |
| Total wall time | 23.0s |
| Peak VRAM | 1,096 MiB allocated / 1,598 MiB reserved (of 23,034 MiB) |
| Checkpoint strict reload | passed |
| Generated samples | 3, reference-audio-conditioned (voice cloning from one validation clip), all non-silent, no clipping |

Peak VRAM is a small fraction of the L4's 23 GiB -- headroom this run
didn't need because the DiT was shrunk ~70x below paper scale, not
because F5 is inherently lighter than VITS at comparable capacity.
Training was ~20x faster than the VITS smoke run (20.8s vs 415.8s) almost
entirely because of that same size difference (4.45M vs 84.7M params),
not because flow-matching trains faster than the VITS/GAN objective step
for step. This is an even earlier-stage result than the VITS baseline:
a from-scratch run this small, on a model this shrunk, makes no
naturalness, quality, or voice-similarity claim whatsoever -- it confirms
the F5 data/training/checkpoint/reference-conditioned-inference pipeline
is wired correctly for Ibibio, nothing more.

### 2026-08-10 — reviewed correction: Cyrillic-schwa prompt defect

Before spending GPU time on a full-scale run, I investigated the 198 rows
(6.4% of imports) flagged `unsupported_script`, rather than leaving them
unreviewed or blindly "fixing" them.

Evidence gathered (`data/interim/common_voice_ibb/imported.parquet`):
- All 198 flagged rows have exactly one issue: U+04D9 CYRILLIC SMALL
  LETTER SCHWA (`ә`) where U+0259 LATIN SMALL LETTER SCHWA (`ə`) was
  almost certainly intended -- visually near-identical glyphs, easy
  mistake on a Cyrillic-adjacent keyboard/IME.
- The correct Latin schwa (U+0259) **never appears anywhere** in the
  3,106-row corpus -- zero rows, flagged or not. If Cyrillic `ә` were a
  deliberate, distinct choice, some rows would plausibly use the real
  Latin schwa too. They don't.
- The 198 rows trace to only **63 unique sentence prompts**, each read by
  1-7 different speakers (mean 3.1), across 15 of 17 speakers. Common
  Voice contributors read assigned prompts; they do not type the
  `sentence` field. A defect this concentrated in a small, shared prompt
  set and spread across most speakers is a **prompt-bank text defect**
  (whoever authored those ~63 sentences made the substitution once each),
  not a per-speaker error -- there is no plausible per-speaker mechanism
  that would produce this pattern.

Decision: added a narrow, explicitly documented correction table
(`ibibio_tts.text.KNOWN_CONFUSABLE_SUBSTITUTIONS`, currently one entry)
and `apply_reviewed_corrections()`, applied by the importer as a distinct
step *after* `normalize_text()` -- not folded into `normalize_text()`
itself, which stays exactly the fixed NFC/whitespace/punctuation-spacing
contract `PROTOCOL.md` specifies. `raw_text` is untouched; the manifest
gained a `text_corrected` boolean column so every correction stays
auditable. This is a canonicalization of an input-method artifact, not
translation, paraphrase, or ASCII transliteration -- PROTOCOL.md's actual
prohibitions.

Re-ran `scripts/import_common_voice.py` (archive already local, so this
only re-imports `validated.tsv`) and `scripts/build_manifests.py`.
Result: **0 flagged rows remain** (was 198), 198 rows marked
`text_corrected=True`, identical split sizes and zero leakage (996
clusters, train 2,765 / validation 153 / test 153, same as before --
correcting the text didn't change which sentences were distinct
clusters, only their content).

**Caveat, unresolved**: this correction is evidence-based but was made
without a native Ibibio speaker confirming U+0259 is in fact correct.
Per PROTOCOL.md, no production claim is possible before native-speaker
evaluation regardless -- this doesn't change that requirement, it just
means the training data going into that eventual evaluation is cleaner
than it was.

### 2026-08-10 — full-scale VITS run: launched, stopped, mid-diagnosis

**Status: unresolved, paused mid-investigation. Nothing is currently
running (verified: no tmux session, no `train_vits*`/`diag_vits*`
processes, GPU at 0 MiB / 0% util).**

Added `scripts/train_vits_full.py` (full leakage-safe train/validation
splits, no subsampling, batch_size=12, epochs=150 -> ~34,650 steps target)
and `configs/training/full.yaml`. Launched in tmux session `vits-full`,
logging to `/home/ubuntu/ibibio-tts/logs/vits_full_run.log`.

**Problem found**: step 0 logged `step_time: 22.27s`. Assumed at first
this was one-time CUDNN kernel-autotune warmup (very common, harmless).
It was not -- confirmed via direct diagnostics (`py-spy` stack dumps
showing the main thread genuinely inside `torch.autograd.backward`,
`nvidia-smi dmon` tracing near-zero GPU SM utilization with occasional
spikes) that steps continued at **~14s/step sustained**, not just step 0.
Killed the `vits-full` tmux session/process (no checkpoint had been
reached yet, nothing lost) rather than let a possibly-multi-day run
continue on an unverified premise.

Wrote an ad hoc diagnostic harness (kept outside the repo as scratch,
not checked in) that times `Trainer.train_step` directly with
`print_step=1` for fast feedback, and bisected the difference from the
smoke run (which felt fast, ~0.4-0.5s/step by its console log lines):

- **Ruled out**: clip-duration distribution. Smoke's random 200-clip
  sample (seed 20260804) and the full 2,765-row train split have nearly
  identical duration stats (mean 9,121ms vs 9,006ms, same spread) -- not
  the cause.
- **Ruled out**: dataset pool size. Re-running the diagnostic with the
  pool capped to the same 200 rows as the smoke test, at the same
  batch_size=12 *and* at the original batch_size=16, both still showed
  ~14s/step. Not a function of total dataset size either.
- **Leading hypothesis, not yet confirmed**: `cudnn_benchmark=True`
  autotunes (times multiple algorithms and picks the fastest) for every
  *new, distinct* padded input shape it encounters, and only reuses the
  cached choice when that exact shape recurs. With variable-length audio
  and no length-bucketed sampler, shapes vary batch to batch, especially
  early in a run/epoch, so this tax gets paid repeatedly at first. The
  original smoke run's `metrics.json` reports
  `"seconds_per_step": 1.28` (415.8s / 325 steps) -- 1.28s average is
  itself dragged up by a slow start, consistent with a handful of early
  ~14-22s steps followed by much faster ones once shapes started
  repeating within the small 200-clip / 13-steps-per-epoch pool. The
  40-step diagnostic run (same 200-row pool, batch 16) was still slow
  through step 12 (`timeout` cut it off before confirming a drop-off) --
  consistent with this theory but **not yet proof**; the full 2,765-row
  pool has far more distinct clips and therefore likely far more distinct
  shapes, so this tax may not fall off the same way at full scale even if
  the theory is right for the smoke-scale case.

**Not yet done, next steps for whoever picks this up**:
1. Let the 200-row diagnostic run past ~1 full epoch (13 steps) to see
   if step_time actually drops once shapes start repeating -- confirms
   or kills the CUDNN-benchmark theory directly.
2. If confirmed: either set `cudnn_benchmark=False` (trades away the
   speedup CUDNN benchmark gives for *fixed*-shape workloads, but avoids
   repeated autotune tax on variable shapes -- likely a net win here) or
   add length-bucketed batching (`batch_group_size` exists on
   `VitsConfig` but is currently unset/default) so shapes repeat far more
   and CUDNN benchmark's cache actually pays off.
3. Re-benchmark at full scale (2,765 rows) with whichever fix, get a real
   steps/second number, *then* decide on an actual epoch/step budget and
   relaunch -- do not resume the original 150-epoch plan blind.
4. `scripts/train_vits_full.py` and `configs/training/full.yaml` are
   otherwise believed correct (same architecture/pipeline as the working
   smoke run, just unsubsampled) -- the open question is purely a
   training-speed/config issue, not a correctness one.

### 2026-08-10 — root cause confirmed and fixed: `cudnn_benchmark`

Resumed the investigation. Fixed the diagnostic harness's own bug first:
it called `trainer.train_epoch()` directly with `epochs=1` in the config,
so it could never observe more than one epoch (13 steps at the 200-row
pool) no matter how high `max_steps` was set -- changed to `epochs=10`
and `trainer.fit()` so it can run across epoch boundaries.

**Confirmed the CUDNN-benchmark theory directly.** 200-row pool,
`cudnn_benchmark=True`: epoch 0 (steps 0-12) stayed at 14-23s/step for
every single step; epoch 1 onward dropped immediately to **0.33-0.52s/step**
-- a clean ~30x speedup exactly at the epoch boundary, when previously-seen
batch shapes started recurring and CUDNN's per-shape algorithm cache
started hitting.

**Then confirmed why the full 2,765-row pool never sped up even briefly**:
same test at full scale (2,765 rows, 173 batches/epoch vs the smoke pool's
13) with `cudnn_benchmark=False` instead -- step 0 took 2.6s (minor one-time
init, not a per-shape search), and **every step from step 1 onward held at
~0.40-0.42s**, no epoch-boundary jump needed at all. With 173 distinct
batches per epoch instead of 13, the shape space is large enough that exact
repeats are rare, so `cudnn_benchmark=True`'s search cost was being paid on
nearly every single batch indefinitely, never amortizing the way it does at
smoke scale.

**Fix applied**: `cudnn_benchmark=False` in both `scripts/train_vits_full.py`
and `scripts/train_vits_smoke.py` (smoke worked "by luck" -- its small pool
happened to let CUDNN's cache saturate after one epoch, but the same trap
would reappear if the smoke subsample size ever changed, so fixed there too
for consistency). Full test/lint suite still green after the change.

**Revised full-run estimate**: 34,650 steps x ~0.42s/step (plus eval
overhead each epoch) ~ 4-4.5 hours, materially unchanged from the original
projection made before the `cudnn_benchmark` trap was discovered. Relaunching
`scripts/train_vits_full.py` next, in tmux session `vits-full` again, same as
before but with this fix.

### 2026-08-10 — full-run completed; results and known limitation

Run completed cleanly at 12:22 UTC (launched 08:08:32 UTC). Summary from
`checkpoints/vits-full/.../metrics.json`:

| metric | value |
|---|---|
| steps | 34,650 (150 epochs) |
| train / eval / test clips | 2,765 / 153 / 5 |
| speakers | 17 |
| parameters | 84,733,100 |
| seconds/step (sustained) | 0.4403 |
| training wall time | 15,255s (~4h14m) |
| peak VRAM allocated / reserved | 5,431 MiB / 6,994 MiB |
| unknown characters (train or inference) | none |
| checkpoint strict-reload | passed |

`step_time` held at 0.44s/step for the entire run -- the `cudnn_benchmark`
fix from the prior entry stood up at full scale with no regression.

**Qualitative listening result -- first native-speaker evaluation of this
project.** Samples were reviewed by a native Ibibio speaker I consulted
(I am not a native speaker myself). Verdict: prosody/
intonation sounded correct; word-level articulation was not clear --
individual words were hard to make out even though the overall contour
sounded plausibly Ibibio. This is the evaluation `PROTOCOL.md` gates any
"production" claim behind -- it has now happened once, informally, on the
full-run checkpoint, and the result is a clear no-go on intelligibility.
Not yet a systematic evaluation (no scoring rubric, no multiple listeners,
no per-sample/per-speaker breakdown) -- treat as a first data point, not a
final verdict.

**Investigated whether this is a bug or an expected ceiling, found evidence
for the latter**:

1. **Eval mel-loss plateaued, did not improve with more training.**
   `avg_loss_mel` oscillated in the 22.8-25.4 range from roughly step 21,600
   onward with no further monotonic improvement -- confirmed the *best*
   checkpoint was saved at step 25,641 (11:16 UTC), and training continued
   another ~9,000 steps / ~1 hour past that with zero eval gain by the time
   it finished at step 34,650 (12:22 UTC). This rules out "just needs more
   steps on this data" -- the run already trained well past the point where
   eval loss stopped responding.

2. **No phonemizer/G2P exists for Ibibio.** Checked `espeak-ng --voices`:
   no `ibb` entry. Per `PROTOCOL.md`'s fixed normalization contract, the
   model is trained on raw NFC characters, not phonemes -- it has to learn
   grapheme-to-sound mapping implicitly from audio alone, with no
   phonological prior to lean on. VITS's duration predictor (rhythm/stress,
   a comparatively low-dimensional target) typically converges well before
   the flow-based decoder's fine spectral detail does; with only 7.68h of
   audio and no G2P shortcut, prosody looking right before word-level
   clarity does is the expected order of convergence, not a training bug.

3. **Severe per-speaker clip imbalance.** 17 speakers, clip counts range
   10-462 (median 148, mean 162.6, std 157.6) -- computed from
   `manifests/train.parquet` after flag filtering. Several speakers in the
   generated sample set have very little data (13 and 18 clips respectively
   for two of the five synthesis speakers), nowhere near enough for the
   shared model to learn crisp per-speaker articulation. This plausibly
   compounds (1) and (2) rather than being an independent cause.

**Conclusion**: this is a data ceiling, not a fixable training-recipe bug.
The dataset is fixed at 7.68h total (no more Common Voice Ibibio audio
available) and no Ibibio phonemizer is known to exist, so neither of the
two levers that would most directly improve word-level clarity (more
per-speaker data, phoneme-level input) is available within this project's
current constraints. Logging this as the honest result of a from-scratch,
character-input, imbalanced-multi-speaker VITS baseline on a small,
research-scale dataset. `PROTOCOL.md` gates any production claim behind
native-speaker evaluation -- that evaluation has now happened (informally,
see above) and the result is negative on intelligibility, so this remains
a research baseline, now with direct native-speaker evidence rather than
just a data-budget argument for why. No further training runs planned
against this same dataset; next real improvements would require either
new Ibibio speech data or a usable G2P resource, neither in scope right
now.
