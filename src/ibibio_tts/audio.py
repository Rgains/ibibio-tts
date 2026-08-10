"""Decode-once audio handling.

Per `PROTOCOL.md`: decode source audio once to mono, 24 kHz, signed 16-bit PCM
WAV; accept clips from 1 to 15 seconds; never overwrite raw audio or
automatically denoise it.
"""

from __future__ import annotations

import hashlib
import wave
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np

TARGET_SAMPLE_RATE = 24_000
MINIMUM_DURATION_SECONDS = 1.0
MAXIMUM_DURATION_SECONDS = 15.0


@dataclass(frozen=True)
class DecodeResult:
    decode_status: str
    source_sample_rate: int | None = None
    source_channels: int | None = None
    duration_ms: float | None = None
    error: str | None = None


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def decode_to_mono_pcm16(source: Path) -> tuple[DecodeResult, np.ndarray | None]:
    """Decode `source` once to mono int16 PCM at TARGET_SAMPLE_RATE.

    Returns the decode result plus the resampled samples (None on failure).
    Never writes anywhere or touches `source`.
    """
    try:
        container = av.open(str(source))
        stream = container.streams.audio[0]
        source_sample_rate = stream.rate
        source_channels = stream.channels
        resampler = av.AudioResampler(format="s16", layout="mono", rate=TARGET_SAMPLE_RATE)
        chunks: list[np.ndarray] = []
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray())
        for resampled in resampler.resample(None):
            chunks.append(resampled.to_ndarray())
        container.close()
    except (av.FFmpegError, OSError, IndexError) as error:
        return DecodeResult(decode_status="error", error=f"{type(error).__name__}: {error}"), None

    if not chunks:
        return DecodeResult(decode_status="error", error="no audio frames decoded"), None

    samples = np.concatenate(chunks, axis=1).reshape(-1)
    if samples.size == 0 or not np.isfinite(samples).all():
        return DecodeResult(decode_status="error", error="empty or non-finite decode"), None

    duration_ms = 1000.0 * samples.size / TARGET_SAMPLE_RATE
    return (
        DecodeResult(
            decode_status="ok",
            source_sample_rate=source_sample_rate,
            source_channels=source_channels,
            duration_ms=duration_ms,
        ),
        samples,
    )


def is_usable_duration(duration_ms: float) -> bool:
    return MINIMUM_DURATION_SECONDS * 1000 <= duration_ms <= MAXIMUM_DURATION_SECONDS * 1000


def write_wav_pcm16(path: Path, samples: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.astype("<i2").tobytes())
