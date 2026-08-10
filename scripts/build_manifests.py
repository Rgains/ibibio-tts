#!/usr/bin/env python3
"""Leakage-safe train/validation/test manifests.

Per `PROTOCOL.md`: split by sentence identity/normalized text, never by row;
the main test set is sentence-disjoint; audio hashes must also be disjoint;
use ratios 90/5/5 and seed 20260804.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import pandas as pd

IMPORTED_IN = Path("data/interim/common_voice_ibb/imported.parquet")
AUDITED_IN = Path("data/interim/common_voice_ibb/audited.parquet")
MANIFEST_DIR = Path("data/processed/common_voice_ibb/manifests")
REPORT_JSON = Path("reports/manifest_summary.json")

SEED = 20260804
FRACTIONS = {"train": 0.90, "validation": 0.05, "test": 0.05}
MANIFEST_COLUMNS = (
    "client_id",
    "clip_path",
    "processed_wav_path",
    "raw_text",
    "normalized_text",
    "flags",
    "duration_ms",
    "split",
)


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, key: str) -> str:
        self.parent.setdefault(key, key)
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, a: str, b: str) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[root_a] = root_b


def _cluster_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group rows so no normalized_text or audio_sha256 crosses a split."""
    uf = UnionFind()
    for row in rows:
        text_key = f"text:{row['normalized_text']}"
        hash_key = f"hash:{row['audio_sha256']}"
        uf.find(text_key)
        uf.find(hash_key)
        uf.union(text_key, hash_key)

    row_root: dict[int, str] = {}
    for index, row in enumerate(rows):
        text_key = f"text:{row['normalized_text']}"
        row_root[index] = uf.find(text_key)

    clusters: dict[str, list[dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        clusters.setdefault(row_root[index], []).append(row)
    return clusters


def _assign_splits(clusters: dict[str, list[dict[str, Any]]], seed: int) -> dict[str, str]:
    """Deterministically assign each cluster to a split, targeting row-count fractions."""
    total_rows = sum(len(rows) for rows in clusters.values())
    targets = {name: fraction * total_rows for name, fraction in FRACTIONS.items()}

    ordered_keys = sorted(clusters.keys())
    random.Random(seed).shuffle(ordered_keys)

    assignment: dict[str, str] = {}
    allocated = {"test": 0, "validation": 0, "train": 0}
    for key in ordered_keys:
        size = len(clusters[key])
        for name in ("test", "validation"):
            if allocated[name] + size <= targets[name] or allocated[name] == 0:
                assignment[key] = name
                allocated[name] += size
                break
        else:
            assignment[key] = "train"
            allocated["train"] += size
    return assignment


def build(project_root: Path) -> dict[str, Any]:
    imported = pd.read_parquet(project_root / IMPORTED_IN)
    audited = pd.read_parquet(project_root / AUDITED_IN)
    merged = imported.merge(audited, on="clip_path", how="inner")
    usable = merged[merged["usable"] == True].reset_index(drop=True)  # noqa: E712

    rows = usable.to_dict(orient="records")
    clusters = _cluster_rows(rows)
    assignment = _assign_splits(clusters, SEED)

    manifest_rows: list[dict[str, Any]] = []
    for cluster_key, cluster_rows in clusters.items():
        split = assignment[cluster_key]
        for row in cluster_rows:
            manifest_rows.append(
                {
                    "client_id": row["client_id"],
                    "clip_path": row["clip_path"],
                    "processed_wav_path": row["processed_wav_path"],
                    "raw_text": row["raw_text"],
                    "normalized_text": row["normalized_text"],
                    "flags": row["flags"],
                    "duration_ms": row["duration_ms"],
                    "split": split,
                }
            )

    manifest = pd.DataFrame.from_records(manifest_rows, columns=list(MANIFEST_COLUMNS))
    manifest_dir = project_root / MANIFEST_DIR
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for split_name in FRACTIONS:
        split_frame = manifest[manifest["split"] == split_name]
        split_frame.to_parquet(manifest_dir / f"{split_name}.parquet", index=False)

    text_by_split: dict[str, set[str]] = {}
    hash_by_split: dict[str, set[str]] = {}
    for split_name in FRACTIONS:
        split_frame = manifest[manifest["split"] == split_name]
        text_by_split[split_name] = set(split_frame["normalized_text"])
        hash_by_split[split_name] = set(
            usable[usable["clip_path"].isin(split_frame["clip_path"])]["audio_sha256"]
        )

    leaked_text_pairs = {
        f"{a}_{b}": len(text_by_split[a] & text_by_split[b])
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    leaked_hash_pairs = {
        f"{a}_{b}": len(hash_by_split[a] & hash_by_split[b])
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    if any(leaked_text_pairs.values()) or any(leaked_hash_pairs.values()):
        raise AssertionError(
            f"Manifest leakage detected: text={leaked_text_pairs}, audio={leaked_hash_pairs}"
        )

    summary: dict[str, Any] = {
        "seed": SEED,
        "fractions": FRACTIONS,
        "split_unit": "cluster of rows sharing normalized_text or audio_sha256",
        "usable_rows_considered": len(usable),
        "clusters": len(clusters),
        "splits": {
            split_name: {
                "rows": int((manifest["split"] == split_name).sum()),
                "hours": round(
                    manifest.loc[manifest["split"] == split_name, "duration_ms"].sum()
                    / 3_600_000,
                    6,
                ),
                "speakers": int(
                    manifest.loc[manifest["split"] == split_name, "client_id"].nunique()
                ),
                "unique_sentences": len(text_by_split[split_name]),
            }
            for split_name in FRACTIONS
        },
        "leakage_checks": {
            "normalized_text_overlap": leaked_text_pairs,
            "audio_sha256_overlap": leaked_hash_pairs,
        },
    }
    (project_root / REPORT_JSON).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    started = time.monotonic()
    summary = build(project_root)
    elapsed = time.monotonic() - started
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[build_manifests] finished in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
