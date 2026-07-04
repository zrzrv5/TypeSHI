"""Shared preprocessing: Snapshot -> feature record -> bucketed npz shards.

Records are bucketed by T (number of types) so every shard array has a fixed
shape; padding across T happens only at collate time.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from .augment import MAX_TYPES, augment
from .descriptors import K_ENV, M_ENV, N_BINS, N_GLOB, N_PAIR_EXTRA, compute_features
from .io import Snapshot


def group_id(symbols: tuple[str, ...]) -> int:
    """Stable hash of the sorted element set -> train/val split key.

    Split on element-set (not formula) so e.g. all Li-P-S structures land on
    one side; prevents near-duplicate leakage across datasets.
    """
    key = ",".join(sorted(symbols))
    return int(hashlib.md5(key.encode()).hexdigest()[:12], 16)


def make_records(snap: Snapshot, labels_z: np.ndarray, gid: int,
                 rng: np.random.Generator, n_aug: int) -> list[dict]:
    """Feature records for one structure: 1 clean + n_aug augmented variants."""
    records = []
    variants = [(snap, labels_z)]
    for _ in range(n_aug):
        variants.append(augment(snap, labels_z, rng))
    for s, z in variants:
        if s.n_types > MAX_TYPES:
            continue
        try:
            f = compute_features(s, with_env=True, rng=rng)
        except Exception:
            continue
        records.append({**f, "labels": z.astype(np.int8), "gid": gid})
    return records


class ShardWriter:
    """Accumulate records bucketed by T; flush fixed-shape npz shards."""

    def __init__(self, out_dir: str | Path, prefix: str, shard_size: int = 8192):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.shard_size = shard_size
        self.buckets: dict[int, list[dict]] = defaultdict(list)
        self.n_shards: dict[int, int] = defaultdict(int)
        self.total = 0

    def add(self, rec: dict):
        t = len(rec["labels"])
        self.buckets[t].append(rec)
        self.total += 1
        if len(self.buckets[t]) >= self.shard_size:
            self._flush(t)

    def _flush(self, t: int):
        recs = self.buckets.pop(t, [])
        if not recs:
            return
        path = self.out_dir / f"{self.prefix}_T{t}_{self.n_shards[t]:05d}.npz"
        arrays = dict(
            rdf=np.stack([r["rdf"] for r in recs]).astype(np.float16),
            pair_extra=np.stack([r["pair_extra"] for r in recs]).astype(np.float16),
            frac=np.stack([r["frac"] for r in recs]).astype(np.float16),
            glob=np.stack([r["glob"] for r in recs]).astype(np.float32),
            labels=np.stack([r["labels"] for r in recs]),
            gid=np.array([r["gid"] for r in recs], dtype=np.uint64),
        )
        if "env_d" in recs[0]:                      # v3 shards
            arrays["env_d"] = np.stack([r["env_d"] for r in recs]).astype(np.float16)
            arrays["env_t"] = np.stack([r["env_t"] for r in recs])  # int8, -1 pad
        np.savez_compressed(path, **arrays)
        self.n_shards[t] += 1

    def close(self):
        for t in list(self.buckets):
            self._flush(t)


FEATURE_DIMS = dict(n_bins=N_BINS, n_pair_extra=N_PAIR_EXTRA, n_glob=N_GLOB,
                    max_types=MAX_TYPES, m_env=M_ENV, k_env=K_ENV)
