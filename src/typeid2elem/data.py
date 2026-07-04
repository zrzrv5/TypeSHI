"""Dataset over preprocessed npz shards + padded collate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess import FEATURE_DIMS

N_CLASSES = 94  # atomic numbers 1..94 -> class Z-1
VAL_FRACTION_MOD = 10  # gid % 10 == 0 -> validation (split by element-set)


class ShardDataset(Dataset):
    """Loads all shards into RAM (float16 features are compact)."""

    def __init__(self, shard_dirs: list[str | Path], split: str = "train",
                 max_types: int | None = None):
        files = sorted(f for d in shard_dirs for f in Path(d).glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"no shards under {shard_dirs}")
        self.items: list[tuple] = []
        for f in files:
            with np.load(f) as z:
                gid = z["gid"]
                is_val = (gid % VAL_FRACTION_MOD) == 0
                keep = is_val if split == "val" else ~is_val
                if max_types is not None:
                    keep &= np.array([len(l) <= max_types for l in z["labels"]])
                if not keep.any():
                    continue
                rdf, pe = z["rdf"][keep], z["pair_extra"][keep]
                fr, gl, lb = z["frac"][keep], z["glob"][keep], z["labels"][keep]
                has_env = "env_d" in z.files
                ed = z["env_d"][keep] if has_env else None
                et = z["env_t"][keep] if has_env else None
            for i in range(len(lb)):
                self.items.append((rdf[i], pe[i], fr[i], gl[i], lb[i],
                                   ed[i] if has_env else None,
                                   et[i] if has_env else None))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rdf, pe, fr, gl, lb, ed, et = self.items[i]
        out = {
            "rdf": torch.from_numpy(rdf.astype(np.float32)),
            "pair_extra": torch.from_numpy(pe.astype(np.float32)),
            "frac": torch.from_numpy(fr.astype(np.float32)),
            "glob": torch.from_numpy(gl.astype(np.float32)),
            "labels": torch.from_numpy(lb.astype(np.int64)) - 1,  # class = Z-1
        }
        if ed is not None:
            out["env_d"] = torch.from_numpy(ed.astype(np.float32))
            out["env_t"] = torch.from_numpy(et.astype(np.int64))
        return out

    def class_counts(self) -> np.ndarray:
        counts = np.zeros(N_CLASSES, dtype=np.int64)
        for it in self.items:
            lb = it[4]
            np.add.at(counts, lb.astype(np.int64) - 1, 1)
        return counts


def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    B = len(batch)
    Tm = max(len(b["labels"]) for b in batch)
    nb = FEATURE_DIMS["n_bins"]
    npx = FEATURE_DIMS["n_pair_extra"]
    out = {
        "rdf": torch.zeros(B, Tm, Tm, nb),
        "pair_extra": torch.zeros(B, Tm, Tm, npx),
        "frac": torch.zeros(B, Tm),
        "glob": torch.stack([b["glob"] for b in batch]),
        "labels": torch.full((B, Tm), -100, dtype=torch.int64),
        "mask": torch.zeros(B, Tm, dtype=torch.bool),
    }
    has_env = all("env_d" in b for b in batch)
    if has_env:
        m, k = batch[0]["env_d"].shape[-2:]
        out["env_d"] = torch.zeros(B, Tm, m, k)
        out["env_t"] = torch.full((B, Tm, m, k), -1, dtype=torch.int64)
    for i, b in enumerate(batch):
        t = len(b["labels"])
        out["rdf"][i, :t, :t] = b["rdf"]
        out["pair_extra"][i, :t, :t] = b["pair_extra"]
        out["frac"][i, :t] = b["frac"]
        out["labels"][i, :t] = b["labels"]
        out["mask"][i, :t] = True
        if has_env:
            out["env_d"][i, :t] = b["env_d"]
            out["env_t"][i, :t] = b["env_t"]
    return out


def features_to_batch(feats: dict[str, np.ndarray], device="cpu") -> dict[str, torch.Tensor]:
    """Single inference sample (from descriptors.compute_features) -> batch of 1."""
    t = len(feats["frac"])
    out = {
        "rdf": torch.as_tensor(feats["rdf"], dtype=torch.float32, device=device)[None],
        "pair_extra": torch.as_tensor(feats["pair_extra"], dtype=torch.float32, device=device)[None],
        "frac": torch.as_tensor(feats["frac"], dtype=torch.float32, device=device)[None],
        "glob": torch.as_tensor(feats["glob"], dtype=torch.float32, device=device)[None],
        "mask": torch.ones(1, t, dtype=torch.bool, device=device),
    }
    if "env_d" in feats:
        out["env_d"] = torch.as_tensor(feats["env_d"], dtype=torch.float32, device=device)[None]
        out["env_t"] = torch.as_tensor(feats["env_t"], dtype=torch.int64, device=device)[None]
    return out
