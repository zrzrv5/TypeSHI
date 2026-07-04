"""Preprocess MPTrj (12GB nested JSON) into feature shards, streaming with ijson.

Samples up to --frames-per-material frames per mp-id (first/middle/last of the
relaxation trajectory = most diverse geometries).

Usage:
  uv run python scripts/preprocess_mptrj.py Data/MPtrj/MPtrj_2022.9_full.json \
      --out Data/processed/mptrj --n-aug 1 --frames-per-material 2 [--limit 1000]
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool

import ijson
import numpy as np
from tqdm import tqdm

from typeid2elem.io import Snapshot
from typeid2elem.preprocess import ShardWriter, group_id, make_records

_N_AUG = 0


def _init(n_aug):
    global _N_AUG
    _N_AUG = n_aug


def _work(args):
    idx, cell, positions, symbols = args
    try:
        symbols = np.asarray(symbols)
        uniq, type_ids = np.unique(symbols, return_inverse=True)
        if len(uniq) < 2 or len(symbols) < 2:
            return []
        from ase.data import atomic_numbers

        zs = np.array([atomic_numbers[s] for s in uniq], dtype=np.int64)
        snap = Snapshot(np.asarray(positions), type_ids.astype(np.int64),
                        np.asarray(cell), True, orig_type_labels=list(uniq))
        rng = np.random.default_rng(idx)
        return make_records(snap, zs, group_id(tuple(uniq)), rng, _N_AUG)
    except Exception:
        return []


def frame_iter(path, frames_per_material, limit):
    with open(path, "rb") as fh:
        idx = 0
        for n_mat, (mp_id, frames) in enumerate(ijson.kvitems(fh, "", use_float=True)):
            if limit and n_mat >= limit:
                return
            keys = list(frames)
            if frames_per_material and len(keys) > frames_per_material:
                pick = np.linspace(0, len(keys) - 1, frames_per_material).astype(int)
                keys = [keys[i] for i in np.unique(pick)]
            for k in keys:
                st = frames[k]["structure"]
                cell = st["lattice"]["matrix"]
                positions = [s["xyz"] for s in st["sites"]]
                symbols = [s["species"][0]["element"] for s in st["sites"]]
                idx += 1
                yield (idx, cell, positions, symbols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="mptrj")
    ap.add_argument("--n-aug", type=int, default=1)
    ap.add_argument("--frames-per-material", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()

    writer = ShardWriter(args.out, args.prefix)
    frames = frame_iter(args.json_path, args.frames_per_material, args.limit)
    with Pool(args.workers, initializer=_init, initargs=(args.n_aug,)) as pool:
        for recs in tqdm(pool.imap_unordered(_work, frames, chunksize=32),
                         unit="frame", smoothing=0.01, mininterval=10):
            for r in recs:
                writer.add(r)
    writer.close()
    print(f"wrote {writer.total} records to {args.out}")


if __name__ == "__main__":
    main()
