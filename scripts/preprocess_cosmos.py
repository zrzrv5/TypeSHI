"""Preprocess an ASE-readable trajectory file (COSMOS extxyz) into feature shards.

Usage:
  uv run python scripts/preprocess_cosmos.py Data/COSMOS/DBS/dbs_total.extxyz \
      --out Data/processed/cosmos --n-aug 2 [--limit 50000] [--workers 16]
"""

from __future__ import annotations

import argparse
import itertools
from multiprocessing import Pool

import numpy as np
from ase.io import iread
from tqdm import tqdm

from typeid2elem.io import snapshot_from_atoms
from typeid2elem.preprocess import ShardWriter, group_id, make_records

_N_AUG = 0


def _init(n_aug):
    global _N_AUG
    _N_AUG = n_aug


def _work(args):
    idx, atoms = args
    try:
        snap, zs = snapshot_from_atoms(atoms)
        if len(snap.type_ids) < 2:
            return []
        gid = group_id(tuple(snap.orig_type_labels))
        rng = np.random.default_rng(idx)  # deterministic per frame
        return make_records(snap, zs, gid, rng, _N_AUG)
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="cosmos")
    ap.add_argument("--n-aug", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    frames = enumerate(iread(args.traj, index=":"))
    if args.limit:
        frames = itertools.islice(frames, args.limit)

    writer = ShardWriter(args.out, args.prefix)
    with Pool(args.workers, initializer=_init, initargs=(args.n_aug,)) as pool:
        for recs in tqdm(pool.imap_unordered(_work, frames, chunksize=64),
                         unit="frame", smoothing=0.01, mininterval=10):
            for r in recs:
                writer.add(r)
    writer.close()
    print(f"wrote {writer.total} records to {args.out}")


if __name__ == "__main__":
    main()
