"""Preprocess COD (Crystallography Open Database) CIF files into feature shards.

COD is EXPERIMENTAL data: diverse chemistry (organics, minerals,
organometallics) that no DFT corpus covers, but with experimental quirks --
partially occupied/disordered sites are skipped, and X-ray structures often
omit H (harmless: we label the atoms that are present).

Usage:
  uv run python scripts/preprocess_cod.py Data/COD/cif \
      --out Data/processed/cod_v3 [--limit 300000] [--workers 12]
"""

from __future__ import annotations

import argparse
import itertools
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

from typeid2elem.io import snapshot_from_atoms
from typeid2elem.preprocess import ShardWriter, group_id, make_records

_N_AUG = 0
MAX_ATOMS = 2000


def _init(n_aug):
    global _N_AUG
    _N_AUG = n_aug


def _work(args):
    idx, path = args
    try:
        from ase.io import read
        atoms = read(path, format="cif")
        occ = atoms.info.get("occupancy", {})
        if occ and any(o < 0.95 for site in occ.values()
                       for o in (site.values() if isinstance(site, dict)
                                 else [site])):
            return []                        # disordered / partial occupancy
        if not (2 <= len(atoms) <= MAX_ATOMS):
            return []
        if atoms.cell.volume < 1.0:
            return []
        snap, zs = snapshot_from_atoms(atoms)
        # zs.min() < 1 catches dummy species ('X', Z=0) in experimental CIFs --
        # a single such label crashes CE with a device-side assert
        if snap.n_types < 2 or zs.max() > 94 or zs.min() < 1:
            return []
        gid = group_id(tuple(snap.orig_type_labels))
        rng = np.random.default_rng(idx)
        return make_records(snap, zs, gid, rng, _N_AUG)
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="cod")
    ap.add_argument("--n-aug", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    paths = enumerate(Path(args.root).rglob("*.cif"))
    if args.limit:
        paths = itertools.islice(paths, args.limit)

    writer = ShardWriter(args.out, args.prefix)
    n_in = 0
    with Pool(args.workers, initializer=_init, initargs=(args.n_aug,)) as pool:
        for recs in tqdm(pool.imap_unordered(_work, paths, chunksize=32),
                         unit="cif", smoothing=0.01, mininterval=10):
            n_in += 1
            for r in recs:
                writer.add(r)
    writer.close()
    print(f"scanned {n_in} CIFs -> wrote {writer.total} records to {args.out}")


if __name__ == "__main__":
    main()
