"""Preprocess DeePMD-format system trees (openLAM domains) into feature shards.

A system dir has: type.raw (per-atom type idx), type_map.raw (idx -> element),
set.*/coord.npy (F, N*3) and box.npy (F, 9; all-zero rows => non-periodic).

Usage:
  uv run python scripts/preprocess_deepmd.py Data/openLAM/Alloy Data/openLAM/Anode \
      --out Data/processed/openlam --max-frames-per-sys 4 --n-aug 1 --workers 12
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from ase.data import atomic_numbers
from tqdm import tqdm

from typeid2elem.io import Snapshot
from typeid2elem.preprocess import ShardWriter, group_id, make_records

_ARGS = None


def _init(a):
    global _ARGS
    _ARGS = a


def find_systems(roots):
    systems = []
    for root in roots:
        for tm in Path(root).rglob("type_map.raw"):
            d = tm.parent
            if (d / "type.raw").exists() and list(d.glob("set.*")):
                systems.append(d)
    return sorted(set(systems))


def _work(args):
    sys_idx, d = args
    try:
        d = Path(d)
        type_map = d.joinpath("type_map.raw").read_text().split()
        atom_types = np.loadtxt(d / "type.raw", dtype=int).reshape(-1)
        n = len(atom_types)

        # DeePMD "mixed-type" systems store real species per frame in
        # set.*/real_atom_types.npy (type.raw is then meaningless padding;
        # virtual atoms have type -1 and must be stripped).
        mixed = any((s / "real_atom_types.npy").exists() for s in d.glob("set.*"))
        coords, boxes, frame_types = [], [], []
        for s in sorted(d.glob("set.*")):
            c = np.load(s / "coord.npy")
            b = np.load(s / "box.npy") if (s / "box.npy").exists() else np.zeros((len(c), 9))
            coords.append(c.reshape(len(c), -1, 3))
            boxes.append(b.reshape(len(b), 3, 3))
            if mixed:
                frame_types.append(np.load(s / "real_atom_types.npy").reshape(len(c), -1))
        coords = np.concatenate(coords)
        boxes = np.concatenate(boxes)
        if mixed:
            frame_types = np.concatenate(frame_types)
        else:
            if coords.shape[1] != n:
                return []
            frame_types = np.broadcast_to(atom_types, (len(coords), n))

        k = _ARGS.max_frames_per_sys
        pick = np.unique(np.linspace(0, len(coords) - 1, min(k, len(coords))).astype(int))
        rng = np.random.default_rng(sys_idx)
        records = []
        for fi in pick:
            ft = np.asarray(frame_types[fi])
            real = ft >= 0
            symbols = [type_map[t] for t in ft[real]]
            uniq, type_ids = np.unique(symbols, return_inverse=True)
            if len(uniq) < 2:
                continue
            zs = np.array([atomic_numbers.get(s, 0) for s in uniq], dtype=np.int64)
            if zs.min() < 1 or zs.max() > 94:
                continue
            cell, pbc = boxes[fi], True
            pos = coords[fi][real]
            if abs(np.linalg.det(cell)) < 1e-6:
                cell, pbc = None, False
            else:
                frac = np.linalg.solve(cell.T, pos.T).T % 1.0
                pos = frac @ cell
            snap = Snapshot(pos, type_ids.astype(np.int64), cell, pbc,
                            orig_type_labels=list(uniq))
            records += make_records(snap, zs, group_id(tuple(uniq)), rng, _ARGS.n_aug)
        return records
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="openlam")
    ap.add_argument("--max-frames-per-sys", type=int, default=4)
    ap.add_argument("--n-aug", type=int, default=1)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    systems = find_systems(args.roots)
    print(f"{len(systems)} systems found")
    writer = ShardWriter(args.out, args.prefix)
    with Pool(args.workers, initializer=_init, initargs=(args,)) as pool:
        for recs in tqdm(pool.imap_unordered(_work, enumerate(systems), chunksize=8),
                         total=len(systems), unit="sys", smoothing=0.01, mininterval=10):
            for r in recs:
                writer.add(r)
    writer.close()
    print(f"wrote {writer.total} records to {args.out}")


if __name__ == "__main__":
    main()
