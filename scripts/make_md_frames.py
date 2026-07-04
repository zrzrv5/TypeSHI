"""Generate finite-temperature MD frames from MPTrj structures with MACE-MP-0.

Rationale (replaces the RAFT idea, docs/RESEARCH2.md Q5): RAFT distills a
verifier's label choices, but every corpus we may train on already has TRUE
element labels -- verifier-chosen labels are strictly noisier. What the
verifier/RL discussion actually points at is the domain gap: training data is
near-equilibrium DFT, deployment data is hot MD. So generate genuine thermal
ensembles (short NVT MD with an MLIP) for training structures and fine-tune
on them with their known labels.

Skips val-split element sets (gid % 10 == 0) so validation and the real bench
stay clean. Output: extxyz frames, later preprocessed by preprocess_cosmos.py.

Usage:
  uv run python scripts/make_md_frames.py Data/MPtrj/MPtrj_2022.9_full.json \
      --out Data/MDgen/hot_frames.extxyz --n-materials 1500 --steps 300 --workers 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import ijson
import numpy as np
from ase import Atoms, units
from ase.io import write
from ase.md.langevin import Langevin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from typeid2elem.preprocess import group_id  # noqa: E402

VAL_MOD = 10


def structure_iter(path, stride):
    """Yield the first frame of every stride-th material."""
    with open(path, "rb") as fh:
        for n_mat, (mp_id, frames) in enumerate(
                ijson.kvitems(fh, "", use_float=True)):
            if n_mat % stride:
                continue
            key = next(iter(frames))
            st = frames[key]["structure"]
            cell = np.array([np.asarray(r) for r in st["lattice"]["matrix"]],
                            dtype=float)
            xyz = np.array([np.asarray(s["xyz"]) for s in st["sites"]],
                           dtype=float)
            syms = [s["species"][0]["element"] for s in st["sites"]]
            yield mp_id, Atoms(symbols=syms, positions=xyz, cell=cell, pbc=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-materials", type=int, default=1500)
    ap.add_argument("--stride", type=int, default=100,
                    help="take every Nth material (MPTrj has ~146k)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--frames-per-traj", type=int, default=3)
    ap.add_argument("--max-atoms", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from mace.calculators import mace_mp
    calc = mace_mp(model="small", device="cuda", default_dtype="float32")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    n_done = n_skip = 0
    t0 = time.time()
    with open(out, "w") as fh:
        for mp_id, atoms in structure_iter(args.json_path, args.stride):
            if n_done >= args.n_materials:
                break
            uniq = sorted(set(atoms.get_chemical_symbols()))
            if (len(atoms) < 4 or len(atoms) > args.max_atoms or len(uniq) < 2
                    or group_id(tuple(uniq)) % VAL_MOD == 0):
                n_skip += 1
                continue
            temp = float(rng.uniform(300, 900))
            try:
                atoms.calc = calc
                dyn = Langevin(atoms, 2.0 * units.fs, temperature_K=temp,
                               friction=0.02)
                save_at = np.linspace(args.steps // 2, args.steps,
                                      args.frames_per_traj).astype(int)
                frames = []
                for step in range(1, args.steps + 1):
                    dyn.run(1)
                    if step in save_at:
                        # melted/exploded guard: nearest neighbor sanity
                        d = atoms.get_all_distances(mic=True)
                        np.fill_diagonal(d, np.inf)
                        if d.min() < 0.6 or not np.isfinite(
                                atoms.get_potential_energy()):
                            raise RuntimeError("unphysical")
                        a = atoms.copy()
                        a.info = {"mp_id": mp_id, "temp_K": round(temp)}
                        frames.append(a)
                for a in frames:
                    write(fh, a, format="extxyz")
                fh.flush()
                n_done += 1
            except Exception:
                n_skip += 1
                continue
            if n_done % 25 == 0:
                rate = n_done / (time.time() - t0)
                print(f"{n_done} trajs ({n_skip} skipped), "
                      f"{rate:.2f} traj/s, eta "
                      f"{(args.n_materials - n_done) / max(rate, 1e-9) / 60:.0f} min",
                      flush=True)
    print(f"done: {n_done} trajectories x {args.frames_per_traj} frames -> {out}")


if __name__ == "__main__":
    main()
