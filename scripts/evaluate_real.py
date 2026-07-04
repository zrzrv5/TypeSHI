"""Evaluate a checkpoint on the real-world targets (known ground truth).

Usage: uv run python scripts/evaluate_real.py <checkpoint.ckpt> [--frames 8]
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from ase.data import chemical_symbols
from ase.io import iread

from typeid2elem.augment import jitter
from typeid2elem.data import features_to_batch
from typeid2elem.decode import CompositionPrior
from typeid2elem.descriptors import average_features, compute_features
from typeid2elem.io import read_lammps_data, snapshot_from_atoms
from typeid2elem.model import TypeSetClassifier

LAMMPS_TARGETS = [
    # (name, path, element per original type id; None = type absent from file)
    ("LiSiPS-400K", "/home/zrzrv5/Documents/LiSiPS_Opt/Run_Jan2/490942_64507963/02_MSD/Data/400K.run.data",
     {"1": "Li", "2": "Si", "3": "P", "4": "S"}),
    ("LACO-TT1", "/home/zrzrv5/Documents/LACO/Relax_TT1/TT1.data",
     {"1": "Li", "2": "Al", "3": "Cl", "4": "O"}),
    ("NaCBH-600K", "/home/zrzrv5/Documents/NaCBH/iter58/Na2B12H12.Lo/Data/600K.data",
     {"1": "Na", "2": "C", "3": "B", "4": "H"}),
]
LPSC = "Data/COSMOS/benchmark_results/LPSC_MD/r2scan_DFT.extxyz"


def predict(models, snaps, tta=0, prior=None):
    if tta:
        rng = np.random.default_rng(0)
        snaps = list(snaps) + [jitter(s, rng, sigma_max=0.15)
                               for s in snaps for _ in range(tta)]
    feats = [compute_features(s) for s in snaps]
    if len(feats) > 1:
        feats.append(average_features(feats))
    logp = [torch.log(m.predict_probs(features_to_batch(f))[0] + 1e-12)
            for f in feats for m in models]
    mean_logp = torch.stack(logp).mean(0)
    if prior is not None:
        probs = torch.from_numpy(
            prior.marginals(mean_logp.numpy(), snaps[0].type_fractions()))
    else:
        probs = torch.exp(mean_logp)
    return probs / probs.sum(-1, keepdim=True)


def report(name, snap, probs, truth: dict[str, str], k_report=5):
    hits = {1: 0, 3: 0, 5: 0}
    n = 0
    print(f"\n== {name} ==")
    for t, label in enumerate(snap.orig_type_labels):
        true_el = truth.get(label)
        if true_el is None:
            continue
        n += 1
        order = torch.argsort(probs[t], descending=True)
        symbols = [chemical_symbols[z + 1] for z in order.tolist()]
        rank = symbols.index(true_el) + 1
        for k in hits:
            hits[k] += rank <= k
        top = ", ".join(f"{s} {probs[t, order[i]]:.1%}"
                        for i, s in enumerate(symbols[:k_report]))
        mark = "OK " if rank == 1 else f"r={rank}"
        print(f"  type {label} (true {true_el:>2}) [{mark}]: {top}")
    return hits, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="+", help="one or more checkpoints (ensembled)")
    ap.add_argument("--frames", type=int, default=8, help="LPSC frames to fuse")
    ap.add_argument("--tta", type=int, default=0, help="jittered TTA copies per frame")
    ap.add_argument("--decode", action="store_true",
                    help="composition-prior joint decoding (PMI + charge neutrality)")
    args = ap.parse_args()
    prior = CompositionPrior() if args.decode else None
    models = []
    for c in args.ckpt:
        m = TypeSetClassifier.load_from_checkpoint(c, map_location="cpu")
        m.eval()
        models.append(m)

    tot = {1: 0, 3: 0, 5: 0}
    n_tot = 0
    for name, path, truth in LAMMPS_TARGETS:
        snap = read_lammps_data(path)
        h, n = report(name, snap, predict(models, [snap], args.tta, prior), truth)
        for k in tot:
            tot[k] += h[k]
        n_tot += n

    # LPSC: elements are in the file; use them as anonymous type ids
    for tag, sl in [("LPSC-MD-1frame", slice(0, 1)),
                    (f"LPSC-MD-{args.frames}frames", slice(0, 200, max(1, 200 // args.frames)))]:
        snaps = [snapshot_from_atoms(a)[0] for a in iread(LPSC, index=sl)]
        snap = snaps[0]
        truth = {lab: lab for lab in snap.orig_type_labels}
        h, n = report(tag, snap, predict(models, snaps, args.tta, prior), truth)
        for k in tot:
            tot[k] += h[k]
        n_tot += n

    print(f"\nTOTAL over {n_tot} type-ids: "
          + "  ".join(f"top-{k}: {tot[k]}/{n_tot}" for k in (1, 3, 5)))


if __name__ == "__main__":
    main()
