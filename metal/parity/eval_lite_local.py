"""Score the lite (ONNX + decode) predictor on the eval-bench cases that ship in
this repo (Data/eval_cases/{MLFF,MDRun}). The full BENCH in
scripts/evaluate_bench.py references files under /home/... and Data/COSMOS that a
fresh clone / this Mac doesn't have; those are simply omitted here.

This is the pipeline the Metal port feeds (descriptors -> deploy model -> decode);
Metal descriptors are proven equal to these Python descriptors by the parity
harness, so this measures the on-device accuracy on real files. Uses the same
4-env-draw pooling + composition decode as scripts/predict_lite.py.

Usage:  <verify-venv>/bin/python metal/parity/eval_lite_local.py [--no-decode]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from ase.data import chemical_symbols  # noqa: E402
from ase.io import read  # noqa: E402

from predict_lite import OnnxPredictor  # noqa: E402  (torch-free)
from typeid2elem.assets import deploy_onnx  # noqa: E402
from typeid2elem.decode import CompositionPrior  # noqa: E402
from typeid2elem.descriptors import compute_features_capped  # noqa: E402
from typeid2elem.io import read_lammps_data, snapshot_from_atoms  # noqa: E402

E = "Data/eval_cases"
# Subset of scripts/evaluate_bench.py BENCH whose files are committed to the repo.
BENCH = [
    ("MLFF-Li2O", "ase", f"{E}/MLFF/Li2O.poscar", None),
    ("MLFF-MnO2", "ase", f"{E}/MLFF/MnO2.poscar", None),
    ("MLFF-NiO", "ase", f"{E}/MLFF/NiO.poscar", None),
    ("MLFF-Co3O4", "ase", f"{E}/MLFF/Co3O4.poscar", None),
    ("MLFF-Li23-NCM", "ase", f"{E}/MLFF/Li23-NCM.poscar", None),
    ("MLFF-NCM-Li100-10K", "ase", f"{E}/MLFF/NCM811-Li100-10K.poscar", None),
    ("MLFF-NCM-poly", "ase", f"{E}/MLFF/NCM-poly.poscar", None),
    ("MLFF-NCM811-Ovac", "ase", f"{E}/MLFF/NCM811-1200K-Ovac.poscar", None),
    ("MLFF-NCM333-Ovac", "ase", f"{E}/MLFF/NCM333-1200K-Ovac.poscar", None),
    ("MLFF-NCM811-Zr", "ase", f"{E}/MLFF/NCM811-Zr-defect.poscar", None),
    ("MLFF-NCM811-Al-noLi", "ase", f"{E}/MLFF/NCM811-Al-noLi.poscar", None),
    ("MLFF-NCM333-Fe", "ase", f"{E}/MLFF/NCM333-Fe.poscar", None),
    ("MDR-LFP-min", "lammps", f"{E}/MDRun/LFP333.min.data",
     {"1": "Fe", "2": "Li", "3": "O", "4": "P"}),
    ("MDR-LFMMP-min", "lammps", f"{E}/MDRun/LFMMP.min.data",
     {"1": "Fe", "2": "Li", "3": "O", "4": "P", "5": "Mn", "6": "Mg"}),
    ("MDR-LFMP-93", "lammps", f"{E}/MDRun/LFMP333.93.data",
     {"1": "Fe", "2": "Li", "3": "O", "4": "P", "5": "Mn"}),
    ("MDR-LFP-coreshell", "lammps", f"{E}/MDRun/LFP333.coreshell.data",
     {"1": "Fe", "2": "Li", "3": "O", "4": "P", "5": "Fe", "6": "O"}),
    ("MDR-NMC523", "lammps", f"{E}/MDRun/NMC523.data",
     {"1": "Li", "2": "Ni", "3": "O", "4": "Co", "5": "Mn"}),
    ("MDR-LiNiO2-5000", "lammps", f"{E}/MDRun/LiNiO2-5000.data",
     {"1": "Li", "2": "Ni", "3": "O"}),
]


def load_case(kind, path, truth):
    if kind == "lammps":
        return read_lammps_data(REPO / path), truth
    snap, _ = snapshot_from_atoms(read(REPO / path))
    return snap, {lab: lab for lab in snap.orig_type_labels}


def main() -> None:
    decode = "--no-decode" not in sys.argv
    model = OnnxPredictor(deploy_onnx())
    prior = CompositionPrior() if decode else None

    tot = {1: 0, 3: 0, 5: 0}
    n_tot = 0
    print(f"{'case':<22} {'n_at':>6} top1 top3 top5   n   misses")
    for name, kind, path, truth in BENCH:
        snap, tmap = load_case(kind, path, truth)
        logp = np.mean([model.logprobs(compute_features_capped(
                            snap, rng=np.random.default_rng(1000 + d)))
                        for d in range(4)], axis=0)
        probs = prior.marginals(logp, snap.type_fractions()) if decode else np.exp(logp)
        h = {1: 0, 3: 0, 5: 0}
        n = 0
        misses = []
        for t, lab in enumerate(snap.orig_type_labels):
            true_el = tmap.get(lab)
            if true_el is None:
                continue
            n += 1
            order = [chemical_symbols[z + 1] for z in np.argsort(-probs[t])]
            rank = order.index(true_el) + 1
            for k in h:
                h[k] += rank <= k
            if rank > 1:
                misses.append(f"{true_el}(r{rank}:{order[0]})")
        for k in tot:
            tot[k] += h[k]
        n_tot += n
        print(f"{name:<22} {len(snap.type_ids):>6} {h[1]:>4} {h[3]:>4} {h[5]:>4} "
              f"{n:>3}   {' '.join(misses)}")
    print(f"\n{'TOTAL':<22} {'':>6} {tot[1]:>4} {tot[3]:>4} {tot[5]:>4} {n_tot:>3}  "
          f"decode={decode}")
    print(f"top-1 {tot[1]}/{n_tot} = {tot[1]/n_tot:.0%}  "
          f"top-3 {tot[3]}/{n_tot} = {tot[3]/n_tot:.0%}  "
          f"top-5 {tot[5]}/{n_tot} = {tot[5]/n_tot:.0%}")


if __name__ == "__main__":
    main()
