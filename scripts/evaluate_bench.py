"""Extended real-world benchmark: original targets + files collected from ~/Documents.

Ground truth comes from element symbols in the file (xyz/pdb), Masses sections
(mapped to elements, never shown to the model), or known composition. The model
always sees anonymized type ids + geometry only.

Usage:
  uv run python scripts/evaluate_bench.py <ckpt...> [--tta N] [--decode] [--quiet]
"""

from __future__ import annotations

import argparse

from ase.io import read

from typeid2elem.decode import CompositionPrior
from typeid2elem.inference import load_models, predict, report
from typeid2elem.io import read_lammps_data, snapshot_from_atoms

D = "/home/zrzrv5/Documents"

# kind: lammps (truth dict by orig type label) | ase (truth = symbols in file)
#       ase-override (ase-parsed but all types are one known element)
BENCH = [
    # -- original targets --
    ("LiSiPS-400K", "lammps", f"{D}/LiSiPS_Opt/Run_Jan2/490942_64507963/02_MSD/Data/400K.run.data",
     {"1": "Li", "2": "Si", "3": "P", "4": "S"}),
    ("LACO-TT1-38at", "lammps", f"{D}/LACO/Relax_TT1/TT1.data",
     {"1": "Li", "2": "Al", "3": "Cl", "4": "O"}),
    ("NaCBH-600K", "lammps", f"{D}/NaCBH/iter58/Na2B12H12.Lo/Data/600K.data",
     {"1": "Na", "2": "C", "3": "B", "4": "H"}),
    ("LPSC-MD", "ase", "Data/COSMOS/benchmark_results/LPSC_MD/r2scan_DFT.extxyz", None),
    # -- collected: same chemistries, different sizes/phases --
    ("TT1-5x5x5-pdb", "ase", f"{D}/RMC/TT1/TT1_5x5x5.pdb", None),
    ("TT1-traj-xyz", "ase", f"{D}/RMC_ppt/Fit2/trajectory.xyz", None),
    ("LACO-RMC-TT3GR", "lammps", f"{D}/LACO/RMC/TT3GR.data",
     {"1": "Li", "2": "Al", "3": "Cl", "4": "O"}),
    ("LACO-RMC-TT3RR", "lammps", f"{D}/LACO/RMC/TT3RR.data",
     {"1": "Li", "2": "Al", "3": "Cl", "4": "O"}),
    ("LPS-glass-70:30", "lammps", f"{D}/LPS_dp/Data/70Li2S_30P2S5.data",
     {"1": "Li", "2": "P", "3": "S"}),
    ("NaBH-300K", "lammps", f"{D}/PlumedPG/DIFPG/EQ.300K.data",
     {"1": "Na", "2": "B", "3": "H"}),
    ("HfBaO-amor", "lammps", f"{D}/HfBaO/amorGen/Data/HfO2_020_1.data",
     {"1": "Hf", "2": "Ba", "3": "O"}),
    ("HfBaO-pog", "lammps", f"{D}/HfBaO/amorGen/pogdata/HfO2_1.data",
     {"1": "Hf", "2": "Ba", "3": "O"}),
    # -- collected: new regimes --
    ("NiTi-alloy", "ase", f"{D}/RMC_ppt/atomicNiTi/system.pdb", None),
    ("CHO-molecule", "ase", f"{D}/CHO/iter6.xyz", None),
    ("P2S6-molecule", "ase", f"{D}/LiSiPS_Opt/templates/frag/P2S6.xyz", None),
    # deepmd MD dump; per test_Na/input.lammps masses: types 1-4 = Na S P O
    # (ASE assigns placeholder symbols H/He/Li... to dump type ids 1/2/3...)
    ("NaSPO-MD", "ase-map", f"{D}/test_Na/traj/25000.lammpstrj",
     {"H": "Na", "He": "S", "Li": "P", "Be": "O"}),
    # MLFF folder (yolonas SMB share), copied to Data/eval_cases/MLFF/ --
    # POSCARs, element truth from the file headers (verified 2026-07-03).
    # Mostly NCM cathode chemistry (Li-Ni-Co-Mn-O +- Zr/Al/Fe dopants,
    # O-vacancies) + binary oxides; cathode-heavy on purpose (user's domain).
    ("MLFF-Li2O", "ase", "Data/eval_cases/MLFF/Li2O.poscar", None),
    ("MLFF-MnO2", "ase", "Data/eval_cases/MLFF/MnO2.poscar", None),
    ("MLFF-NiO", "ase", "Data/eval_cases/MLFF/NiO.poscar", None),
    ("MLFF-Co3O4", "ase", "Data/eval_cases/MLFF/Co3O4.poscar", None),
    ("MLFF-Li23-NCM", "ase", "Data/eval_cases/MLFF/Li23-NCM.poscar", None),
    ("MLFF-NCM-Li100-10K", "ase", "Data/eval_cases/MLFF/NCM811-Li100-10K.poscar", None),
    ("MLFF-NCM-poly", "ase", "Data/eval_cases/MLFF/NCM-poly.poscar", None),
    ("MLFF-NCM811-Ovac", "ase", "Data/eval_cases/MLFF/NCM811-1200K-Ovac.poscar", None),
    ("MLFF-NCM333-Ovac", "ase", "Data/eval_cases/MLFF/NCM333-1200K-Ovac.poscar", None),
    ("MLFF-NCM811-Zr", "ase", "Data/eval_cases/MLFF/NCM811-Zr-defect.poscar", None),
    ("MLFF-NCM811-Al-noLi", "ase", "Data/eval_cases/MLFF/NCM811-Al-noLi.poscar", None),
    ("MLFF-NCM333-Fe", "ase", "Data/eval_cases/MLFF/NCM333-Fe.poscar", None),
    # MDRun folder (yolonas SMB), copied to Data/eval_cases/MDRun/. Truths
    # verified via Masses sections / input scripts / stoichiometry (2026-07-03).
    # LFMP: input script has WRONG mass for type 5 (Fe's 55.845 on the Mn type)
    # -- stoichiometry 32:76 = Fe0.3:Mn0.7 settles it; mass lookup would fail.
    ("MDR-LFP-min", "lammps", "Data/eval_cases/MDRun/LFP333.min.data",
     {"1": "Fe", "2": "Li", "3": "O", "4": "P"}),
    ("MDR-LFMMP-min", "lammps", "Data/eval_cases/MDRun/LFMMP.min.data",
     {"1": "Fe", "2": "Li", "3": "O", "4": "P", "5": "Mn", "6": "Mg"}),
    ("MDR-LFMP-93", "lammps", "Data/eval_cases/MDRun/LFMP333.93.data",
     {"1": "Fe", "2": "Li", "3": "O", "4": "P", "5": "Mn"}),
    # core-shell polarizable model: each Fe/O is TWO particles (core+shell,
    # near-zero separation) -- physically pathological geometry, split types
    ("MDR-LFP-coreshell", "lammps", "Data/eval_cases/MDRun/LFP333.coreshell.data",
     {"1": "Fe", "2": "Li", "3": "O", "4": "P", "5": "Fe", "6": "O"}),
    ("MDR-NMC523", "lammps", "Data/eval_cases/MDRun/NMC523.data",
     {"1": "Li", "2": "Ni", "3": "O", "4": "Co", "5": "Mn"}),
    ("MDR-LiNiO2-5000", "lammps", "Data/eval_cases/MDRun/LiNiO2-5000.data",
     {"1": "Li", "2": "Ni", "3": "O"}),
]


def load_case(kind, path, truth):
    if kind == "lammps":
        snap = read_lammps_data(path)
        # verify masses (when present) agree with the stated truth -- labels only
        return snap, truth
    atoms = read(path, index=-1) if kind == "ase-map" or path.endswith(
        (".lammpstrj", ".xyz")) else read(path)
    snap, _ = snapshot_from_atoms(atoms)
    if kind == "ase-map":
        return snap, {lab: truth[lab] for lab in snap.orig_type_labels if lab in truth}
    return snap, {lab: lab for lab in snap.orig_type_labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="+")
    ap.add_argument("--tta", type=int, default=0)
    ap.add_argument("--decode", action="store_true")
    ap.add_argument("--debias", type=float, default=0.0)
    ap.add_argument("--fuse", default="conf",
                    choices=["logmean", "probmean", "conf", "median"])
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    models = load_models(args.ckpt)
    prior = CompositionPrior() if args.decode else None

    tot, n_tot = {1: 0, 3: 0, 5: 0}, 0
    rows = []
    for name, kind, path, truth in BENCH:
        try:
            snap, truth_map = load_case(kind, path, truth)
        except Exception as e:
            print(f"== {name} == LOAD ERROR: {e}")
            continue
        probs = predict(models, [snap], args.tta, prior, args.debias,
                        fuse=args.fuse)
        h, n = report(name, snap, probs, truth_map, quiet=args.quiet)
        rows.append((name, h, n))
        for k in tot:
            tot[k] += h[k]
        n_tot += n

    print(f"\n{'case':<18} top1 top3 top5   n")
    for name, h, n in rows:
        print(f"{name:<18} {h[1]:>4} {h[3]:>4} {h[5]:>4} {n:>3}")
    print(f"{'TOTAL':<18} {tot[1]:>4} {tot[3]:>4} {tot[5]:>4} {n_tot:>3}")


if __name__ == "__main__":
    main()
