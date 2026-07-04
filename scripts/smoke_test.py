"""Smoke test: parse eval LAMMPS files + one COSMOS frame, compute descriptors."""

import time

import numpy as np
from ase.io import iread

from typeid2elem.io import read_lammps_data, snapshot_from_atoms, mass_lookup
from typeid2elem.descriptors import compute_features

EVAL_FILES = {
    "LiSiPS (Li Si P S)": "/home/zrzrv5/Documents/LiSiPS_Opt/Run_Jan2/490942_64507963/02_MSD/Data/400K.run.data",
    "LACO   (Li Al Cl O)": "/home/zrzrv5/Documents/LACO/Relax_TT1/TT1.data",
    "NaCBH  (Na C B H)": "/home/zrzrv5/Documents/NaCBH/iter58/Na2B12H12.Lo/Data/600K.data",
}

for name, path in EVAL_FILES.items():
    t0 = time.time()
    snap = read_lammps_data(path)
    feats = compute_features(snap)
    dt = time.time() - t0
    print(f"\n=== {name} ===  N={len(snap.type_ids)} T={snap.n_types}  ({dt:.2f}s)")
    print(f"  fractions: {np.round(snap.type_fractions(), 3)}")
    print(f"  glob: {np.round(feats['glob'], 3)}  rdf shape {feats['rdf'].shape}")
    for t in range(snap.n_types):
        rmin = feats["pair_extra"][t, :, -1] * 8.0
        peak = feats["rdf"][t].max()
        mass = snap.type_masses.get(t)
        guess = mass_lookup(mass, 3) if mass else None
        print(f"  type {snap.orig_type_labels[t]}: closest-approach per partner "
              f"{np.round(rmin, 2)}  max g={peak:.1f}  mass-baseline={guess}")

print("\n=== COSMOS first frames ===")
for i, atoms in enumerate(iread("Data/COSMOS/DBS/dbs_total.extxyz", index=":3")):
    snap, zs = snapshot_from_atoms(atoms)
    feats = compute_features(snap)
    print(f"  frame {i}: {atoms.get_chemical_formula()} N={len(atoms)} "
          f"T={snap.n_types} Z={zs} rdf_max={feats['rdf'].max():.1f} "
          f"info_keys={list(atoms.info.keys())[:8]}")
