"""Predict elements for each type id in an MD file (geometry only).

Usage:
  uv run python scripts/predict.py <file> [<file2> ...] [--top-k 5] \
      [--format lammps-data|ase] [--ckpt a.ckpt b.ckpt ...] [--no-decode]

Defaults to the production ensemble in runs/production/ with composition-prior
decoding. Multiple files of the SAME system (frames) are fused via time-averaged
descriptors + mean log-probs. Masses (if present in the file) are shown as an
independent baseline column -- they are never fed to the model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ase.data import chemical_symbols

from typeid2elem.decode import CompositionPrior
from typeid2elem.inference import load_models, predict
from typeid2elem.io import mass_lookup, read_lammps_data
from typeid2elem.units import UNIT_TO_ANG, infer_units, rescaled_snapshot

PROD = sorted(str(p) for p in Path("runs/production").glob("*.ckpt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--ckpt", nargs="+", default=PROD)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--format", default="lammps-data", choices=["lammps-data", "ase"])
    ap.add_argument("--no-decode", action="store_true")
    ap.add_argument("--units", default="auto",
                    choices=["auto"] + list(UNIT_TO_ANG),
                    help="length unit of the input file (default: infer; "
                         "also gates out non-atomistic e.g. DEM data)")
    ap.add_argument("--force", action="store_true",
                    help="predict even if the input does not look atomistic")
    ap.add_argument("--conformal", action="store_true",
                    help="also print coverage-guaranteed candidate sets "
                         "(needs runs/production/calib.npz from calibrate.py)")
    args = ap.parse_args()
    if not args.ckpt:
        raise SystemExit("no checkpoints found (runs/production/ empty; pass --ckpt)")

    snaps = []
    for p in args.files:
        if args.format == "lammps-data":
            snaps.append(read_lammps_data(p))
        else:
            from ase.io import read

            from typeid2elem.io import snapshot_from_atoms
            snaps.append(snapshot_from_atoms(read(p))[0])

    models = load_models(args.ckpt)

    if args.units == "auto":
        res = infer_units(snaps[0], models=models)
        if res["verdict"] == "not_atoms":
            print(f"\nNOT ATOMISTIC: {res['reason']}")
            print("(element prediction skipped; use --force to override, "
                  "--units <unit> to set the unit manually)")
            if not args.force:
                raise SystemExit(2)
        else:
            note = ""
            if res["tiebreak"]:
                c = res["tiebreak"]["confidence"]
                note = (" (model-confidence tiebreak: "
                        + ", ".join(f"{u} {v:.2f}" for u, v in c.items()) + ")")
            print(f"\nInferred length unit: {res['unit']} "
                  f"(NN in-window {res['score']:.0%}){note}")
            snaps = [rescaled_snapshot(s, res) for s in snaps]
    elif UNIT_TO_ANG[args.units] != 1.0:
        s = UNIT_TO_ANG[args.units]
        snaps = [rescaled_snapshot(x, dict(verdict="atoms", scale=s))
                 for x in snaps]

    prior = None if args.no_decode else CompositionPrior()
    probs = predict(models, snaps, prior=prior)

    ref = snaps[0]
    print(f"\nSystem: {len(ref.type_ids)} atoms, {ref.n_types} types, "
          f"{len(snaps)} frame(s), {len(models)}-model ensemble"
          f"{', composition decode' if prior else ''}")
    calib = None
    if args.conformal:
        import numpy as np

        from calibrate import raps_set
        from typeid2elem.assets import calib_npz
        cpath = Path(calib_npz())
        if not cpath.exists():
            raise SystemExit("--conformal needs calib.npz (weights/ or "
                             "runs/production/; run scripts/calibrate.py)")
        calib = dict(np.load(cpath).items())

    for t in range(ref.n_types):
        top = torch.topk(probs[t], args.top_k)
        cand = ", ".join(f"{chemical_symbols[z + 1]} {p:.1%}"
                         for p, z in zip(top.values.tolist(), top.indices.tolist()))
        line = f"  type {ref.orig_type_labels[t]}: {cand}"
        if t in ref.type_masses:
            line += f"   [mass baseline: {mass_lookup(ref.type_masses[t], 1)[0][0]}]"
        print(line)
        if calib is not None:
            import numpy as np
            z = np.log(probs[t].numpy() + 1e-12) / calib["temperature"]
            z -= z.max()
            p = np.exp(z) / np.exp(z).sum()
            s = raps_set(p, calib["qhat"], calib["lam"], int(calib["k_reg"]))
            syms = [chemical_symbols[c + 1] for c in s]
            note = "  <- highly uncertain" if len(s) > 15 else ""
            print(f"      {calib['coverage']:.0%}-coverage set "
                  f"({len(s)}): {{{', '.join(syms)}}}{note}")


if __name__ == "__main__":
    main()
