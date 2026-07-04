"""Verifier discriminativity check (research plan #1).

On labeled COSMOS structures, compare MACE force-RMS of the TRUE species vs a
radius-twin swap (the confusions our classifier actually makes). If truth wins
consistently, best-of-N verification is viable.

Usage: uv run python scripts/verifier_check.py [--n 80] [--device cpu]
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
from ase.io import iread

TWINS = {"Li": "Bi", "Na": "Ag", "Ba": "Rb", "Hf": "Sb", "O": "N",
         "Cl": "S", "Si": "P", "K": "Cs", "Ca": "Sr", "Ti": "Zr"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--source", default="Data/COSMOS/DBS/dbs_total.extxyz")
    args = ap.parse_args()

    from mace.calculators import mace_mp
    calc = mace_mp(model="small", device=args.device, default_dtype="float32")

    def frms(atoms):
        a = atoms.copy()
        a.calc = calc
        try:
            f = a.get_forces()
            return float(np.sqrt((f ** 2).sum(1).mean())) if np.isfinite(f).all() else 1e3
        except Exception:
            return 1e3

    wins, results = 0, []
    # stride through the file for source diversity (it's stored in db blocks)
    it = iread(args.source, index="::997")
    for atoms in itertools.islice(it, 4000):
        if len(results) >= args.n:
            break
        syms = set(atoms.get_chemical_symbols())
        swappable = [s for s in syms if s in TWINS and TWINS[s] not in syms]
        if not swappable or len(atoms) > 300 or len(syms) < 2:
            continue
        el = swappable[0]
        f_true = frms(atoms)
        sw = atoms.copy()
        sw.set_chemical_symbols([TWINS[s] if s == el else s
                                 for s in sw.get_chemical_symbols()])
        f_swap = frms(sw)
        truth_wins = f_true < f_swap
        wins += truth_wins
        results.append((el, TWINS[el], f_true, f_swap))
        if len(results) % 20 == 0:
            print(f"  {len(results)} done, truth wins {wins}/{len(results)}")

    print(f"\nTruth wins: {wins}/{len(results)} = {wins/len(results):.0%}")
    from collections import defaultdict
    per = defaultdict(lambda: [0, 0])
    for el, tw, ft, fs in results:
        per[f"{el}->{tw}"][0] += ft < fs
        per[f"{el}->{tw}"][1] += 1
    for k, (w, n) in sorted(per.items()):
        print(f"  {k:<8} truth wins {w}/{n}")


if __name__ == "__main__":
    main()
