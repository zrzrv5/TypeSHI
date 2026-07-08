"""Exercise the two descriptor paths the eval files don't cover: an OPEN-BOX
(cell-less molecule -> has_cell=1.0, vacuum-box volume, pbc=0 kernel) and a small
TRICLINIC cell (off-diagonal tilt -> tests the s.x*a0+s.y*a1+s.z*a2 image shift and
the perpendicular-height replica counts). For each, diff Metal desc-dump against
the Python descriptors.compute_features reference.

Prereq:  swift build --package-path metal
Usage:   <verify-venv>/bin/python metal/parity/check_edge_cases.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from typeid2elem.descriptors import compute_features  # noqa: E402
from typeid2elem.io import Snapshot  # noqa: E402

DESC_DUMP = REPO / "metal/.build/debug/desc-dump"
NB, NPX, MENV, KENV = 64, 8, 16, 16
TOL = {"rdf": 5e-2, "pair_extra": 5e-3, "frac": 1e-4, "glob": 1e-3}


def metal(snap):
    with tempfile.TemporaryDirectory() as td:
        inp, out = Path(td) / "in.json", Path(td) / "out.json"
        cell = snap.cell.tolist() if (snap.cell is not None and snap.pbc) else None
        inp.write_text(json.dumps({
            "positions": snap.positions.tolist(),
            "type_ids": [int(t) for t in snap.type_ids],
            "cell": cell,
        }))
        subprocess.run([str(DESC_DUMP), str(inp), str(out), "0"],
                       check=True, capture_output=True)
        d = json.loads(out.read_text())
    t = d["nTypes"]
    return {
        "rdf": np.array(d["rdf"], np.float32).reshape(t, t, NB),
        "pair_extra": np.array(d["pair_extra"], np.float32).reshape(t, t, NPX),
        "frac": np.array(d["frac"], np.float32),
        "glob": np.array(d["glob"], np.float32),
    }


def check(name, snap):
    m = metal(snap)
    p = compute_features(snap, with_env=False)
    print(f"\n== {name}: {len(snap.type_ids)} atoms, {snap.n_types} types, "
          f"cell={'yes' if snap.cell is not None and snap.pbc else 'OPEN'} ==")
    ok = True
    for key in ("rdf", "pair_extra", "frac", "glob"):
        dd = float(np.abs(m[key] - p[key]).max())
        passed = dd <= TOL[key]
        ok = ok and passed
        print(f"  {'OK ' if passed else 'BAD'} {key:<11} max|Δ| = {dd:.3e}  (tol {TOL[key]})")
    print(f"  glob (metal): {np.round(m['glob'],4).tolist()}   "
          f"glob (py): {np.round(p['glob'],4).tolist()}")
    return ok


def main() -> None:
    if not DESC_DUMP.exists():
        sys.exit(f"missing {DESC_DUMP}; run: swift build --package-path metal")
    rng = np.random.default_rng(7)

    # (a) OPEN-BOX molecule: 3 types, cell-less. Spread atoms so a few pairs fall
    # within R_MAX but images never interact (that's the vacuum-box premise).
    pos = np.array([
        [0, 0, 0], [1.1, 0, 0], [0, 1.2, 0], [2.4, 0.3, 0.1], [1.0, 1.0, 1.0],
        [3.5, 1.0, 0.5], [0.2, 3.0, 1.0], [2.0, 2.0, 2.0], [4.0, 4.0, 0.0],
        [1.5, 3.5, 2.5], [3.0, 0.5, 3.0], [0.5, 0.5, 4.0], [4.5, 2.0, 1.5],
        [2.5, 4.5, 3.5], [1.0, 2.0, 5.0],
    ], float)
    typ = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], np.int64)
    open_snap = Snapshot(positions=pos, type_ids=typ, cell=None, pbc=False,
                         orig_type_labels=["1", "2", "3"])

    # (b) TRICLINIC crystal small enough that R_MAX spans multiple images; strong
    # off-diagonal tilt stresses the general image shift.
    cell = np.array([[6.0, 0.0, 0.0], [2.0, 6.0, 0.0], [1.0, 1.5, 6.0]])
    frac = rng.random((40, 3))
    tpos = frac @ cell
    ttyp = np.array([0, 1] * 20, np.int64)
    tri_snap = Snapshot(positions=tpos, type_ids=ttyp, cell=cell, pbc=True,
                        orig_type_labels=["1", "2"])

    ok1 = check("open-box molecule", open_snap)
    ok2 = check("triclinic crystal (tilted, small cell)", tri_snap)
    print(f"\n{'PASS' if ok1 and ok2 else 'FAIL'}")
    sys.exit(0 if ok1 and ok2 else 1)


if __name__ == "__main__":
    main()
