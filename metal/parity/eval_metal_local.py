"""Verify the Metal descriptor path on the REAL eval files (not just the synthetic
golden): for each committed eval case, dump Metal GPU descriptors via the Swift
`desc-dump` binary, diff the deterministic descriptors against the Python
reference, then push the Metal descriptors through the deploy ONNX + decode and
compare top-1 to the pure-Python path and to ground truth.

Prereq:  swift build --package-path metal   (builds .build/debug/desc-dump)
Usage:   <verify-venv>/bin/python metal/parity/eval_metal_local.py
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

from ase.data import chemical_symbols  # noqa: E402

from eval_lite_local import BENCH, load_case  # noqa: E402  (reuse the case list)
sys.path.insert(0, str(REPO / "metal/parity"))
from onnx_bridge import run_onnx  # noqa: E402

from typeid2elem.assets import deploy_onnx  # noqa: E402
from typeid2elem.decode import CompositionPrior  # noqa: E402
from typeid2elem.descriptors import compute_features  # noqa: E402

DESC_DUMP = REPO / "metal/.build/debug/desc-dump"
NB, NPX, MENV, KENV = 64, 8, 16, 16


def metal_descriptors(snap, seed=0):
    with tempfile.TemporaryDirectory() as td:
        inp, out = Path(td) / "in.json", Path(td) / "out.json"
        cell = snap.cell.tolist() if (snap.cell is not None and snap.pbc) else None
        inp.write_text(json.dumps({
            "positions": snap.positions.tolist(),
            "type_ids": [int(t) for t in snap.type_ids],
            "cell": cell,
        }))
        subprocess.run([str(DESC_DUMP), str(inp), str(out), str(seed)],
                       check=True, capture_output=True)
        d = json.loads(out.read_text())
    t = d["nTypes"]
    return {
        "rdf": np.array(d["rdf"], np.float32).reshape(t, t, NB),
        "pair_extra": np.array(d["pair_extra"], np.float32).reshape(t, t, NPX),
        "frac": np.array(d["frac"], np.float32),
        "glob": np.array(d["glob"], np.float32),
        "env_d": np.array(d["env_d"], np.float32).reshape(t, MENV, KENV),
        "env_t": np.array(d["env_t"], np.float32).reshape(t, MENV, KENV),
    }


def top1(feats, prior, onnx_path):
    logp = run_onnx(feats, onnx_path)
    marg = prior.marginals(logp, feats["frac"].astype(np.float64))
    return [chemical_symbols[int(np.argmax(marg[t])) + 1] for t in range(len(marg))]


def main() -> None:
    if not DESC_DUMP.exists():
        sys.exit(f"missing {DESC_DUMP}; run: swift build --package-path metal")
    prior = CompositionPrior()
    onnx_path = deploy_onnx()

    hdr = f"{'case':<22} {'n_at':>6} {'rdf Δ':>9} {'pe Δ':>9} {'glob Δ':>9} agree metalOK pyOK"
    print(hdr)
    agree_tot = 0
    n_tot = 0
    metal_hit = 0
    py_hit = 0
    for name, kind, path, truth in BENCH:
        snap, tmap = load_case(kind, path, truth)
        m = metal_descriptors(snap, seed=0)
        pf = compute_features(snap, with_env=True, rng=np.random.default_rng(0))

        rdf_d = float(np.abs(m["rdf"] - pf["rdf"]).max())
        pe_d = float(np.abs(m["pair_extra"] - pf["pair_extra"]).max())
        glob_d = float(np.abs(m["glob"] - pf["glob"]).max())

        m_top1 = top1(m, prior, onnx_path)
        p_top1 = top1(pf, prior, onnx_path)

        agree = n = mh = ph = 0
        for t, lab in enumerate(snap.orig_type_labels):
            true_el = tmap.get(lab)
            if true_el is None:
                continue
            n += 1
            agree += m_top1[t] == p_top1[t]
            mh += m_top1[t] == true_el
            ph += p_top1[t] == true_el
        agree_tot += agree
        n_tot += n
        metal_hit += mh
        py_hit += ph
        print(f"{name:<22} {len(snap.type_ids):>6} {rdf_d:>9.2e} {pe_d:>9.2e} "
              f"{glob_d:>9.2e} {agree}/{n}  {mh}/{n}   {ph}/{n}")

    print(f"\nMetal vs Python top-1 agreement: {agree_tot}/{n_tot} "
          f"({agree_tot/n_tot:.1%})")
    print(f"Metal path top-1 vs truth : {metal_hit}/{n_tot} ({metal_hit/n_tot:.0%})")
    print(f"Python path top-1 vs truth: {py_hit}/{n_tot} ({py_hit/n_tot:.0%})")


if __name__ == "__main__":
    main()
