"""Golden reference for the Swift composition-decode + conformal port.

Loads golden.json's model `log_probs` (deterministic, already committed) and runs
the Python CompositionPrior.marginals + RAPS conformal set on them, then writes
decode_golden.json. The Swift ParityCheck reproduces these from the SAME log_probs
and diffs — an on-device unit test for Decode.swift that needs no CoreML model.

To keep the comparison exact, the PMI matrix and calibration are taken from the
bundled decode_assets.json (what Swift actually loads), not recomputed — so any
rounding in the bake is shared by both sides.

Regenerate:  <verify-venv>/bin/python metal/parity/gen_decode_golden.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from typeid2elem.decode import CompositionPrior  # noqa: E402


def raps_set(probs, qhat, lam, k_reg):
    order = np.argsort(-probs)
    cum, out = 0.0, []
    for i, cls in enumerate(order):
        cum += probs[cls]
        out.append(int(cls))
        if cum + lam * max(0, i + 1 - k_reg) >= qhat:
            break
    return out


def main() -> None:
    golden = json.loads((REPO / "metal/parity/golden.json").read_text())
    logp = np.array(golden["log_probs"], np.float64)          # (T, 94)
    frac = np.array(golden["expected"]["frac"], np.float64)   # (T,)

    assets = json.loads(
        (REPO / "metal/Sources/InaccurateRDFCalculator/Resources/decode_assets.json").read_text())
    pmi = np.array(assets["pmi"], np.float64).reshape(94, 94)
    cal = assets["calib"]

    prior = CompositionPrior()          # w_pmi=0.5, w_neut=1.5 defaults (match Swift)
    prior.pmi = pmi                     # use the baked (shared) PMI

    marg = prior.marginals(logp, frac)  # (T, 94), defaults top_k=10 beam=300

    conf = []
    for t in range(len(marg)):
        z = np.log(marg[t] + 1e-12) / cal["temperature"]
        z -= z.max()
        p = np.exp(z) / np.exp(z).sum()
        conf.append(sorted(raps_set(p, cal["qhat"], cal["lam"], int(cal["k_reg"]))))

    out = {
        "note": "Python CompositionPrior.marginals + RAPS on golden.log_probs; "
                "Swift ParityCheck must reproduce.",
        "marginals": np.round(marg, 8).tolist(),   # (T, 94)
        "conformal": conf,                          # per type: sorted class ids (Z-1)
    }
    dst = REPO / "metal/parity/decode_golden.json"
    dst.write_text(json.dumps(out, separators=(",", ":")))
    top1 = [int(np.argmax(m)) + 1 for m in marg]
    print(f"wrote {dst}")
    print(f"decoded top-1 (Z): {top1}")
    print(f"conformal set sizes: {[len(c) for c in conf]}")


if __name__ == "__main__":
    main()
