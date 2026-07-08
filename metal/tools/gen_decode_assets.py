"""Bake the composition-decode + conformal assets into a bundled JSON for the
on-device (Swift) path.

The desktop path loads weights/costats.npz (element co-occurrence counts) and
weights/calib.npz (conformal temperature + RAPS qhat) at runtime. iOS/macOS can't
read .npz easily, so this precomputes the PMI matrix exactly as
`typeid2elem.decode.CompositionPrior` (alpha=2 smoothing) and emits it, together
with the calibration scalars, as a compact JSON resource compiled into the Swift
package.

Regenerate whenever costats.npz / calib.npz change:
    <verify-venv>/bin/python metal/tools/gen_decode_assets.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
N_EL = 94
ALPHA = 2.0  # must match decode.CompositionPrior default


def main() -> None:
    co = np.load(REPO / "weights/costats.npz")
    n = float(co["n_sets"])
    p1 = (co["single"] + ALPHA) / (n + ALPHA * N_EL)
    p2 = (co["pair"] + ALPHA) / (n + ALPHA * N_EL**2)
    pmi = np.log(p2) - np.log(p1[:, None]) - np.log(p1[None, :])  # (94, 94)

    cal = np.load(REPO / "weights/calib.npz")
    out = {
        "n_el": N_EL,
        "alpha": ALPHA,
        "pmi": [round(float(x), 6) for x in pmi.reshape(-1)],  # row-major (94*94)
        "calib": {
            "temperature": float(cal["temperature"]),
            "qhat": float(cal["qhat"]),
            "lam": float(cal["lam"]),
            "k_reg": int(cal["k_reg"]),
            "coverage": float(cal["coverage"]),
        },
    }
    dst = REPO / "metal/Sources/InaccurateRDFCalculator/Resources/decode_assets.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {dst}  ({dst.stat().st_size/1024:.0f} KB)")
    print(f"pmi range [{pmi.min():.3f}, {pmi.max():.3f}], calib={out['calib']}")


if __name__ == "__main__":
    main()
