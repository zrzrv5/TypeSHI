"""End-to-end Metal-path check without CoreML: run the Metal-computed descriptors
through the committed deploy ONNX and compare predictions to golden.json.

On-device the descriptors go Metal -> CoreML; CoreML can't run off an Apple
device and the .mlpackage is a Release artifact, so this substitutes the same
committed model gen_reference.py used (weights/env_codfull_sharp30.int8.onnx). If
top-1 per type matches golden with the *Metal* descriptors, the descriptor port
is validated all the way to predictions — the acceptance test in metal/README.md.

Usage:
    <verify-venv>/bin/python metal/parity/onnx_bridge.py <metal_dump.json> \
        [golden.json] [model.onnx]

`metal_dump.json` is produced by:  swift run parity-check golden.json --dump <path>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

T_FIXED = 8
NB, NPX, MENV, KENV = 64, 8, 16, 16


def run_onnx(feats: dict, onnx_path: str) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    t = len(feats["frac"])
    T = T_FIXED
    rdf = np.zeros((1, T, T, NB), np.float32)
    pe = np.zeros((1, T, T, NPX), np.float32)
    frac = np.zeros((1, T), np.float32)
    mask = np.zeros((1, T), np.float32)
    rdf[0, :t, :t] = feats["rdf"]
    pe[0, :t, :t] = feats["pair_extra"]
    frac[0, :t] = feats["frac"]
    mask[0, :t] = 1.0
    inputs = {"rdf": rdf, "pair_extra": pe, "frac": frac,
              "glob": feats["glob"][None].astype(np.float32), "mask": mask}
    if "env_d" in names:
        env_d = np.zeros((1, T, MENV, KENV), np.float32)
        env_t = np.full((1, T, MENV, KENV), -1.0, np.float32)
        env_d[0, :t] = feats["env_d"]
        env_t[0, :t] = feats["env_t"]
        inputs |= {"env_d": env_d, "env_t": env_t}
    return sess.run(None, inputs)[0][0, :t]


def main() -> None:
    from ase.data import chemical_symbols
    from typeid2elem.decode import CompositionPrior

    dump = json.loads(Path(sys.argv[1]).read_text())
    golden_path = sys.argv[2] if len(sys.argv) > 2 else str(REPO / "metal/parity/golden.json")
    onnx_path = sys.argv[3] if len(sys.argv) > 3 else None
    if onnx_path is None:
        from typeid2elem.assets import deploy_onnx
        onnx_path = deploy_onnx()
    golden = json.loads(Path(golden_path).read_text())

    t = dump["nTypes"]
    feats = {
        "rdf": np.array(dump["rdf"], np.float32).reshape(t, t, NB),
        "pair_extra": np.array(dump["pair_extra"], np.float32).reshape(t, t, NPX),
        "frac": np.array(dump["frac"], np.float32),
        "glob": np.array(dump["glob"], np.float32),
        "env_d": np.array(dump["env_d"], np.float32).reshape(t, MENV, KENV),
        "env_t": np.array(dump["env_t"], np.float32).reshape(t, MENV, KENV),
    }
    logp = run_onnx(feats, onnx_path)
    raw_top1 = [chemical_symbols[int(np.argmax(logp[i])) + 1] for i in range(t)]

    prior = CompositionPrior()
    marg = prior.marginals(logp, feats["frac"].astype(np.float64))
    dec_top1 = [chemical_symbols[int(np.argmax(marg[i])) + 1] for i in range(t)]

    ref = golden["top1_per_type"]
    ok = raw_top1 == ref
    print(f"model: {Path(onnx_path).name}")
    print(f"golden top-1     : {ref}")
    print(f"metal->onnx top-1: {raw_top1}   {'OK' if ok else 'MISMATCH'}")
    print(f"metal->onnx+decode: {dec_top1}")
    # also confirm the model log_probs from Metal descriptors track golden's
    ref_logp = np.array(golden["log_probs"], np.float32)
    print(f"log_probs max|Δ| vs golden: {np.abs(logp - ref_logp).max():.4g}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
