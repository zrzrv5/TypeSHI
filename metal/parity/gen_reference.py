"""Generate a golden reference for the Metal descriptor port.

Runs on any platform (no Metal needed). Builds a small, fully-specified structure,
computes the reference descriptors with the production Python path
(`typeid2elem.descriptors.compute_features`), runs the committed int8 ONNX model,
and writes `golden.json`:

  input:    positions / type_ids / cell        (the Swift test builds the same arrays)
  expected: rdf, pair_extra, frac, glob         (DETERMINISTIC — the descriptor
                                                 kernel must reproduce these)
  env:      env_d, env_t for seed 0             (SAMPLED — structural check only;
                                                 a different RNG gives a different
                                                 but equivalent sample)
  log_probs, top1                               (full-pipeline check through CoreML)

The Metal `InaccurateRDFCalculator` is allowed to *approximate* the RDF (subsampled
centers, coarse neighbor search); "pass" means top-1 per type matches and the
90% conformal set is unchanged, NOT bit-exact descriptors. Tolerances live in the
metal/ README.

Regenerate:  uv run python metal/parity/gen_reference.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from typeid2elem.assets import deploy_onnx
from typeid2elem.descriptors import compute_features
from typeid2elem.io import Snapshot

T_FIXED = 8            # exported CoreML/ONNX fixed type dimension
A = 4.2                # rocksalt lattice constant (Angstrom); NN = A/2 = 2.1 A
NREP = 2               # 2x2x2 conventional cells -> 64 atoms, both types present

# rocksalt (B1): type-0 FCC + type-1 FCC offset by (0.5,0,0) of the conv. cell
_FCC = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
_BASIS = [(0, _FCC), (1, _FCC + np.array([0.5, 0, 0]))]


def build_structure() -> Snapshot:
    pos, typ = [], []
    for i in range(NREP):
        for j in range(NREP):
            for k in range(NREP):
                shift = np.array([i, j, k], float)
                for t, basis in _BASIS:
                    for b in basis:
                        pos.append((b + shift) * A)
                        typ.append(t)
    cell = np.eye(3) * (A * NREP)
    return Snapshot(
        positions=np.array(pos, float),
        type_ids=np.array(typ, np.int64),
        cell=cell, pbc=True,
        orig_type_labels=["1", "2"],
    )


def onnx_logprobs(feats: dict) -> np.ndarray:
    import onnxruntime as ort
    sess = ort.InferenceSession(deploy_onnx(),
                                providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    t = len(feats["frac"])
    T, nb, npx = T_FIXED, feats["rdf"].shape[-1], feats["pair_extra"].shape[-1]
    rdf = np.zeros((1, T, T, nb), np.float32)
    pe = np.zeros((1, T, T, npx), np.float32)
    frac = np.zeros((1, T), np.float32)
    mask = np.zeros((1, T), np.float32)
    rdf[0, :t, :t] = feats["rdf"]
    pe[0, :t, :t] = feats["pair_extra"]
    frac[0, :t] = feats["frac"]
    mask[0, :t] = 1.0
    inputs = {"rdf": rdf, "pair_extra": pe, "frac": frac,
              "glob": feats["glob"][None].astype(np.float32), "mask": mask}
    if "env_d" in names:
        m, k = feats["env_d"].shape[-2:]
        env_d = np.zeros((1, T, m, k), np.float32)
        env_t = np.full((1, T, m, k), -1.0, np.float32)
        env_d[0, :t] = feats["env_d"]
        env_t[0, :t] = feats["env_t"].astype(np.float32)
        inputs |= {"env_d": env_d, "env_t": env_t}
    return sess.run(None, inputs)[0][0, :t]


def main():
    from ase.data import chemical_symbols
    snap = build_structure()
    feats = compute_features(snap, with_env=True, rng=np.random.default_rng(0))
    logp = onnx_logprobs(feats)
    top1 = [chemical_symbols[int(np.argmax(logp[t])) + 1]
            for t in range(len(logp))]

    def r(a, n=6):
        return np.round(np.asarray(a), n).tolist()

    out = {
        "meta": {
            "structure": f"rocksalt B1, a={A} A, {NREP}x{NREP}x{NREP} conv cells",
            "n_atoms": int(len(snap.positions)),
            "n_types": int(snap.n_types),
            "onnx": Path(deploy_onnx()).name,
            "constants": {"R_MAX": 8.0, "N_BINS": 64, "CN_RADII": [2, 3, 4, 6],
                          "M_ENV": 16, "K_ENV": 16, "R_ENV": 6.0,
                          "T_FIXED": T_FIXED},
        },
        "input": {
            "positions": r(snap.positions, 6),
            "type_ids": snap.type_ids.tolist(),
            "cell": r(snap.cell, 6),
        },
        "expected": {                       # deterministic — kernel must match
            "rdf": r(feats["rdf"], 5),
            "pair_extra": r(feats["pair_extra"], 5),
            "frac": r(feats["frac"], 6),
            "glob": r(feats["glob"], 6),
        },
        "env_seed0": {                      # sampled — structural check only
            "env_d": r(feats["env_d"], 5),
            "env_t": feats["env_t"].astype(int).tolist(),
        },
        "log_probs": r(logp, 5),
        "top1_per_type": top1,
    }
    dst = Path(__file__).with_name("golden.json")
    dst.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {dst}  ({dst.stat().st_size/1024:.0f} KB)")
    print(f"structure: {out['meta']['structure']}, {out['meta']['n_atoms']} atoms")
    print(f"top-1 per type: {top1}")


if __name__ == "__main__":
    main()
