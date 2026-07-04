"""Minimal-footprint element prediction: ONNX Runtime, no torch/lightning.

Runs the single exported production model (default: env_codfull_sharp30.onnx,
which matches the full 6-model torch ensemble on the real-file bench) with
the same feature pipeline, unit gate, composition decode, and conformal sets
as predict.py. Dependency surface: numpy, scipy, matscipy, ase (element
tables + optional readers), onnxruntime -- no torch, no lightning.

Usage:
  uv run python scripts/predict_lite.py <file> [--top-k 5] [--onnx path]
      [--format lammps-data|ase] [--no-decode] [--conformal]
      [--units auto|angstrom|bohr|...] [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ase.data import chemical_symbols  # noqa: E402

from typeid2elem.decode import CompositionPrior  # noqa: E402
from typeid2elem.descriptors import compute_features_capped  # noqa: E402
from typeid2elem.io import mass_lookup, read_lammps_data  # noqa: E402
from typeid2elem.units import UNIT_TO_ANG, infer_units, rescaled_snapshot  # noqa: E402

from typeid2elem.assets import calib_npz, deploy_onnx  # noqa: E402

T_FIXED = 8
DEFAULT_ONNX = deploy_onnx()


class OnnxPredictor:
    def __init__(self, onnx_path: str):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(onnx_path,
                                         providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.sess.get_inputs()]

    def logprobs(self, feats: dict) -> np.ndarray:
        """Padded fixed-shape single-sample inference -> (T_real, 94) logp."""
        t = len(feats["frac"])
        T = T_FIXED
        rdf = np.zeros((1, T, T, feats["rdf"].shape[-1]), np.float32)
        pe = np.zeros((1, T, T, feats["pair_extra"].shape[-1]), np.float32)
        frac = np.zeros((1, T), np.float32)
        mask = np.zeros((1, T), np.float32)
        rdf[0, :t, :t] = feats["rdf"]
        pe[0, :t, :t] = feats["pair_extra"]
        frac[0, :t] = feats["frac"]
        mask[0, :t] = 1.0
        inputs = {"rdf": rdf, "pair_extra": pe, "frac": frac,
                  "glob": feats["glob"][None].astype(np.float32), "mask": mask}
        if "env_d" in self.input_names:
            m, k = feats["env_d"].shape[-2:]
            env_d = np.zeros((1, T, m, k), np.float32)
            env_t = np.full((1, T, m, k), -1.0, np.float32)
            env_d[0, :t] = feats["env_d"]
            env_t[0, :t] = feats["env_t"].astype(np.float32)
            inputs |= {"env_d": env_d, "env_t": env_t}
        out = self.sess.run(None, inputs)[0]                # (1, 8, 94) logp
        return out[0, :t]

    def mean_conf(self, snap) -> float:
        lp = self.logprobs(compute_features_capped(snap))
        return float(np.exp(lp).max(-1).mean())


def raps_set(probs, qhat, lam, k_reg):
    order = np.argsort(-probs)
    cum = 0.0
    out = []
    for i, cls in enumerate(order):
        cum += probs[cls]
        out.append(int(cls))
        if cum + lam * max(0, i + 1 - k_reg) >= qhat:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--onnx", default=DEFAULT_ONNX)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--format", default="lammps-data", choices=["lammps-data", "ase"])
    ap.add_argument("--no-decode", action="store_true")
    ap.add_argument("--conformal", action="store_true")
    ap.add_argument("--units", default="auto",
                    choices=["auto"] + list(UNIT_TO_ANG))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    snaps = []
    for p in args.files:
        if args.format == "lammps-data":
            snaps.append(read_lammps_data(p))
        else:
            from ase.io import read

            from typeid2elem.io import snapshot_from_atoms
            snaps.append(snapshot_from_atoms(read(p))[0])

    model = OnnxPredictor(args.onnx)

    if args.units == "auto":
        res = infer_units(snaps[0], conf_fn=model.mean_conf)
        if res["verdict"] == "not_atoms":
            print(f"\nNOT ATOMISTIC: {res['reason']}")
            if not args.force:
                raise SystemExit(2)
        else:
            print(f"\nInferred length unit: {res['unit']} "
                  f"(NN in-window {res['score']:.0%})")
            snaps = [rescaled_snapshot(s, res) for s in snaps]
    elif UNIT_TO_ANG[args.units] != 1.0:
        snaps = [rescaled_snapshot(x, dict(verdict="atoms",
                                           scale=UNIT_TO_ANG[args.units]))
                 for x in snaps]

    # multi-frame + multi-env-draw: mean of log-probs (predict.py convention;
    # env sets are 16-atom samples -- pooling draws removes near-tie flips)
    logp = np.mean([model.logprobs(compute_features_capped(
                        s, rng=np.random.default_rng(1000 + d)))
                    for s in snaps for d in range(4)], axis=0)

    ref = snaps[0]
    if args.no_decode:
        probs = np.exp(logp)
        probs /= probs.sum(-1, keepdims=True)
    else:
        prior = CompositionPrior()
        probs = prior.marginals(logp, ref.type_fractions())

    calib = None
    if args.conformal:
        cpath = Path(calib_npz())
        if cpath.exists():
            calib = dict(np.load(cpath).items())
        else:
            print("(no calib.npz -- conformal sets skipped)")

    print(f"\nSystem: {len(ref.type_ids)} atoms, {ref.n_types} types, "
          f"{len(snaps)} frame(s), single ONNX model"
          f"{'' if args.no_decode else ', composition decode'}")
    for t in range(ref.n_types):
        order = np.argsort(-probs[t])[:args.top_k]
        cand = ", ".join(f"{chemical_symbols[z + 1]} {probs[t, z]:.1%}"
                         for z in order)
        line = f"  type {ref.orig_type_labels[t]}: {cand}"
        if t in ref.type_masses:
            line += f"   [mass baseline: {mass_lookup(ref.type_masses[t], 1)[0][0]}]"
        print(line)
        if calib is not None:
            z = np.log(probs[t] + 1e-12) / calib["temperature"]
            z -= z.max()
            p = np.exp(z) / np.exp(z).sum()
            s = raps_set(p, calib["qhat"], calib["lam"], int(calib["k_reg"]))
            syms = [chemical_symbols[c + 1] for c in s]
            print(f"      {calib['coverage']:.0%}-coverage set "
                  f"({len(s)}): {{{', '.join(syms)}}}")


if __name__ == "__main__":
    main()
