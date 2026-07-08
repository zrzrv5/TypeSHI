"""Score a chosen model backend on the committed eval cases (Data/eval_cases/*),
same pipeline as scripts/predict_lite.py (4-env-draw pooling + composition decode).
Only the model swaps — descriptors + decode are identical across backends — so the
three runs are directly comparable.

Backends (run each in an env that has it):
  --backend onnx    [verifyenv]  int8 ONNX (weights/env_codfull_sharp30.int8.onnx)
  --backend coreml  [verifyenv]  CoreML .mlpackage via coremltools (fp16 deploy / fp32)
  --backend aimodel [coreaienv]  Core AI .aimodel via the coreai runtime (fp32)

Usage:
  <verifyenv>/bin/python  metal/parity/eval_backends_local.py --backend onnx
  <verifyenv>/bin/python  metal/parity/eval_backends_local.py --backend coreml  --model <mlpackage>
  <coreaienv>/bin/python  metal/parity/eval_backends_local.py --backend aimodel --model <aimodel>
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "metal/parity"))

from ase.data import chemical_symbols  # noqa: E402

from eval_lite_local import BENCH, load_case  # noqa: E402
from typeid2elem.decode import CompositionPrior  # noqa: E402
from typeid2elem.descriptors import compute_features_capped  # noqa: E402

T = 8
NB, NPX, MENV, KENV = 64, 8, 16, 16


def pad(feats: dict) -> dict:
    """Descriptors -> the 7 fixed-shape (T=8) model inputs (predict_lite convention)."""
    t = len(feats["frac"])
    rdf = np.zeros((1, T, T, NB), np.float32); rdf[0, :t, :t] = feats["rdf"]
    pe = np.zeros((1, T, T, NPX), np.float32); pe[0, :t, :t] = feats["pair_extra"]
    frac = np.zeros((1, T), np.float32); frac[0, :t] = feats["frac"]
    glob = feats["glob"][None].astype(np.float32)
    mask = np.zeros((1, T), np.float32); mask[0, :t] = 1.0
    env_d = np.zeros((1, T, MENV, KENV), np.float32); env_d[0, :t] = feats["env_d"]
    env_t = np.full((1, T, MENV, KENV), -1.0, np.float32); env_t[0, :t] = feats["env_t"]
    return {"rdf": rdf, "pair_extra": pe, "frac": frac, "glob": glob,
            "mask": mask, "env_d": env_d, "env_t": env_t}


class OnnxBackend:
    def __init__(self, path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.names = {i.name for i in self.sess.get_inputs()}

    def logprobs(self, feats):
        p = pad(feats)
        if "env_d" not in self.names:
            p = {k: v for k, v in p.items() if k not in ("env_d", "env_t")}
        return self.sess.run(None, p)[0][0, :len(feats["frac"])]


class CoreMLBackend:
    def __init__(self, path):
        import coremltools as ct
        self.m = ct.models.MLModel(path)

    def logprobs(self, feats):
        out = self.m.predict(pad(feats))["log_probs"]
        return np.asarray(out)[0, :len(feats["frac"])]


class AiModelBackend:
    def __init__(self, path):
        import asyncio
        from coreai import runtime as rt
        self.rt = rt
        self.loop = asyncio.new_event_loop()
        self.mdl = self.loop.run_until_complete(rt.AIModel.load(Path(path)))
        fn = self.mdl.load_function("main")
        self.fn = self.loop.run_until_complete(fn) if inspect.isawaitable(fn) else fn

    def logprobs(self, feats):
        p = pad(feats)
        r = self.loop.run_until_complete(self.fn({k: self.rt.NDArray(v) for k, v in p.items()}))
        return r["log_probs"].numpy()[0, :len(feats["frac"])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["onnx", "coreml", "aimodel"])
    ap.add_argument("--model", default=None, help="path to .onnx / .mlpackage / .aimodel")
    ap.add_argument("--no-decode", action="store_true")
    ap.add_argument("--draws", type=int, default=4)
    args = ap.parse_args()

    if args.backend == "onnx":
        from typeid2elem.assets import deploy_onnx
        backend = OnnxBackend(args.model or deploy_onnx())
    elif args.backend == "coreml":
        backend = CoreMLBackend(args.model)
    else:
        backend = AiModelBackend(args.model)
    prior = None if args.no_decode else CompositionPrior()

    tot = {1: 0, 3: 0, 5: 0}
    n_tot = 0
    print(f"backend={args.backend}  model={Path(args.model).name if args.model else 'deploy_onnx'}  "
          f"draws={args.draws}  decode={not args.no_decode}\n")
    print(f"{'case':<22} top1 top3 top5   n   misses")
    for name, kind, path, truth in BENCH:
        snap, tmap = load_case(kind, path, truth)
        feats4 = [compute_features_capped(snap, rng=np.random.default_rng(1000 + d))
                  for d in range(args.draws)]
        logp = np.mean([backend.logprobs(f) for f in feats4], axis=0)
        probs = (prior.marginals(logp, snap.type_fractions()) if prior is not None
                 else np.exp(logp))
        h = {1: 0, 3: 0, 5: 0}; n = 0; misses = []
        for t, lab in enumerate(snap.orig_type_labels):
            true_el = tmap.get(lab)
            if true_el is None:
                continue
            n += 1
            order = [chemical_symbols[z + 1] for z in np.argsort(-probs[t])]
            rank = order.index(true_el) + 1
            for k in h:
                h[k] += rank <= k
            if rank > 1:
                misses.append(f"{true_el}(r{rank}:{order[0]})")
        for k in tot:
            tot[k] += h[k]
        n_tot += n
        print(f"{name:<22} {h[1]:>4} {h[3]:>4} {h[5]:>4} {n:>3}   {' '.join(misses)}")
    print(f"\n[{args.backend}] TOTAL  top1 {tot[1]}/{n_tot} ({tot[1]/n_tot:.0%})  "
          f"top3 {tot[3]}/{n_tot} ({tot[3]/n_tot:.0%})  "
          f"top5 {tot[5]}/{n_tot} ({tot[5]/n_tot:.0%})")


if __name__ == "__main__":
    main()
