"""Score the lite (ONNX) predictor on the real-file bench.

Reuses BENCH/load_case from evaluate_bench and the OnnxPredictor from
predict_lite; reports hits@{1,3,5} for a given .onnx (fp32 or int8).

Usage:
  uv run python scripts/evaluate_lite.py runs/export/env_codfull_sharp30.onnx [--no-decode]
"""

from __future__ import annotations

import argparse

import numpy as np

from evaluate_bench import BENCH, load_case
from predict_lite import OnnxPredictor
from typeid2elem.decode import CompositionPrior
from typeid2elem.descriptors import compute_features
from ase.data import chemical_symbols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("--no-decode", action="store_true")
    args = ap.parse_args()
    model = OnnxPredictor(args.onnx)
    prior = None if args.no_decode else CompositionPrior()

    tot = {1: 0, 3: 0, 5: 0}
    n_tot = 0
    for name, kind, path, truth in BENCH:
        try:
            snap, truth_map = load_case(kind, path, truth)
        except Exception as e:
            print(f"{name}: LOAD ERROR {e}")
            continue
        logp = model.logprobs(compute_features(snap, with_env=True))
        if prior is not None:
            probs = prior.marginals(logp, snap.type_fractions())
        else:
            probs = np.exp(logp)
        for t, lab in enumerate(snap.orig_type_labels):
            true_el = truth_map.get(lab)
            if true_el is None:
                continue
            n_tot += 1
            order = np.argsort(-probs[t])
            rank = [chemical_symbols[z + 1] for z in order].index(true_el) + 1
            for k in tot:
                tot[k] += rank <= k
    print(f"lite bench: top1 {tot[1]}/{n_tot} top3 {tot[3]}/{n_tot} "
          f"top5 {tot[5]}/{n_tot}")


if __name__ == "__main__":
    main()
