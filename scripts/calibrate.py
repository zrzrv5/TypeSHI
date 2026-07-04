"""Calibrate the production pipeline: temperature scaling + RAPS conformal sets.

Runs the production ensemble (conf fusion + composition decode, i.e. exactly
what predict.py outputs) on a sample of the val split (gid%10==0 element
sets), fits a temperature by NLL, then computes the RAPS conformal quantile
so that "keep candidates until cumulative score >= qhat" covers the true
element with the requested rate. Saves runs/production/calib.npz.

RAPS score of the true class = randomizer-free cumulative probability mass
down the sorted list + lam * max(0, rank - k_reg) (Angelopoulos et al. 2021)
-- the penalty keeps tail classes from inflating sets.

Usage:
  uv run python scripts/calibrate.py --data Data/processed/cosmos_v3 \
      Data/processed/mptrj_v3 Data/processed/openlam_v3 --sample 8000 \
      --coverage 0.9
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from typeid2elem.data import ShardDataset, collate
from typeid2elem.decode import CompositionPrior
from typeid2elem.inference import fuse_logp, load_models

PROD = sorted(str(p) for p in Path("runs/production").glob("*.ckpt"))


def pipeline_probs(models, prior, batch, fuse="conf"):
    """(B, T, 94) final pipeline probabilities + validity mask."""
    logp = fuse_logp(torch.stack(
        [m(batch).log_softmax(-1) for m in models]), fuse)
    probs = logp.exp().cpu().numpy()
    out = np.zeros_like(probs)
    mask = batch["mask"].cpu().numpy()
    frac = batch["frac"].cpu().numpy()
    for b in range(len(probs)):
        t = int(mask[b].sum())
        lp = np.log(probs[b, :t] + 1e-12)
        out[b, :t] = (prior.marginals(lp, frac[b, :t])
                      if prior is not None else probs[b, :t])
    return out, mask


def raps_score(probs, label, lam, k_reg):
    """RAPS conformity score of the true label under sorted probs."""
    order = np.argsort(-probs)
    rank = int(np.where(order == label)[0][0])          # 0-based
    cum = probs[order][:rank + 1].sum()
    return cum + lam * max(0, rank + 1 - k_reg)


def raps_set(probs, qhat, lam, k_reg, max_size=94):
    order = np.argsort(-probs)
    cum, out = 0.0, []
    for i, cls in enumerate(order[:max_size]):
        cum += probs[cls]
        out.append(int(cls))
        if cum + lam * max(0, i + 1 - k_reg) >= qhat:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--ckpt", nargs="+", default=PROD)
    ap.add_argument("--sample", type=int, default=8000)
    ap.add_argument("--coverage", type=float, default=0.9)
    ap.add_argument("--lam", type=float, default=0.02)
    ap.add_argument("--k-reg", type=int, default=4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--no-decode", action="store_true")
    args = ap.parse_args()

    ds = ShardDataset(args.data, "val")
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(ds))[:args.sample]
    models = [m.cuda() for m in load_models(args.ckpt)]
    prior = None if args.no_decode else CompositionPrior()

    # gather final-pipeline probabilities + labels
    all_p, all_y = [], []
    with torch.no_grad():
        for i in range(0, len(idx), args.batch):
            items = [ds[j] for j in idx[i:i + args.batch]]
            batch = collate(items)
            labels = batch.pop("labels").numpy()
            batch = {k: v.cuda() for k, v in batch.items()}
            probs, mask = pipeline_probs(models, prior, batch)
            for b in range(len(probs)):
                t = int(mask[b].sum())
                for a in range(t):
                    if labels[b, a] >= 0:
                        all_p.append(probs[b, a])
                        all_y.append(labels[b, a])
    P = np.stack(all_p)                                  # (N, 94)
    Y = np.array(all_y)
    print(f"calibration points: {len(Y)}")

    # temperature on the final probs (grid; NLL)
    logp = np.log(P + 1e-12)
    temps = np.linspace(0.5, 8.0, 76)
    nll = []
    for T in temps:
        z = logp / T
        z = z - z.max(-1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(-1, keepdims=True)
        nll.append(-np.log(p[np.arange(len(Y)), Y] + 1e-12).mean())
    T_best = float(temps[int(np.argmin(nll))])
    z = logp / T_best
    z = z - z.max(-1, keepdims=True)
    P_cal = np.exp(z) / np.exp(z).sum(-1, keepdims=True)
    print(f"temperature = {T_best:.2f} "
          f"(NLL {min(nll):.3f} vs {nll[np.searchsorted(temps, 1.0)]:.3f} at T=1)")

    # RAPS quantile
    scores = np.array([raps_score(P_cal[i], Y[i], args.lam, args.k_reg)
                       for i in range(len(Y))])
    n = len(scores)
    q = np.quantile(scores, min(1.0, np.ceil((n + 1) * args.coverage) / n),
                    method="higher")
    sets = [raps_set(P_cal[i], q, args.lam, args.k_reg) for i in range(len(Y))]
    sizes = np.array([len(s) for s in sets])
    cov = np.mean([Y[i] in sets[i] for i in range(len(Y))])
    print(f"RAPS qhat = {q:.4f} -> empirical coverage {cov:.1%}, "
          f"set size median {np.median(sizes):.0f} / mean {sizes.mean():.1f} / "
          f"p90 {np.percentile(sizes, 90):.0f}")

    out = Path("runs/production/calib.npz")
    np.savez(out, temperature=T_best, qhat=q, lam=args.lam, k_reg=args.k_reg,
             coverage=args.coverage, n_cal=n,
             decode=not args.no_decode)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
