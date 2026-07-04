"""E18: does a LEARNED joint decoder (masked discrete diffusion / discrete flow
matching) beat the production INDEPENDENT-softmax head -- with or without the
hand-built composition beam -- on the SAME trunk and the SAME data?

Trains a baseline (model.TypeSetClassifier) and a diffusion decoder
(model_diffusion.DiffusionTypeClassifier) from scratch on an identical shard
subset for identical epochs, then evaluates both on a fixed val subset with a
common top-k routine:

  A baseline marginal            per-type softmax (production head, no coupling)
  B baseline + composition beam  A re-coupled with the PMI/charge-neutrality prior
  C diffusion joint decode       confidence-ordered iterative unmasking
  D diffusion marginal           single all-masked shot (trunk-only sanity)

Usage:
  uv run python scripts/exp_diffusion.py --epochs 10 --max-shards 25 --eval-structs 4000
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader

from typeid2elem.data import ShardDataset, collate
from typeid2elem.decode import CompositionPrior
from typeid2elem.model import TypeSetClassifier
from typeid2elem.model_diffusion import DiffusionTypeClassifier


def subset_dir(specs: list[str], max_shards: int) -> str:
    """Symlink the first `max_shards` npz from each 'dir[:n]' spec into a tmp dir."""
    d = tempfile.mkdtemp(prefix="e18_")
    for spec in specs:
        src = Path(spec)
        n = max_shards
        files = sorted(src.glob("*.npz"))[:n]
        for f in files:
            os.symlink(f.resolve(), Path(d) / f"{src.name}__{f.name}")
    return d


def train_one(model, train_ds, val_ds, args, name):
    dl = dict(batch_size=args.batch_size, collate_fn=collate,
              num_workers=args.workers, pin_memory=True,
              persistent_workers=args.workers > 0)
    trainer = L.Trainer(
        max_epochs=args.epochs, accelerator="gpu", precision="bf16-mixed",
        gradient_clip_val=1.0, default_root_dir=f"runs/{name}",
        logger=False, enable_checkpointing=False,
        enable_progress_bar=False, log_every_n_steps=50)
    trainer.fit(model,
                DataLoader(train_ds, shuffle=True, drop_last=True, **dl),
                DataLoader(val_ds, shuffle=False, **dl))
    return model.cuda().eval()


@torch.no_grad()
def evaluate(baseline, diff, val_ds, prior, cap, batch_size):
    dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                    collate_fn=collate, num_workers=4)
    keys = ["A_base", "B_base_decode", "C_diff_decode", "D_diff_marg"]
    hits = {kk: {1: 0, 3: 0, 5: 0} for kk in keys}
    n = 0
    seen = 0
    for batch in dl:
        b = {k: v.cuda() for k, v in batch.items()}
        labels = b["labels"]
        valid = labels != -100
        # A: baseline marginal
        pA = baseline.predict_probs(b)                       # (B,T,94)
        # C/D: diffusion
        ctx, mask = diff(b)
        pC = diff.decode_posteriors(ctx, mask)
        pD = diff.marginal_posteriors(ctx, mask)

        def acc(p, key):
            probs = p[valid]
            tgt = labels[valid][:, None]
            tk = probs.topk(5, -1).indices
            for k in (1, 3, 5):
                hits[key][k] += int((tk[:, :k] == tgt).any(-1).sum())

        acc(pA, "A_base")
        acc(pC, "C_diff_decode")
        acc(pD, "D_diff_marg")

        # B: baseline + composition beam (per-structure numpy; the coupling
        # the production pipeline actually ships)
        logp = torch.log(pA + 1e-12).cpu().numpy()
        m = mask.cpu().numpy()
        lab = labels.cpu().numpy()
        for i in range(len(lab)):
            t = int(m[i].sum())
            fr = batch["frac"][i, :t].numpy().astype(np.float64)
            fr = fr / max(fr.sum(), 1e-9)
            pb = prior.marginals(logp[i, :t], fr)            # (t,94)
            for j in range(t):
                if lab[i, j] == -100:
                    continue
                order = np.argsort(-pb[j])
                rank = int(np.where(order == lab[i, j])[0][0]) + 1
                for k in (1, 3, 5):
                    hits["B_base_decode"][k] += rank <= k

        n += int(valid.sum())
        seen += len(lab)
        if seen >= cap:
            break
    return hits, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+",
                    default=["Data/processed/mptrj_v3", "Data/processed/cod_v3"])
    ap.add_argument("--max-shards", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-structs", type=int, default=4000)
    args = ap.parse_args()

    torch.set_float32_matmul_precision("high")
    sd = subset_dir(args.dirs, args.max_shards)
    train_ds = ShardDataset([sd], "train")
    val_ds = ShardDataset([sd], "val")
    print(f"[data] train={len(train_ds)} val={len(val_ds)} from {sd}")

    counts = train_ds.class_counts()
    w = 1.0 / np.sqrt(np.maximum(counts, 1))
    w *= (counts > 0)
    w = w / w[w > 0].mean()
    w = np.clip(w, 0.0, 10.0)

    torch.manual_seed(0)
    base = TypeSetClassifier(class_weights=w, epochs=args.epochs, use_env=False)
    print("[train] baseline (independent softmax head)")
    base = train_one(base, train_ds, val_ds, args, "e18_base")

    torch.manual_seed(0)
    diff = DiffusionTypeClassifier(class_weights=w, epochs=args.epochs,
                                   use_env=False)
    print("[train] diffusion (masked discrete-diffusion joint decoder)")
    diff = train_one(diff, train_ds, val_ds, args, "e18_diff")

    prior = CompositionPrior()
    hits, n = evaluate(base, diff, val_ds, prior, args.eval_structs,
                       args.batch_size)
    print(f"\n=== E18 results (val, {n} type-ids) ===")
    print(f"{'method':<26}{'top-1':>8}{'top-3':>8}{'top-5':>8}")
    labels = {
        "A_base": "A baseline marginal",
        "B_base_decode": "B baseline + PMI beam",
        "C_diff_decode": "C diffusion joint decode",
        "D_diff_marg": "D diffusion marginal",
    }
    for kk in ["A_base", "B_base_decode", "C_diff_decode", "D_diff_marg"]:
        h = hits[kk]
        print(f"{labels[kk]:<26}"
              + "".join(f"{h[k] / max(n, 1):>7.1%} " for k in (1, 3, 5)))


if __name__ == "__main__":
    main()
