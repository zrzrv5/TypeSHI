"""Train the type-set classifier.

Usage:
  uv run python scripts/train.py --data Data/processed/cosmos_dbs [Data/processed/mptrj] \
      --name run1 --epochs 20 --batch-size 512
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader


def _load_wandb_key():
    if "WANDB_API_KEY" in os.environ:
        return True
    keyfile = Path(__file__).resolve().parent.parent / "WANDB_API_KEY"
    if keyfile.exists():
        os.environ["WANDB_API_KEY"] = keyfile.read_text().strip()
        return True
    return False

from typeid2elem.data import ShardDataset, collate
from typeid2elem.model import TypeSetClassifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--name", default="run")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--two-pass", action="store_true")
    ap.add_argument("--use-env", action="store_true",
                    help="learned environment encoder (needs v3 shards with env_*)")
    ap.add_argument("--init-ckpt", default=None,
                    help="fine-tune: initialize weights from this checkpoint "
                         "(fresh optimizer/schedule; pair with low --lr)")
    args = ap.parse_args()

    torch.set_float32_matmul_precision("high")
    train_ds = ShardDataset(args.data, "train")
    val_ds = ShardDataset(args.data, "val")
    print(f"train={len(train_ds)} val={len(val_ds)}")
    if args.use_env and train_ds.items[0][5] is None:
        raise SystemExit("--use-env requires v3 shards (env_d/env_t arrays)")

    counts = train_ds.class_counts()
    weights = 1.0 / np.sqrt(np.maximum(counts, 1))
    weights *= (counts > 0)                       # never reward absent classes
    weights = weights / weights[weights > 0].mean()
    weights = np.clip(weights, 0.0, 10.0)

    model = TypeSetClassifier(lr=args.lr, d_model=args.d_model,
                              class_weights=weights, epochs=args.epochs,
                              two_pass=args.two_pass, use_env=args.use_env)
    if args.init_ckpt:
        state = torch.load(args.init_ckpt, map_location="cpu",
                           weights_only=False)["state_dict"]
        state["class_weights"] = model.class_weights  # keep new data's weights
        model.load_state_dict(state)
        print(f"initialized from {args.init_ckpt}")

    dl_kw = dict(batch_size=args.batch_size, collate_fn=collate,
                 num_workers=args.workers, pin_memory=True,
                 persistent_workers=args.workers > 0)
    trainer = L.Trainer(
        max_epochs=args.epochs, accelerator="gpu", precision="bf16-mixed",
        gradient_clip_val=1.0,
        default_root_dir=f"runs/{args.name}",
        logger=[CSVLogger("runs", name=args.name)] + (
            [WandbLogger(project="typeid2elem", name=args.name,
                         save_dir=f"runs/{args.name}")]
            if _load_wandb_key() else []),
        callbacks=[
            ModelCheckpoint(monitor="val/top1", mode="max", save_top_k=1,
                            filename="best-{epoch}-{val/top1:.3f}"),
            LearningRateMonitor("step"),
        ],
        log_every_n_steps=50,
    )
    trainer.fit(model,
                DataLoader(train_ds, shuffle=True, drop_last=True, **dl_kw),
                DataLoader(val_ds, shuffle=False, **dl_kw))
    print("best:", trainer.checkpoint_callback.best_model_path,
          trainer.checkpoint_callback.best_model_score)


if __name__ == "__main__":
    main()
