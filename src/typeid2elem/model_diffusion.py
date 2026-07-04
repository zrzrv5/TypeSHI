"""Discrete-diffusion (absorbing/masked) decoder over the joint type-label vector.

EXPERIMENTAL (docs/EXPERIMENTS.md E18). The production model scores every type id
with an INDEPENDENT softmax and couples them afterwards with a hand-built
composition beam (PMI + charge neutrality). This variant instead learns the joint
p(y_1..y_T | descriptors) directly with a masked discrete-diffusion decoder on top
of the *identical* descriptor trunk (model.TypeSetClassifier.encode) -- the only
thing that changes is the head, so any accuracy delta is attributable to modelling
the joint rather than to a bigger/different backbone.

Masked (absorbing-state) diffusion == discrete flow matching under the masking
interpolant (Campbell 2024), so this single model stands in for both families.

Training: mask each type independently with prob t~U(0,1); predict the masked
labels from the trunk context + the *unmasked* sibling labels; MDLM 1/t weighting.
Inference: start all-masked, confidence-order unmask one type at a time, recording
each type's predictive distribution at the step it is unmasked (its posterior
conditioned on already-decided siblings). That per-type distribution is what we
rank for top-k -- the honest "diffusion posterior".
"""

from __future__ import annotations

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .data import N_CLASSES
from .model import TypeSetClassifier

MASK_ID = N_CLASSES  # absorbing state = extra embedding row


class DiffusionTypeClassifier(L.LightningModule):
    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 n_denoise: int = 2, lr: float = 3e-4, weight_decay: float = 1e-2,
                 class_weights: np.ndarray | None = None, epochs: int = 20,
                 use_env: bool = False, t_eps: float = 0.02):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        # identical descriptor trunk as the baseline (its .head goes unused)
        self.trunk = TypeSetClassifier(
            d_model=d_model, n_heads=n_heads, use_env=use_env,
            class_weights=class_weights)
        self.label_embed = nn.Embedding(N_CLASSES + 1, d_model)  # +1 = MASK
        dec = nn.TransformerEncoderLayer(
            d_model, n_heads, 4 * d_model, dropout=0.1,
            batch_first=True, norm_first=True, activation="gelu")
        self.denoiser = nn.TransformerEncoder(dec, n_denoise)
        self.head = nn.Linear(d_model, N_CLASSES)
        w = torch.ones(N_CLASSES) if class_weights is None else torch.as_tensor(
            class_weights, dtype=torch.float32)
        self.register_buffer("class_weights", w)

    def _denoise(self, ctx, y_in, mask):
        """ctx: (B,T,d) trunk context; y_in: (B,T) labels w/ MASK_ID; -> logits."""
        tok = ctx + self.label_embed(y_in)
        h = self.denoiser(tok, src_key_padding_mask=~mask)
        return self.head(h)

    def forward(self, batch):
        ctx, mask = self.trunk.encode(batch)
        return ctx, mask

    def _step(self, batch, stage):
        ctx, mask = self(batch)
        labels = batch["labels"]                       # (B,T), -100 pad
        B, T = mask.shape
        valid = labels != -100
        y = labels.clamp(min=0)
        t = torch.rand(B, 1, device=ctx.device).clamp(self.hparams.t_eps, 1.0)
        drop = (torch.rand(B, T, device=ctx.device) < t) & valid
        # guarantee at least one masked target per sample (else no loss signal)
        no_drop = ~(drop & valid).any(1)
        if no_drop.any():
            first = valid.float().argmax(1)
            drop[no_drop, first[no_drop]] = True
        y_in = torch.where(drop, torch.full_like(y, MASK_ID), y)
        logits = self._denoise(ctx, y_in, mask)
        tgt = torch.where(drop & valid, labels, torch.full_like(labels, -100))
        per = F.cross_entropy(
            logits.reshape(-1, N_CLASSES), tgt.reshape(-1),
            weight=self.class_weights, ignore_index=-100, reduction="none"
        ).reshape(B, T)
        # MDLM continuous-time weight 1/t on masked positions
        wmask = (tgt != -100).float()
        wt = (1.0 / t).expand(B, T) * wmask
        loss = (per * wt).sum() / wt.sum().clamp(min=1)

        with torch.no_grad():
            # cheap monitor: single-shot all-masked marginal top-k
            self._log_topk(ctx, mask, labels, stage)
        self.log(f"{stage}/loss", loss, prog_bar=True,
                 batch_size=int(valid.sum()), sync_dist=stage == "val")
        return loss

    @torch.no_grad()
    def _log_topk(self, ctx, mask, labels, stage):
        post = self.decode_posteriors(ctx, mask)       # (B,T,94) iterative
        valid = labels != -100
        probs = post[valid]
        tgt = labels[valid][:, None]
        topk = probs.topk(5, -1).indices
        bs = int(valid.sum())
        for k in (1, 3, 5):
            acc = (topk[:, :k] == tgt).any(-1).float().mean()
            self.log(f"{stage}/top{k}", acc, prog_bar=(k in (1, 3)),
                     batch_size=bs, sync_dist=stage == "val")

    @torch.no_grad()
    def decode_posteriors(self, ctx, mask, greedy: bool = True):
        """Confidence-ordered iterative unmasking.

        Returns (B,T,94): for each type, the predictive distribution at the step
        it was unmasked (conditioned on the siblings decided before it).
        """
        B, T = mask.shape
        y = torch.full((B, T), MASK_ID, device=ctx.device, dtype=torch.long)
        remaining = mask.clone()
        post = torch.zeros(B, T, N_CLASSES, device=ctx.device)
        bidx = torch.arange(B, device=ctx.device)
        for _ in range(T):
            still = remaining.any(1)
            if not still.any():
                break
            logits = self._denoise(ctx, y, mask)
            probs = logits.softmax(-1)
            conf = probs.amax(-1).masked_fill(~remaining, -1.0)
            pick = conf.argmax(1)                      # (B,) most-confident type
            picked = probs[bidx, pick]                 # (B,94)
            upd = still                                # only rows with work left
            post[bidx[upd], pick[upd]] = picked[upd]
            y[bidx[upd], pick[upd]] = picked[upd].argmax(-1)
            remaining[bidx[upd], pick[upd]] = False
        return post

    @torch.no_grad()
    def marginal_posteriors(self, ctx, mask):
        """Single all-masked forward = per-type marginal (no sibling coupling)."""
        y = torch.full(mask.shape, MASK_ID, device=ctx.device, dtype=torch.long)
        return self._denoise(ctx, y, mask).softmax(-1)

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        total = max(int(self.trainer.estimated_stepping_batches), 10)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.hparams.lr, total_steps=total,
            pct_start=min(0.3, max(0.05, 2.0 / total)))
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"}}
