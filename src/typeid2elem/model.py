"""Type-set classifier: pair-channel DeepSets encoder + type-token attention.

Each type-id token is built by pooling its pair channels (partial RDF to every
other type), then 2 self-attention blocks let types exchange information
(mutual exclusivity, composition context), then a linear head scores 94 elements.
"""

from __future__ import annotations

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .data import N_CLASSES
from .preprocess import FEATURE_DIMS


def mlp(dims, act=nn.SiLU):
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.LayerNorm(b), act()]
    return nn.Sequential(*layers)


class TypeSetClassifier(L.LightningModule):
    def __init__(self, d_pair: int = 128, d_model: int = 256, n_heads: int = 8,
                 n_blocks: int = 2, lr: float = 3e-4, weight_decay: float = 1e-2,
                 class_weights: np.ndarray | None = None, epochs: int = 20,
                 two_pass: bool = False, use_env: bool = False,
                 d_env: int = 64, n_rbf: int = 16):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        nb, npx = FEATURE_DIMS["n_bins"], FEATURE_DIMS["n_pair_extra"]
        pair_in = nb + npx + 2  # + partner fraction + is_self flag

        self.phi = mlp([pair_in, 2 * d_pair, d_pair])
        token_in = 2 * d_pair + 1 + FEATURE_DIMS["n_glob"]  # mean&max pool + frac_a + glob
        self.proj = mlp([token_in, d_model, d_model])
        if use_env:
            # learned species-blind environment encoder: per sampled atom,
            # neighbors are (RBF(distance), partner-type token) -> DeepSets over
            # neighbors -> DeepSets over sampled atoms -> additive token update.
            # Keeps per-atom JOINT coordination that pair-marginal RDFs discard.
            from .descriptors import R_ENV
            self.register_buffer(
                "rbf_centers", torch.linspace(0.0, R_ENV, n_rbf))
            self.rbf_gamma = 0.5 / (R_ENV / n_rbf) ** 2
            self.partner_proj = nn.Linear(d_model, d_env)
            self.env_nbr = mlp([n_rbf + d_env, 2 * d_env, d_env])
            self.env_atom = mlp([d_env, d_env])
            self.env_out = nn.Linear(d_env, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model, n_heads, 4 * d_model, dropout=0.1,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc, n_blocks)
        self.head = nn.Linear(d_model, N_CLASSES)
        if two_pass:
            # soft element-embedding feedback for iterative refinement
            self.elem_embed = nn.Linear(N_CLASSES, d_model, bias=False)

        w = torch.ones(N_CLASSES) if class_weights is None else torch.as_tensor(
            class_weights, dtype=torch.float32)
        self.register_buffer("class_weights", w)

    def encode(self, batch):
        """Descriptor trunk: features -> encoded per-type tokens (B, T, d_model).

        Everything up to (and including) the type-token transformer, without the
        classification head. Exposed so alternative decoders (e.g. the discrete
        diffusion head in model_diffusion.py) can reuse the identical trunk.
        """
        rdf, pe = batch["rdf"], batch["pair_extra"]
        frac, glob, mask = batch["frac"], batch["glob"], batch["mask"]
        B, T = frac.shape
        eye = torch.eye(T, device=rdf.device).expand(B, T, T)
        pair = torch.cat(
            [torch.log1p(rdf), pe, frac[:, None, :].expand(B, T, T)[..., None],
             eye[..., None]], dim=-1)
        h = self.phi(pair)                                   # (B, T, T, d_pair)

        pmask = mask[:, None, :, None]                       # valid partners
        h = h * pmask
        n_valid = mask.sum(1).clamp(min=1)[:, None, None].float()
        h_mean = h.sum(2) / n_valid
        h_max = h.masked_fill(~pmask.expand_as(h), -1e9).amax(2)
        tok = torch.cat(
            [h_mean, h_max, frac[..., None], glob[:, None, :].expand(B, T, -1)],
            dim=-1)
        tok = self.proj(tok)
        if self.hparams.get("use_env", False) and "env_d" in batch:
            ed, et = batch["env_d"], batch["env_t"]          # (B, T, M, K)
            valid = et >= 0
            idx = et.clamp(min=0)
            pemb = self.partner_proj(tok)                    # (B, T, d_env)
            g = torch.gather(
                pemb, 1,
                idx.reshape(B, -1, 1).expand(-1, -1, pemb.size(-1))
            ).reshape(*et.shape, -1)                         # (B, T, M, K, d_env)
            rbf = torch.exp(-self.rbf_gamma *
                            (ed[..., None] - self.rbf_centers) ** 2)
            hn = self.env_nbr(torch.cat([rbf, g], dim=-1)) * valid[..., None]
            atom = hn.sum(-2) / valid.sum(-1).clamp(min=1)[..., None]
            avalid = valid.any(-1)                           # sampled atom exists
            atom = self.env_atom(atom) * avalid[..., None]
            env = atom.sum(-2) / avalid.sum(-1).clamp(min=1)[..., None]
            tok = tok + self.env_out(env)
        return self.encoder(tok, src_key_padding_mask=~mask), mask

    def forward(self, batch):
        h, mask = self.encode(batch)
        tok = h                                              # for two-pass feedback
        logits1 = self.head(h)
        if not self.hparams.two_pass:
            return logits1
        # pass 2: condition every type token on the soft element identities of
        # ALL types (incl. itself), re-encode -- context disambiguates twins
        tok2 = tok + self.elem_embed(logits1.softmax(-1))
        h2 = self.encoder(tok2, src_key_padding_mask=~mask)
        logits2 = self.head(h2)
        return (logits1, logits2)

    def _step(self, batch, stage):
        out = self(batch)
        labels = batch["labels"]
        if isinstance(out, tuple):
            logits1, logits = out
            aux = F.cross_entropy(
                logits1.reshape(-1, N_CLASSES), labels.reshape(-1),
                weight=self.class_weights, ignore_index=-100)
        else:
            logits, aux = out, 0.0
        loss = 0.5 * aux + F.cross_entropy(
            logits.reshape(-1, N_CLASSES), labels.reshape(-1),
            weight=self.class_weights, ignore_index=-100)
        valid = labels != -100
        bs = int(valid.sum())
        with torch.no_grad():
            topk = logits[valid].topk(5, dim=-1).indices
            tgt = labels[valid][:, None]
            for k in (1, 3, 5):
                acc = (topk[:, :k] == tgt).any(-1).float().mean()
                self.log(f"{stage}/top{k}", acc, prog_bar=(k in (1, 3)),
                         batch_size=bs, sync_dist=stage == "val")
        self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=bs,
                 sync_dist=stage == "val")
        return loss

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

    @torch.no_grad()
    def predict_probs(self, batch) -> torch.Tensor:
        out = self(batch)
        if isinstance(out, tuple):
            out = out[1]
        return out.softmax(-1)
