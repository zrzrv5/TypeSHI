"""Shared inference utilities: model loading, multi-frame/TTA/ensemble prediction."""

from __future__ import annotations

import numpy as np
import torch
from ase.data import chemical_symbols

from .augment import jitter
from .data import features_to_batch
from .decode import CompositionPrior
from .descriptors import average_features, compute_features_capped
from .model import TypeSetClassifier


def load_models(ckpts: list[str]) -> list[TypeSetClassifier]:
    models = []
    for c in ckpts:
        m = TypeSetClassifier.load_from_checkpoint(c, map_location="cpu")
        m.eval()
        models.append(m)
    return models


def fuse_logp(logps: torch.Tensor, mode: str = "logmean") -> torch.Tensor:
    """Fuse per-model log-probs (M, ..., 94) -> (..., 94).

    logmean:  geometric mean of probs (log-pool). Consensus-seeking: one model
              assigning ~0 vetoes a class, so specialist knowledge gets diluted
              (HfBaO-pog: Hf 99.6% solo -> rank 3 under the old always-logmean).
    probmean: arithmetic mean of probs. A lone confident model keeps >= p/M
              mass; ignorant-but-diffuse co-members cannot veto.
    conf:     probmean weighted per type by each model's own confidence
              (softmax over models of 4*maxprob) -- lets specialists dominate
              exactly where they are sure.
    median:   elementwise median of log-probs (outlier-robust consensus).
    """
    if mode == "logmean":
        out = logps.mean(0)
    elif mode == "probmean":
        out = torch.logsumexp(logps, 0) - np.log(logps.shape[0])
    elif mode == "conf":
        w = (4.0 * logps.exp().amax(-1)).softmax(0)          # (M, ..., )
        out = torch.log((w[..., None] * logps.exp()).sum(0) + 1e-12)
    elif mode == "median":
        out = logps.median(0).values
    else:
        raise ValueError(f"unknown fuse mode {mode!r}")
    return out.log_softmax(-1)


def predict(models, snaps, tta: int = 0, prior: CompositionPrior | None = None,
            debias: float = 0.0, fuse: str = "conf", env_draws: int = 4):
    """Fused per-type element probabilities (T, 94).

    debias: undo the training-time class weighting by gamma in [0,1]
    (weighted CE tilts the posterior toward up-weighted rare classes;
    logp_adj = logp - gamma*log(w_class)).
    fuse: cross-MODEL fusion rule (see fuse_logp). Frames/TTA variants are
    always log-pooled within a model (time-averaging is consensus by nature).
    env_draws: env sets are a 16-atom sample per type -- a high-variance
    statistic that can flip near-tie predictions between runs; log-pooling
    over several draws removes that inference-time lottery.
    """
    if tta:
        rng = np.random.default_rng(0)
        snaps = list(snaps) + [jitter(s, rng, sigma_max=0.15)
                               for s in snaps for _ in range(tta)]
    # env sets cost little and are ignored by models without use_env;
    # huge periodic systems get the interior-crop cap (bounded memory)
    feats = [compute_features_capped(s, with_env=True,
                                     rng=np.random.default_rng(1000 + d))
             for s in snaps for d in range(max(1, env_draws))]
    if len(feats) > 1:
        feats.append(average_features(feats))
    per_model = []
    for m in models:
        logp = [torch.log(m.predict_probs(features_to_batch(f))[0] + 1e-12)
                for f in feats]
        per_model.append(torch.stack(logp).mean(0))          # frame log-pool
    mean_logp = fuse_logp(torch.stack(per_model), fuse)
    if debias:
        w = models[0].class_weights.clamp(min=1e-6)
        mean_logp = mean_logp - debias * torch.log(w)[None, :]
    if prior is not None:
        probs = torch.from_numpy(
            prior.marginals(mean_logp.numpy(), snaps[0].type_fractions()))
    else:
        probs = torch.exp(mean_logp)
    return probs / probs.sum(-1, keepdim=True)


def report(name, snap, probs, truth: dict[str, str], k_report=5, quiet=False):
    """Print per-type ranking vs truth; return (hits@{1,3,5}, n_scored)."""
    hits = {1: 0, 3: 0, 5: 0}
    n = 0
    if not quiet:
        print(f"\n== {name} ==")
    for t, label in enumerate(snap.orig_type_labels):
        true_el = truth.get(label)
        if true_el is None:
            continue
        n += 1
        order = torch.argsort(probs[t], descending=True)
        symbols = [chemical_symbols[z + 1] for z in order.tolist()]
        rank = symbols.index(true_el) + 1
        for k in hits:
            hits[k] += rank <= k
        if not quiet:
            top = ", ".join(f"{s} {probs[t, order[i]]:.1%}"
                            for i, s in enumerate(symbols[:k_report]))
            mark = "OK " if rank == 1 else f"r={rank}"
            print(f"  type {label} (true {true_el:>2}) [{mark}]: {top}")
    return hits, n
