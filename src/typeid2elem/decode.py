"""Composition-aware joint decoding over per-type element probabilities.

Re-ranks joint assignments (one element per type id) by:
  score = sum_t log p_model(e_t)
        + w_pmi  * mean pairwise PMI(e_i, e_j) over the DISTINCT element set
        + w_neut * charge-neutrality feasibility bonus (fraction-weighted)

Duplicate elements across type ids are allowed (split types are legitimate).
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

N_EL = 94

# Common oxidation states per Z (compact; 0 always allowed is NOT included --
# metals in alloys are handled by the |imbalance| soft penalty instead).
_OX: dict[int, tuple[int, ...]] = {
    1: (1, -1), 2: (0,), 3: (1,), 4: (2,), 5: (3,), 6: (4, 2, -4), 7: (-3, 3, 5),
    8: (-2,), 9: (-1,), 10: (0,), 11: (1,), 12: (2,), 13: (3,), 14: (4, -4),
    15: (5, 3, -3), 16: (-2, 4, 6), 17: (-1, 5, 7), 18: (0,), 19: (1,), 20: (2,),
    21: (3,), 22: (4, 3, 2), 23: (5, 4, 3, 2), 24: (3, 6, 2), 25: (2, 3, 4, 7),
    26: (2, 3), 27: (2, 3), 28: (2,), 29: (2, 1), 30: (2,), 31: (3,), 32: (4, 2),
    33: (3, 5, -3), 34: (-2, 4, 6), 35: (-1, 5), 36: (0,), 37: (1,), 38: (2,),
    39: (3,), 40: (4,), 41: (5, 3), 42: (4, 6, 3), 43: (4, 7), 44: (3, 4),
    45: (3,), 46: (2, 4), 47: (1,), 48: (2,), 49: (3, 1), 50: (2, 4),
    51: (3, 5), 52: (-2, 4, 6), 53: (-1, 5, 7), 54: (0,), 55: (1,), 56: (2,),
    57: (3,), 58: (3, 4), 59: (3,), 60: (3,), 61: (3,), 62: (3, 2), 63: (2, 3),
    64: (3,), 65: (3,), 66: (3,), 67: (3,), 68: (3,), 69: (3,), 70: (3, 2),
    71: (3,), 72: (4,), 73: (5,), 74: (4, 6), 75: (4, 7), 76: (4, 3),
    77: (3, 4), 78: (2, 4), 79: (1, 3), 80: (2, 1), 81: (1, 3), 82: (2, 4),
    83: (3, 5), 84: (-2, 2, 4), 85: (-1,), 86: (0,), 87: (1,), 88: (2,),
    89: (3,), 90: (4,), 91: (5,), 92: (4, 6), 93: (5, 4), 94: (4, 3),
}


class CompositionPrior:
    def __init__(self, costats_path: str | Path | None = None,
                 w_pmi: float = 0.5, w_neut: float = 1.5, alpha: float = 2.0):
        if costats_path is None:
            from .assets import costats_npz
            costats_path = costats_npz()
        z = np.load(costats_path)
        n = float(z["n_sets"])
        p1 = (z["single"] + alpha) / (n + alpha * N_EL)
        p2 = (z["pair"] + alpha) / (n + alpha * N_EL**2)
        self.pmi = np.log(p2) - np.log(p1[:, None]) - np.log(p1[None, :])
        self.w_pmi = w_pmi
        self.w_neut = w_neut

    def _neutrality(self, zs: tuple[int, ...], frac: np.ndarray) -> float:
        """Min |sum_t x_t * q_t| over allowed oxidation states; 0 = neutral possible."""
        best = np.inf
        for qs in itertools.product(*[_OX.get(z, (0,)) for z in zs]):
            best = min(best, abs(float(np.dot(frac, qs))))
            if best < 1e-9:
                return 0.0
        return best

    def score_extra(self, zs: tuple[int, ...], frac: np.ndarray) -> float:
        els = sorted(set(zs))
        if len(els) > 1:
            pairs = [self.pmi[a - 1, b - 1]
                     for i, a in enumerate(els) for b in els[i + 1:]]
            pmi = float(np.mean(pairs))
        else:
            pmi = 0.0
        return self.w_pmi * pmi - self.w_neut * self._neutrality(zs, frac)

    def rerank(self, logp: np.ndarray, frac: np.ndarray, top_k: int = 10,
               beam: int = 300) -> list[tuple[float, tuple[int, ...]]]:
        """logp: (T, N_EL) per-type log-probs. Returns scored assignments (desc).

        Candidates per type = its top_k elements; beam search over types keeps
        `beam` partial assignments ranked by model log-prob, then the composition
        terms re-rank the complete assignments.
        """
        T = logp.shape[0]
        cand = np.argsort(logp, axis=1)[:, ::-1][:, :top_k] + 1   # Z candidates
        partials: list[tuple[float, tuple[int, ...]]] = [(0.0, ())]
        for t in range(T):
            nxt = [(s + float(logp[t, z - 1]), zs + (int(z),))
                   for s, zs in partials for z in cand[t]]
            nxt.sort(key=lambda x: -x[0])
            partials = nxt[:beam]
        scored = [(s + self.score_extra(zs, frac), zs) for s, zs in partials]
        scored.sort(key=lambda x: -x[0])
        return scored

    def marginals(self, logp: np.ndarray, frac: np.ndarray,
                  top_k: int = 10, beam: int = 300) -> np.ndarray:
        """Per-type marginal probs from softmax over re-ranked joint assignments."""
        scored = self.rerank(logp, frac, top_k, beam)
        scores = np.array([s for s, _ in scored])
        w = np.exp(scores - scores.max())
        w /= w.sum()
        T = logp.shape[0]
        out = np.full((T, N_EL), 1e-12)
        for wi, (_, zs) in zip(w, scored):
            for t, z in enumerate(zs):
                out[t, z - 1] += wi
        return out / out.sum(1, keepdims=True)
