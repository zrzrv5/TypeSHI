"""Physics-verifier reranking: score candidate element assignments with a
universal ML interatomic potential (MACE-MP-0).

Rationale: with the RIGHT species, the observed geometry is physically
plausible -- near-zero forces for relaxed structures, thermal-scale forces for
MD frames. Wrong species (e.g. Bi on Li sites) produce huge restoring forces.
This is a best-of-N verifier, no RL training required.
"""

from __future__ import annotations

import numpy as np
import torch
from ase import Atoms

from .decode import CompositionPrior
from .io import Snapshot

_CALC = None


def _calc():
    global _CALC
    if _CALC is None:
        from mace.calculators import mace_mp
        _CALC = mace_mp(model="small", device="cuda" if torch.cuda.is_available()
                        else "cpu", default_dtype="float32")
    return _CALC


def force_score(snap: Snapshot, zs: tuple[int, ...],
                max_atoms: int = 1500) -> float:
    """RMS force (eV/A) of the hypothesized species at the observed geometry.

    Large systems are scored on a random subsample re-wrapped in the same cell
    (cheap approximation: we keep the full cell, forces of missing neighbors
    perturb all candidates equally).
    """
    numbers = np.array([zs[t] for t in snap.type_ids])
    pos, nums = snap.positions, numbers
    if len(pos) > max_atoms:
        idx = np.random.default_rng(0).choice(len(pos), max_atoms, replace=False)
        pos, nums = pos[idx], nums[idx]
    atoms = Atoms(numbers=nums, positions=pos,
                  cell=snap.cell if snap.cell is not None else None,
                  pbc=snap.cell is not None and snap.pbc)
    atoms.calc = _calc()
    try:
        f = atoms.get_forces()
        if not np.isfinite(f).all():
            return 1e3
        return float(np.sqrt((f ** 2).sum(1).mean()))
    except Exception:
        return 1e3


def verify_rerank(logp: np.ndarray, snap: Snapshot, prior: CompositionPrior,
                  n_candidates: int = 12, lam: float = 3.0,
                  f0: float = 1.0) -> np.ndarray:
    """Re-rank the prior's top joint assignments by model logp + force plausibility.

    Returns per-type marginals over the verified candidate set (softmax of
    combined scores), shape (T, 94).
    """
    scored = prior.rerank(logp, snap.type_fractions())[:n_candidates]
    combined = []
    for s, zs in scored:
        frms = force_score(snap, zs)
        combined.append((s - lam * np.log1p(frms / f0), zs, frms))
    scores = np.array([c[0] for c in combined])
    w = np.exp(scores - scores.max())
    w /= w.sum()
    T = logp.shape[0]
    out = np.full((T, logp.shape[1]), 1e-12)
    for wi, (_, zs, _) in zip(w, combined):
        for t, z in enumerate(zs):
            out[t, z - 1] += wi
    return out / out.sum(1, keepdims=True)
