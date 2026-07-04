"""TypeSHI programmatic API: arrays in, ranked element hypotheses out.

The one function most integrations need:

    from typeid2elem.api import predict_structure

    result = predict_structure(positions, type_ids, cell=cell)
    for t in result["types"]:
        print(t["label"], t["candidates"][:3])

No file I/O, no CLI -- give it what an MD engine already has in memory:
positions (N, 3), per-atom type ids (N,), optional 3x3 cell (row vectors).
Everything else (unit inference, descriptor computation, ensemble, composition
decode, conformal sets) happens inside.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from .io import Snapshot
from .units import infer_units, rescaled_snapshot

_REPO = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=2)
def _torch_models(ckpt_key: tuple):
    from .inference import load_models
    return load_models(list(ckpt_key))


@lru_cache(maxsize=2)
def _calib(path_str: str):
    p = Path(path_str)
    return dict(np.load(p).items()) if p.exists() else None


def _raps_set(probs, qhat, lam, k_reg):
    order = np.argsort(-probs)
    cum = 0.0
    out = []
    for i, cls in enumerate(order):
        cum += probs[cls]
        out.append(int(cls))
        if cum + lam * max(0, i + 1 - k_reg) >= qhat:
            break
    return out


def predict_structure(
    positions,
    type_ids,
    cell=None,
    *,
    k: int = 5,
    ckpts: tuple | None = None,
    decode: bool = True,
    units: str = "auto",
    conformal: bool = True,
    env_draws: int = 4,
) -> dict:
    """Rank element candidates for every type id in one structure.

    Args:
        positions: (N, 3) float array. Any length unit (see `units`).
        type_ids: (N,) int or str array of per-atom type labels; arbitrary
            values, need not be contiguous or start at 1.
        cell: optional (3, 3) float array of cell ROW vectors (ASE/LAMMPS
            convention). None for molecules/clusters.
        k: candidates returned per type.
        ckpts: checkpoint paths for the ensemble; default = runs/production/.
        decode: apply the composition prior (co-occurrence + charge
            neutrality); recommended.
        units: "auto" infers the length unit and refuses non-atomistic input;
            or one of typeid2elem.units.UNIT_TO_ANG ("angstrom", "bohr", ...).
        conformal: attach 90%-coverage candidate sets when
            runs/production/calib.npz exists.
        env_draws: environment-set draws pooled per prediction (variance
            control; 1 = fastest, 4 = default).

    Returns dict:
        verdict: "atoms" | "not_atoms" (if not_atoms: reason, and no types)
        unit / scale: inferred or given length unit
        types: list of {label, candidates: [(symbol, prob), ...],
                        set90: [symbols] or None}
    """
    from ase.data import chemical_symbols

    from .decode import CompositionPrior
    from .inference import predict as _predict
    from .units import UNIT_TO_ANG

    positions = np.asarray(positions, dtype=float)
    labels, inverse = np.unique(np.asarray(type_ids), return_inverse=True)
    snap = Snapshot(
        positions=positions,
        type_ids=inverse.astype(np.int64),
        cell=None if cell is None else np.asarray(cell, dtype=float),
        pbc=cell is not None,
        orig_type_labels=[str(l) for l in labels],
    )

    if ckpts is None:
        from .assets import production_ckpts
        ckpts = production_ckpts()
    if not ckpts:
        raise SystemExit(
            "no checkpoints found under runs/production/ — download the weights "
            "tarball from the GitHub Release, or pass ckpts=[...] explicitly. "
            "(The no-torch lite path scripts/predict_lite.py needs only "
            "weights/env_codfull_sharp30.int8.onnx.)")
    models = _torch_models(tuple(ckpts))

    out: dict = {}
    if units == "auto":
        res = infer_units(snap, models=models)
        if res["verdict"] == "not_atoms":
            return {"verdict": "not_atoms", "reason": res["reason"],
                    "candidates_scores": res["candidates"], "types": []}
        snap = rescaled_snapshot(snap, res)
        out |= {"unit": res["unit"], "scale": res["scale"]}
    else:
        s = UNIT_TO_ANG[units]
        if s != 1.0:
            snap = rescaled_snapshot(snap, {"verdict": "atoms", "scale": s})
        out |= {"unit": units, "scale": s}

    prior = CompositionPrior() if decode else None
    probs = _predict(models, [snap], prior=prior,
                     env_draws=env_draws).numpy()

    if conformal:
        from .assets import calib_npz
        calib = _calib(calib_npz())
    else:
        calib = None
    types = []
    for t, label in enumerate(snap.orig_type_labels):
        order = np.argsort(-probs[t])[:k]
        cands = [(chemical_symbols[z + 1], float(probs[t, z])) for z in order]
        set90 = None
        if calib is not None:
            z = np.log(probs[t] + 1e-12) / calib["temperature"]
            z -= z.max()
            p = np.exp(z) / np.exp(z).sum()
            set90 = [chemical_symbols[c + 1] for c in
                     _raps_set(p, calib["qhat"], calib["lam"],
                               int(calib["k_reg"]))]
        types.append({"label": label, "candidates": cands, "set90": set90})
    return out | {"verdict": "atoms", "types": types}
