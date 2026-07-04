"""Length-unit inference + "is this even atomistic?" gate.

MD/DEM files store raw numbers; the unit is convention (LAMMPS `units` styles:
real/metal=Angstrom, electron=Bohr, nano=nm, micro=um, cgs=cm, si=m). Our
model deliberately reasons in ABSOLUTE Angstroms (bond lengths are the
signal), so a wrong unit is catastrophic and a granular/DEM file (mm-scale
spheres) is out of scope entirely.

Detection: nearest-neighbor distances of atomistic matter live in a hard
physical window (~0.7-3.5 A; nothing bonded exists below H2's 0.74 A and no
condensed phase has NN spacing much beyond van-der-Waals contacts). For each
candidate unit, rescale and score the fraction of atoms whose NN distance
lands in that window. No candidate fits -> not atoms (the DEM verdict).

Ambiguity: the window alone cannot split Angstrom from Bohr (a 2.0 A bond
misread as Bohr-valued is 1.06 A -- still window-plausible). Near-ties are
broken by running the classifier under each surviving candidate and keeping
the unit whose geometry the model finds most recognizable (mean top-1
confidence) -- absolute-distance training makes this discriminative.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.spatial import cKDTree

from .io import Snapshot

# candidate length units -> factor converting stored values to Angstrom
UNIT_TO_ANG = {
    "angstrom": 1.0,           # LAMMPS real/metal, most DFT outputs
    "bohr": 0.529177,          # LAMMPS electron, many QM codes
    "nm": 10.0,                # LAMMPS nano, GROMACS
    "um": 1.0e4,               # LAMMPS micro
    "cm": 1.0e8,               # LAMMPS cgs
    "m": 1.0e10,               # LAMMPS si
}
NN_WINDOW = (0.6, 3.6)         # Angstrom; generous bond/contact window
SCORE_ACCEPT = 0.60            # min in-window fraction to call it atomistic
TIE_MARGIN = 0.15              # candidates within this of best -> model tiebreak
ANG_PRIOR_MARGIN = 0.30        # non-Angstrom must beat Angstrom's model
                               # confidence by this much when Angstrom is
                               # window-plausible: almost all real files are A,
                               # and distorted-but-valid A structures (e.g.
                               # RMC fits) can look "more familiar" to the
                               # model under a wrong rescaling
MAX_SAMPLE = 2000


def _nn_distances(snap: Snapshot, rng: np.random.Generator) -> np.ndarray:
    """Per-atom nearest-neighbor distance in STORED units (no pbc: for bulk
    cells interior atoms dominate the sample; edge effects only pad the tail)."""
    pos = snap.positions
    if len(pos) > MAX_SAMPLE:
        # query a sample against the full tree
        idx = rng.choice(len(pos), MAX_SAMPLE, replace=False)
        query = pos[idx]
    else:
        query = pos
    if len(pos) < 2:
        return np.array([])
    tree = cKDTree(pos)
    d, _ = tree.query(query, k=2)
    return d[:, 1]


def _rescaled(snap: Snapshot, scale: float) -> Snapshot:
    return replace(
        snap,
        positions=snap.positions * scale,
        cell=None if snap.cell is None else snap.cell * scale,
    )


def infer_units(snap: Snapshot, models=None, conf_fn=None) -> dict:
    """Detect the length unit of a snapshot, or that it is not atomistic.

    Returns dict(verdict="atoms"|"not_atoms", unit, scale, score, candidates,
    tiebreak). On "atoms", apply `scale` to positions/cell (or use
    `rescaled_snapshot`) before descriptor computation.

    Ambiguity tiebreak uses `conf_fn(snapshot) -> float` (mean top-1 model
    confidence under a candidate rescaling); when only torch `models` are
    given, a conf_fn is built from them. With neither, ties resolve to the
    first candidate in UNIT_TO_ANG order (i.e. Angstrom wins ties).
    """
    rng = np.random.default_rng(0)
    d_nn = _nn_distances(snap, rng)
    if len(d_nn) == 0:
        return dict(verdict="not_atoms", unit=None, scale=None, score=0.0,
                    candidates={}, tiebreak=None,
                    reason="fewer than 2 atoms")

    scores = {}
    for unit, s in UNIT_TO_ANG.items():
        d = d_nn * s
        scores[unit] = float(np.mean((d >= NN_WINDOW[0]) & (d <= NN_WINDOW[1])))
    best = max(scores, key=scores.get)

    if scores[best] < SCORE_ACCEPT:
        return dict(verdict="not_atoms", unit=None, scale=None,
                    score=scores[best], candidates=scores, tiebreak=None,
                    reason=(f"no length unit puts nearest-neighbor distances in "
                            f"the {NN_WINDOW} A bond window (best: {best} at "
                            f"{scores[best]:.0%}); this does not look like "
                            f"atomistic data (median NN = {np.median(d_nn):.4g} "
                            f"stored units)"))

    contenders = [u for u, sc in scores.items()
                  if scores[best] - sc <= TIE_MARGIN]
    tiebreak = None
    if conf_fn is None and models is not None:
        def conf_fn(s):
            from .data import features_to_batch
            from .descriptors import compute_features
            import torch

            batch = features_to_batch(compute_features(s, with_env=True))
            with torch.no_grad():
                p = torch.stack([m.predict_probs(batch)[0] for m in models])
            return float(p.mean(0).amax(-1).mean())
    if len(contenders) > 1 and conf_fn is not None:
        conf = {}
        for u in contenders:
            try:
                conf[u] = float(conf_fn(_rescaled(snap, UNIT_TO_ANG[u])))
            except Exception:
                conf[u] = 0.0
        best = max(conf, key=conf.get)
        if ("angstrom" in contenders and best != "angstrom"
                and conf[best] - conf["angstrom"] < ANG_PRIOR_MARGIN):
            best = "angstrom"
        tiebreak = {"mode": "model_confidence", "confidence": conf}

    return dict(verdict="atoms", unit=best, scale=UNIT_TO_ANG[best],
                score=scores[best], candidates=scores, tiebreak=tiebreak)


def rescaled_snapshot(snap: Snapshot, result: dict) -> Snapshot:
    """Apply an infer_units 'atoms' result; identity for Angstrom input."""
    if result["verdict"] != "atoms" or result["scale"] == 1.0:
        return snap
    return _rescaled(snap, result["scale"])
