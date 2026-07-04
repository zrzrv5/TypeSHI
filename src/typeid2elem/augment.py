"""Train-time structure augmentations bridging clean DFT data -> messy MD reality."""

from __future__ import annotations

import numpy as np

from .io import Snapshot

MAX_TYPES = 8


def type_split(snap: Snapshot, labels: np.ndarray, rng: np.random.Generator,
               max_splits: int = 3) -> tuple[Snapshot, np.ndarray]:
    """Split one element's atoms into 2..max_splits type ids (same element label).

    Mimics real MD files where e.g. type 1 and 2 are both Si (different sublattices).
    """
    T = snap.n_types
    counts = np.bincount(snap.type_ids, minlength=T)
    splittable = np.where(counts >= 4)[0]
    if len(splittable) == 0 or T >= MAX_TYPES:
        return snap, labels
    target = rng.choice(splittable)
    m = int(rng.integers(2, min(max_splits, MAX_TYPES - T + 1) + 1))

    new_types = snap.type_ids.copy()
    idx = np.where(snap.type_ids == target)[0]
    if rng.random() < 0.5:
        # random partition with random (non-empty) group sizes
        order = rng.permutation(len(idx))
    else:
        # spatial partition (sublattice-ish): contiguous slabs along a random axis
        axis = rng.integers(0, 3)
        order = np.argsort(snap.positions[idx, axis])
    if rng.random() < 0.5:
        # heavily uneven sizes (Dirichlet, sparse): teaches dilute minority
        # type-ids like "5 of 128 Na atoms got their own type" (common in
        # hand-edited MD files) -- see docs/EXPERIMENTS.md E5/E6
        w = rng.dirichlet(np.full(m, 0.3))
        sizes = np.maximum(1, np.round(w * len(idx)).astype(int))
        while sizes.sum() > len(idx):
            sizes[np.argmax(sizes)] -= 1
        sizes[np.argmax(sizes)] += len(idx) - sizes.sum()
        cuts = np.cumsum(sizes)[:-1]
    else:
        cuts = np.sort(rng.choice(np.arange(1, len(idx)), size=m - 1, replace=False))
    # keep group 0 as `target`, append the rest as new type ids
    for k, chunk in enumerate(np.split(order, cuts)):
        if k > 0:
            new_types[idx[chunk]] = T + k - 1

    new_labels = np.concatenate([labels, np.full(m - 1, labels[target])])
    out = Snapshot(snap.positions, new_types, snap.cell, snap.pbc,
                   orig_type_labels=snap.orig_type_labels + [f"split{k}" for k in range(1, m)])
    return out, new_labels


def jitter(snap: Snapshot, rng: np.random.Generator, sigma_max: float = 0.35) -> Snapshot:
    """Gaussian thermal-like position noise, sigma ~ U(0.02, sigma_max) Angstrom."""
    sigma = rng.uniform(0.02, sigma_max)
    pos = snap.positions + rng.normal(0.0, sigma, snap.positions.shape)
    return Snapshot(pos, snap.type_ids, snap.cell, snap.pbc,
                    snap.type_masses, snap.orig_type_labels)


def strain(snap: Snapshot, rng: np.random.Generator, eps: float = 0.03) -> Snapshot:
    """Small random symmetric strain on cell + positions (NPT-like fluctuation)."""
    if snap.cell is None:
        return snap
    s = np.eye(3) + rng.uniform(-eps, eps, (3, 3))
    s = 0.5 * (s + s.T)
    return Snapshot(snap.positions @ s, snap.type_ids, np.asarray(snap.cell) @ s,
                    snap.pbc, snap.type_masses, snap.orig_type_labels)


def subsample_type(snap: Snapshot, rng: np.random.Generator) -> Snapshot:
    """Delete 50-95% of one type's atoms -> dilute type (dopant-like fraction)."""
    T = snap.n_types
    counts = np.bincount(snap.type_ids, minlength=T)
    eligible = np.where(counts >= 8)[0]
    if len(eligible) == 0:
        return snap
    target = rng.choice(eligible)
    idx = np.where(snap.type_ids == target)[0]
    keep_frac = rng.uniform(0.05, 0.5)
    n_keep = max(2, int(keep_frac * len(idx)))
    drop = rng.choice(idx, size=len(idx) - n_keep, replace=False)
    keep = np.setdiff1d(np.arange(len(snap.type_ids)), drop)
    return Snapshot(snap.positions[keep], snap.type_ids[keep], snap.cell,
                    snap.pbc, snap.type_masses, snap.orig_type_labels)


def augment(snap: Snapshot, labels: np.ndarray,
            rng: np.random.Generator) -> tuple[Snapshot, np.ndarray]:
    if rng.random() < 0.5:
        snap, labels = type_split(snap, labels, rng)
    if rng.random() < 0.25:
        snap = subsample_type(snap, rng)
    snap = strain(snap, rng)
    snap = jitter(snap, rng)
    return snap, labels
