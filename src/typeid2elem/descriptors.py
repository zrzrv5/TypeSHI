"""Per-type-pair descriptors: partial RDFs + scalar features.

The descriptor set for a snapshot with T types is:
  rdf        (T, T, N_BINS)  smeared partial RDF g_ab(r), ordered pairs (b around a)
  pair_extra (T, T, N_PAIR_EXTRA) running coordination numbers + closest-approach distance
  frac       (T,)            stoichiometric fractions x_a
  glob       (N_GLOB,)       log number density, has_cell flag, 1/n_types
                             (T is always the NUMBER OF TYPE IDS in this file;
                             no thermal information is used anywhere)

All quantities use absolute Angstrom distances (bond lengths are the signal).
"""

from __future__ import annotations

import numpy as np
from matscipy.neighbours import neighbour_list
from scipy.ndimage import gaussian_filter1d

from .io import Snapshot

R_MAX = 8.0
N_BINS = 64
DR = R_MAX / N_BINS
SMEAR_BINS = 1.0                      # Gaussian sigma in bins applied to raw histogram
CN_RADII = (2.0, 3.0, 4.0, 6.0)       # running coordination number checkpoints
# pair extras: log1p(cn)x4, median NN dist, 10th-pct NN dist, peak pos, log1p(peak height)
# NN stats are per-atom medians/percentiles: robust to thermal close contacts,
# unlike a raw first-nonzero-histogram-bin distance (extreme-value statistic
# that collapses at finite temperature -- see docs/EXPERIMENTS.md E1).
N_PAIR_EXTRA = len(CN_RADII) + 4
N_GLOB = 3

# learned-encoder environment sets (v3): per type, M sampled atoms x K nearest
# neighbors within R_ENV, stored as (distance, neighbor type). Unlike pair-
# marginal RDFs these keep the per-atom JOINT coordination (e.g. "each Li sees
# 4 O AND 2 Cl"), which marginals cannot express.
M_ENV = 16
K_ENV = 16
R_ENV = 6.0


def _compute_env(types, T, i_idx, j_idx, dists, rng, center_ok=None):
    env_d = np.zeros((T, M_ENV, K_ENV), dtype=np.float32)
    env_t = np.full((T, M_ENV, K_ENV), -1, dtype=np.int8)   # -1 = pad
    chosen = []
    for a in range(T):
        atoms_a = np.flatnonzero((types == a) if center_ok is None
                                 else ((types == a) & center_ok))
        if len(atoms_a) > M_ENV:
            atoms_a = rng.choice(atoms_a, M_ENV, replace=False)
        chosen.append(atoms_a)
    chosen_all = np.concatenate(chosen) if len(chosen) else np.empty(0, int)
    if len(chosen_all) == 0:
        return env_d, env_t
    sel = (dists <= R_ENV) & np.isin(i_idx, chosen_all)
    ii, jt, dd = i_idx[sel], types[j_idx[sel]], dists[sel]
    order = np.lexsort((dd, ii))                            # by atom, then distance
    ii, jt, dd = ii[order], jt[order], dd[order]
    for a in range(T):
        for m, atom in enumerate(chosen[a]):
            lo = np.searchsorted(ii, atom, "left")
            hi = np.searchsorted(ii, atom, "right")
            k = min(K_ENV, hi - lo)
            env_d[a, m, :k] = dd[lo:lo + k]
            env_t[a, m, :k] = jt[lo:lo + k]
    return env_d, env_t


MAX_ATOMS_DEFAULT = 100_000


def crop_for_inference(snap: Snapshot, max_atoms: int = MAX_ATOMS_DEFAULT):
    """Bounded-resource input for huge periodic systems.

    Returns (snapshot, center_mask, full_stats). Takes a fractional-space
    crop of ~max_atoms; atoms further than R_MAX from every crop face (by
    perpendicular cell heights) become RDF centers with COMPLETE neighborhoods,
    the rest only serve as neighbors. full_stats = (rho_b, frac) of the FULL
    system: crops of crystals are composition-biased (the boundary cuts
    sublattices unevenly), and re-estimating per-type density from the crop
    scales whole RDF rows by that bias -- so normalization must use the
    exact full-system statistics. Peak memory stays O(max_atoms).
    """
    n = len(snap.positions)
    if n <= max_atoms or snap.cell is None or not snap.pbc:
        return snap, None, None
    cell = snap.cell
    volume = abs(np.linalg.det(cell))
    # perpendicular heights of the cell along each lattice direction
    heights = np.array([
        volume / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3]))
        for i in range(3)])
    frac = (np.linalg.solve(cell.T, snap.positions.T).T) % 1.0
    # crop fractions per axis: equal linear shrink, but keep enough room for
    # an interior (2*R_MAX per axis) plus statistics
    shrink = (max_atoms / n) ** (1 / 3)
    crop_f = np.minimum(1.0, np.maximum(shrink, (2 * R_MAX + 4.0) / heights))
    if np.any(crop_f >= 1.0 - 1e-9) and np.prod(crop_f) > 0.9:
        return snap, None, None                    # cell too thin to crop
    keep = np.all(frac < crop_f, axis=1)
    margin_f = R_MAX / heights
    interior = keep & np.all((frac >= margin_f) &
                             (frac < crop_f - margin_f), axis=1)
    if interior.sum() < 500:                       # not enough clean centers
        return snap, None, None
    sub = Snapshot(
        positions=(frac[keep] @ cell),
        type_ids=snap.type_ids[keep],
        cell=cell * crop_f[:, None],               # bounding region (pbc OFF)
        pbc=False,
        orig_type_labels=snap.orig_type_labels,
        type_masses=snap.type_masses,
    )
    counts_full = np.bincount(snap.type_ids, minlength=snap.n_types).astype(np.float64)
    return sub, interior[keep], (counts_full / volume, counts_full / n)


def compute_features(snap: Snapshot, with_env: bool = False,
                     rng: np.random.Generator | None = None,
                     center_mask: np.ndarray | None = None,
                     full_stats: tuple | None = None) -> dict[str, np.ndarray]:
    pos = snap.positions
    types = snap.type_ids
    T = snap.n_types
    n = len(pos)
    counts_per_type = np.bincount(types, minlength=T).astype(np.float64)

    if center_mask is not None:
        # interior-crop mode (crop_for_inference): non-periodic neighbor list
        # over the crop; only complete-neighborhood atoms act as RDF centers;
        # densities/fractions come from the FULL system via full_stats
        lo = pos.min(0)
        span = pos.max(0) - lo + 2.0
        cell, pbc = np.diag(span), (False,) * 3
        pos = pos - lo + 1.0
        volume, has_cell = 1.0, 1.0                # placeholder; rho overridden
    elif snap.cell is not None and snap.pbc:
        cell, pbc, volume, has_cell = snap.cell, (True,) * 3, abs(np.linalg.det(snap.cell)), 1.0
    else:
        # Cell-less input: reproduce the OMol25 vacuum-box convention seen in
        # training (cell = molecular extent + 10 A, treated as periodic; the
        # 10 A padding exceeds R_MAX so periodic images never interact).
        # A convex-hull density path was tried and is OOD vs training -- E5.
        lo = pos.min(0)
        span = pos.max(0) - lo + 10.0
        cell, pbc = np.diag(span), (True,) * 3
        pos = pos - lo + 5.0
        volume, has_cell = float(np.prod(span)), 1.0

    i_idx, j_idx, dists = neighbour_list(
        "ijd", positions=pos, cell=cell, pbc=np.array(pbc), cutoff=R_MAX
    )

    if center_mask is not None:
        sel = center_mask[i_idx]
        ci, cj, cd = i_idx[sel], j_idx[sel], dists[sel]
        counts_centers = np.bincount(
            types[center_mask], minlength=T).astype(np.float64)
    else:
        ci, cj, cd = i_idx, j_idx, dists
        counts_centers = counts_per_type

    # raw pair-distance histogram, ordered (a<-b), centers only
    bins = np.minimum((cd / DR).astype(np.int64), N_BINS - 1)
    flat = (types[ci] * T + types[cj]) * N_BINS + bins
    hist = np.bincount(flat, minlength=T * T * N_BINS).astype(np.float64)
    hist = hist.reshape(T, T, N_BINS)

    # running coordination number n_ab(r) BEFORE smearing (physical counts)
    cum = np.cumsum(hist, axis=-1) / np.maximum(counts_centers, 1.0)[:, None, None]
    cn_idx = [min(int(r / DR), N_BINS - 1) for r in CN_RADII]
    cn = cum[:, :, cn_idx]                                   # (T, T, len(CN_RADII))

    # robust bond-length stats: per-atom nearest-neighbor distance to each type,
    # aggregated as median and 10th percentile over atoms of type a
    dmin = np.full((n, T), R_MAX)
    np.minimum.at(dmin, (ci, types[cj]), cd)
    nn_med = np.full((T, T), R_MAX)
    nn_p10 = np.full((T, T), R_MAX)
    is_center = np.ones(n, bool) if center_mask is None else center_mask
    for a in range(T):
        rows = dmin[(types == a) & is_center]
        if len(rows):
            nn_med[a] = np.median(rows, axis=0)
            nn_p10[a] = np.percentile(rows, 10, axis=0)

    # smeared, ideal-gas-normalized g_ab(r)
    hist = gaussian_filter1d(hist, SMEAR_BINS, axis=-1, mode="constant")
    r_edges = np.arange(N_BINS + 1) * DR
    shell_vol = 4.0 / 3.0 * np.pi * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)
    rho_b = full_stats[0] if center_mask is not None else counts_per_type / volume
    denom = counts_centers[:, None, None] * rho_b[None, :, None] * shell_vol[None, None, :]
    # clip: isolated molecules in huge hull volumes give astronomically large g
    # (tiny rho_b); also keeps values fp16-safe for storage. Model takes log1p.
    rdf = np.minimum(hist / np.maximum(denom, 1e-12), 1e4)

    # dominant-correlation peak of the smoothed g (robust to lone close contacts)
    peak_bin = rdf.argmax(axis=-1)
    peak_pos = (peak_bin + 0.5) * DR / R_MAX                 # (T, T)
    peak_h = np.log1p(np.take_along_axis(rdf, peak_bin[..., None], axis=-1)[..., 0])

    pair_extra = np.concatenate(
        [np.log1p(cn), nn_med[:, :, None] / R_MAX, nn_p10[:, :, None] / R_MAX,
         peak_pos[:, :, None], peak_h[:, :, None]], axis=-1
    )                                                        # (T, T, N_PAIR_EXTRA)
    if center_mask is not None:
        frac = full_stats[1]
        glob = np.array([np.log(full_stats[0].sum()), has_cell, 1.0 / T])
    else:
        frac = counts_per_type / n
        glob = np.array([np.log(n / volume), has_cell, 1.0 / T])

    out = {
        "rdf": rdf.astype(np.float32),
        "pair_extra": pair_extra.astype(np.float32),
        "frac": frac.astype(np.float32),
        "glob": glob.astype(np.float32),
    }
    if with_env:
        if rng is None:
            rng = np.random.default_rng(12345)   # deterministic at inference
        out["env_d"], out["env_t"] = _compute_env(
            types, T, ci, cj, cd, rng,
            center_ok=None if center_mask is None else is_center)
    return out


def compute_features_capped(snap: Snapshot, with_env: bool = True,
                            max_atoms: int = MAX_ATOMS_DEFAULT,
                            rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
    """compute_features with bounded memory/time on huge periodic systems."""
    sub, mask, stats = crop_for_inference(snap, max_atoms)
    return compute_features(sub, with_env=with_env, center_mask=mask,
                            full_stats=stats, rng=rng)


def average_features(feature_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Multi-frame fusion at descriptor level (time-averaged partial RDFs).

    All frames must share the same type labelling. Environment sets (env_*)
    are discrete samples, not densities -- keep frame 0's rather than averaging.
    """
    return {
        key: (feature_list[0][key] if key.startswith("env")
              else np.mean([f[key] for f in feature_list], axis=0))
        for key in feature_list[0]
    }
