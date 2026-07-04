"""Parsers producing a uniform Snapshot from various simulation file formats.

A Snapshot is the model's canonical input: positions, integer type ids
(0-based, contiguous), optional cell/pbc, optional per-type masses.
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Masses for the optional mass-lookup baseline (not used by the ML model).
from ase.data import atomic_masses, chemical_symbols


@dataclass
class Snapshot:
    positions: np.ndarray            # (N, 3) float64, Cartesian, Angstrom
    type_ids: np.ndarray             # (N,) int64, 0-based contiguous
    cell: np.ndarray | None = None   # (3, 3) float64 rows = lattice vectors
    pbc: bool = True
    type_masses: dict[int, float] = field(default_factory=dict)  # 0-based type -> mass
    orig_type_labels: list[str] = field(default_factory=list)    # original ids as strings

    @property
    def n_types(self) -> int:
        return int(self.type_ids.max()) + 1 if len(self.type_ids) else 0

    def type_fractions(self) -> np.ndarray:
        return np.bincount(self.type_ids, minlength=self.n_types) / len(self.type_ids)


def _relabel(raw_types: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Map arbitrary type ids to 0..T-1 preserving sort order of originals."""
    uniq = np.unique(raw_types)
    remap = {t: i for i, t in enumerate(uniq)}
    return np.array([remap[t] for t in raw_types], dtype=np.int64), [str(t) for t in uniq]


# ---------------------------------------------------------------------------
# LAMMPS data file (read_data format), atom styles: atomic / charge / full / molecular
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(
    r"^(Masses|Atoms|Velocities|Bonds|Angles|Dihedrals|Impropers|"
    r"Pair Coeffs|PairIJ Coeffs|Bond Coeffs|Angle Coeffs|Dihedral Coeffs|"
    r"Improper Coeffs|Atom Type Labels)\b"
)


def read_lammps_data(path: str | Path) -> Snapshot:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        lines = fh.readlines()

    n_atoms = None
    xlo = xhi = ylo = yhi = zlo = zhi = None
    xy = xz = yz = 0.0
    atom_style_hint = None
    masses: dict[int, float] = {}

    # --- header ---
    i = 1  # first line is a comment
    while i < len(lines):
        line = lines[i].split("#")[0].strip()
        raw = lines[i]
        if _SECTION_RE.match(lines[i].strip()):
            break
        if line:
            parts = line.split()
            if line.endswith("atoms"):
                n_atoms = int(parts[0])
            elif line.endswith("xlo xhi"):
                xlo, xhi = float(parts[0]), float(parts[1])
            elif line.endswith("ylo yhi"):
                ylo, yhi = float(parts[0]), float(parts[1])
            elif line.endswith("zlo zhi"):
                zlo, zhi = float(parts[0]), float(parts[1])
            elif line.endswith("xy xz yz"):
                xy, xz, yz = float(parts[0]), float(parts[1]), float(parts[2])
        i += 1

    if n_atoms is None or xlo is None:
        raise ValueError(f"{path}: could not parse LAMMPS data header")

    cell = np.array(
        [[xhi - xlo, 0.0, 0.0], [xy, yhi - ylo, 0.0], [xz, yz, zhi - zlo]]
    )
    origin = np.array([xlo, ylo, zlo])

    # --- sections ---
    raw_types = np.empty(n_atoms, dtype=np.int64)
    pos = np.empty((n_atoms, 3))
    images = np.zeros((n_atoms, 3))

    while i < len(lines):
        header = lines[i].strip()
        m = _SECTION_RE.match(header)
        if not m:
            i += 1
            continue
        section = m.group(1)
        if section == "Atoms" and "#" in header:
            atom_style_hint = header.split("#", 1)[1].strip()
        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
        if section == "Masses":
            while i < len(lines) and lines[i].strip() and not _SECTION_RE.match(lines[i].strip()):
                parts = lines[i].split("#")[0].split()
                if len(parts) >= 2:
                    masses[int(parts[0])] = float(parts[1])
                i += 1
        elif section == "Atoms":
            count = 0
            while i < len(lines) and count < n_atoms:
                line = lines[i].split("#")[0].strip()
                i += 1
                if not line:
                    continue
                parts = line.split()
                tid, xyz, img = _parse_atom_line(parts, atom_style_hint)
                raw_types[count] = tid
                pos[count] = xyz
                images[count] = img
                count += 1
            if count != n_atoms:
                raise ValueError(f"{path}: expected {n_atoms} atoms, got {count}")
        else:
            while i < len(lines) and (not lines[i].strip() or not _SECTION_RE.match(lines[i].strip())):
                i += 1

    # unwrap images then wrap into cell (RDF only needs wrapped coords)
    pos = pos - origin + images @ cell
    frac = np.linalg.solve(cell.T, pos.T).T % 1.0
    pos = frac @ cell

    type_ids, labels = _relabel(raw_types)
    type_masses = {}
    for orig, new in zip(labels, range(len(labels))):
        if int(orig) in masses:
            type_masses[new] = masses[int(orig)]
    return Snapshot(pos, type_ids, cell, True, type_masses, labels)


def _parse_atom_line(parts: list[str], style_hint: str | None):
    """Return (type, xyz, image_flags) for one Atoms line."""
    n = len(parts)

    def tail_images(k):  # image flags present if 3 trailing ints beyond k floats
        if n == k + 3:
            try:
                return [int(p) for p in parts[k:]]
            except ValueError:
                pass
        return [0, 0, 0]

    if style_hint in ("atomic", "metal", None):
        # id type x y z [ix iy iz]; fall through to guessing if hint is None
        if style_hint == "atomic" or (style_hint is None and n in (5, 8)):
            return int(parts[1]), [float(x) for x in parts[2:5]], tail_images(5)
    if style_hint == "charge" or (style_hint is None and n in (6, 9)):
        return int(parts[1]), [float(x) for x in parts[3:6]], tail_images(6)
    if style_hint in ("full", "molecular") or (style_hint is None and n in (7, 10)):
        if style_hint == "molecular":
            return int(parts[2]), [float(x) for x in parts[3:6]], tail_images(6)
        return int(parts[2]), [float(x) for x in parts[4:7]], tail_images(7)
    raise ValueError(f"cannot infer atom style from line with {n} columns (hint={style_hint})")


# ---------------------------------------------------------------------------
# ASE-backed reader (extxyz, POSCAR, cif, ...) - used for training data
# ---------------------------------------------------------------------------

def snapshot_from_atoms(atoms) -> tuple[Snapshot, np.ndarray]:
    """ASE Atoms -> (Snapshot with types = element groups, per-type atomic numbers)."""
    numbers = atoms.get_atomic_numbers()
    uniq = np.unique(numbers)
    remap = {z: i for i, z in enumerate(uniq)}
    type_ids = np.array([remap[z] for z in numbers], dtype=np.int64)
    cell = np.asarray(atoms.get_cell()) if atoms.cell.rank == 3 else None
    snap = Snapshot(
        atoms.get_positions(), type_ids, cell,
        bool(np.all(atoms.get_pbc())) and cell is not None,
        orig_type_labels=[chemical_symbols[z] for z in uniq],
    )
    return snap, uniq.astype(np.int64)


# ---------------------------------------------------------------------------
# Mass baseline
# ---------------------------------------------------------------------------

def mass_lookup(mass: float, k: int = 3) -> list[tuple[str, float]]:
    """Top-k elements by |standard atomic mass - mass|."""
    diffs = np.abs(atomic_masses[1:95] - mass)
    order = np.argsort(diffs)[:k]
    return [(chemical_symbols[z + 1], float(diffs[z])) for z in order]
