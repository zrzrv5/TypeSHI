"""Locate bundled runtime assets, preferring the committed weights/ bundle.

A fresh GitHub clone ships only the minimal *runtime* bundle in weights/
(deploy int8 model + decode/conformal stats, ~2 MB). A full local working tree
additionally has the originals under runs/ and Data/processed/, and the complete
set of checkpoints/ONNX/CoreML lives in the GitHub Release tarball.

Each resolver returns the weights/ copy if present, else falls back to the
legacy training-time location, so both a bare clone and a full working tree run
without configuration.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _first(*rel: str) -> str:
    for r in rel:
        p = REPO / r
        if p.exists():
            return str(p)
    return str(REPO / rel[0])          # default (may not exist yet)


def deploy_onnx() -> str:
    """Single production model for the no-torch lite path (int8 preferred)."""
    return _first("weights/env_codfull_sharp30.int8.onnx",
                  "runs/export/env_codfull_sharp30.int8.onnx",
                  "runs/export/env_codfull_sharp30.onnx")


def costats_npz() -> str:
    """Element co-occurrence stats for the composition-decode prior."""
    return _first("weights/costats.npz", "Data/processed/costats.npz")


def calib_npz() -> str:
    """Conformal calibration (temperature + RAPS qhat)."""
    return _first("weights/calib.npz", "runs/production/calib.npz")


def production_ckpts() -> tuple[str, ...]:
    """Torch ensemble checkpoints (full artifacts; from the Release tarball)."""
    d = REPO / "runs/production"
    return tuple(sorted(str(p) for p in d.glob("*.ckpt"))) if d.exists() else ()
