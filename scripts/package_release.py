"""Package trained artifacts for GitHub sharing.

Produces two things:

  weights/                         (committed to the repo, ~2 MB)
    env_codfull_sharp30.int8.onnx  the single deploy model (no-torch lite path)
    costats.npz, calib.npz         composition-decode + conformal stats
    README.md                      manifest

  dist/typeshi-weights-<ver>/      (gitignored; upload as a GitHub Release asset)
    checkpoints/*.ckpt             6 production ckpts, OPTIMIZER STATE STRIPPED
                                   (~7.4 MB each vs 21 MB) -- enough to re-export
    onnx/                          int8 + per-member fp32 ONNX
    coreml/                        fp16 .mlpackage per member + env_codfull fp32
                                   (the CoreML/Metal export target)
    costats.npz, calib.npz, MANIFEST.md
  ... and dist/typeshi-weights-<ver>.tar.gz of that tree.

Run from repo root:  uv run python scripts/package_release.py --version v0.1
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
PROD = REPO / "runs/production"
EXPORT = REPO / "runs/export"
DEPLOY = "env_codfull_sharp30"          # E16 single-model deploy pick

KEEP_CKPT_KEYS = ("state_dict", "hyper_parameters", "pytorch-lightning_version",
                  "hparams_name", "epoch", "global_step")


def strip_ckpt(src: Path, dst: Path) -> tuple[int, int]:
    """Drop optimizer/scheduler/callback state; keep what load_from_checkpoint needs."""
    ck = torch.load(src, map_location="cpu", weights_only=False)
    slim = {k: ck[k] for k in KEEP_CKPT_KEYS if k in ck}
    torch.save(slim, dst)
    return src.stat().st_size, dst.stat().st_size


def human(n: int) -> str:
    return f"{n / 1e6:.1f} MB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v0.1")
    ap.add_argument("--no-tar", action="store_true")
    args = ap.parse_args()

    ckpts = sorted(PROD.glob("*.ckpt"))
    if not ckpts:
        raise SystemExit(f"no checkpoints in {PROD}")

    # ---- committed runtime bundle: weights/ ----
    weights = REPO / "weights"
    weights.mkdir(exist_ok=True)
    runtime = {
        EXPORT / f"{DEPLOY}.int8.onnx": weights / f"{DEPLOY}.int8.onnx",
        REPO / "Data/processed/costats.npz": weights / "costats.npz",
        PROD / "calib.npz": weights / "calib.npz",
    }
    for src, dst in runtime.items():
        if not src.exists():
            raise SystemExit(f"missing runtime asset {src}")
        shutil.copy2(src, dst)
    wsize = sum(p.stat().st_size for p in weights.glob("*") if p.is_file())

    # ---- release staging: dist/ ----
    stage = REPO / "dist" / f"typeshi-weights-{args.version}"
    for sub in ("checkpoints", "onnx", "coreml"):
        (stage / sub).mkdir(parents=True, exist_ok=True)

    ckpt_rows = []
    for c in ckpts:
        dst = stage / "checkpoints" / c.name
        a, b = strip_ckpt(c, dst)
        ckpt_rows.append((c.stem, a, b))

    for onnx in EXPORT.glob("*.onnx"):
        shutil.copy2(onnx, stage / "onnx" / onnx.name)

    # fp16 mlpackage per member (deploy-size) + env_codfull fp32 (Metal ref)
    ml_keep = []
    for pkg in EXPORT.glob("*.mlpackage"):
        name = pkg.name
        if name.startswith("top1="):                 # stray mis-exported artifact
            continue
        if name.endswith("_fp32.mlpackage") and not name.startswith(DEPLOY):
            continue
        shutil.copytree(pkg, stage / "coreml" / name, dirs_exist_ok=True)
        ml_keep.append(name)

    for extra in ("calib.npz",):
        shutil.copy2(PROD / extra, stage / extra)
    shutil.copy2(REPO / "Data/processed/costats.npz", stage / "costats.npz")

    # ---- manifests ----
    (weights / "README.md").write_text(_weights_readme(args.version))
    (stage / "MANIFEST.md").write_text(
        _manifest(args.version, ckpt_rows, sorted(ml_keep)))

    print(f"weights/  {human(wsize)}  ({len(list(weights.glob('*')))} files)")
    print("stripped checkpoints:")
    for stem, a, b in ckpt_rows:
        print(f"  {stem:<26} {human(a)} -> {human(b)}")

    if not args.no_tar:
        tar_path = REPO / "dist" / f"typeshi-weights-{args.version}.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tf:
            tf.add(stage, arcname=stage.name)
        print(f"\nrelease tarball: {tar_path}  ({human(tar_path.stat().st_size)})")
    print(f"staging tree:    {stage}")


def _weights_readme(ver: str) -> str:
    return f"""# weights/ — committed runtime bundle

Everything the **no-torch inference path** needs, so a fresh clone runs with no
extra download (~2 MB total):

| file | what | used by |
|---|---|---|
| `{DEPLOY}.int8.onnx` | single deploy model, int8 (E16/E17) | `scripts/predict_lite.py`, `assets.deploy_onnx()` |
| `costats.npz` | element co-occurrence stats | composition decode (`decode.CompositionPrior`) |
| `calib.npz` | conformal temperature + RAPS qhat | `--conformal` sets |

```bash
uv run python scripts/predict_lite.py <file>          # uses weights/ automatically
```

## Full artifacts (torch checkpoints, fp32 ONNX, all CoreML packages)

Not committed — they live in the **GitHub Release `{ver}`** tarball
`typeshi-weights-{ver}.tar.gz`. Needed only for training, the torch ensemble
(`scripts/predict.py`, `typeid2elem.api`), or re-exporting to CoreML/Metal.
Unpack it into `runs/` (`checkpoints/`→`runs/production/`, `onnx/`+`coreml/`
→`runs/export/`); the code prefers `weights/` and falls back to `runs/`.
"""


def _manifest(ver: str, ckpt_rows, ml_keep) -> str:
    rows = "\n".join(f"| `{s}.ckpt` | {b/1e6:.1f} MB |" for s, _, b in ckpt_rows)
    ml = "\n".join(f"- `{m}`" for m in ml_keep)
    return f"""# TypeSHI weights — Release {ver}

Full trained artifacts. The repo's `weights/` folder already carries the 2 MB
runtime subset; this tarball is for training / torch ensemble / CoreML re-export.

## Install
Unpack into the working tree:
```bash
tar xzf typeshi-weights-{ver}.tar.gz
mkdir -p runs/production runs/export
cp typeshi-weights-{ver}/checkpoints/*.ckpt runs/production/
cp typeshi-weights-{ver}/calib.npz          runs/production/
cp -r typeshi-weights-{ver}/onnx/*  typeshi-weights-{ver}/coreml/*  runs/export/
```

## checkpoints/ (optimizer state stripped; load via `TypeSetClassifier.load_from_checkpoint`)
| file | size |
|---|---|
{rows}

Deploy pick (E16, matches the 6-model ensemble on the real-file bench):
**`{DEPLOY}`**.

## coreml/ — CoreML packages (Metal / on-device export target)
{ml}

fp16 packages are the ~3.5 MB deploy size; `{DEPLOY}_fp32.mlpackage` is included
as the full-precision reference for the `metal/InaccurateRDFCalculator` work.

## onnx/
`{DEPLOY}.int8.onnx` (1.9 MB deploy) + per-member fp32 ONNX.
"""


if __name__ == "__main__":
    main()
