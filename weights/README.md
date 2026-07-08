# weights/ — committed lite-runtime bundle

Everything the **no-torch inference path** needs at runtime, with no model download
(~2 MB total):

| file | what | used by |
|---|---|---|
| `env_codfull_sharp30.int8.onnx` | single deploy model, int8 (E16/E17) | `scripts/predict_lite.py`, `assets.deploy_onnx()` |
| `costats.npz` | element co-occurrence stats | composition decode (`decode.CompositionPrior`) |
| `calib.npz` | conformal temperature + RAPS qhat | `--conformal` sets |

```bash
uv run python scripts/predict_lite.py <file>          # uses weights/ automatically
```

Small packaging caveat: `predict_lite.py` does not import torch, but the repo's default
`uv` environment is still the full training environment and pins CUDA Torch wheels. On a
Linux CUDA machine, `uv run` works. On macOS/Apple Silicon, run the lite script from a
small venv with `numpy scipy ase matscipy onnxruntime` until the project gets a separate
runtime dependency group.

## Full artifacts (torch checkpoints, fp32 ONNX, all CoreML packages)

Not committed — they live in the **GitHub Release `v0.1`** tarball
`typeshi-weights-v0.1.tar.gz`. Needed only for training, the torch ensemble
(`scripts/predict.py`, `typeid2elem.api`), or re-exporting to CoreML/Metal.
Unpack it into `runs/` (`checkpoints/`→`runs/production/`, `onnx/`+`coreml/`
→`runs/export/`); the code prefers `weights/` and falls back to `runs/`.
