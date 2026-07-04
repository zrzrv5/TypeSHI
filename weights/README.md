# weights/ — committed runtime bundle

Everything the **no-torch inference path** needs, so a fresh clone runs with no
extra download (~2 MB total):

| file | what | used by |
|---|---|---|
| `env_codfull_sharp30.int8.onnx` | single deploy model, int8 (E16/E17) | `scripts/predict_lite.py`, `assets.deploy_onnx()` |
| `costats.npz` | element co-occurrence stats | composition decode (`decode.CompositionPrior`) |
| `calib.npz` | conformal temperature + RAPS qhat | `--conformal` sets |

```bash
uv run python scripts/predict_lite.py <file>          # uses weights/ automatically
```

## Full artifacts (torch checkpoints, fp32 ONNX, all CoreML packages)

Not committed — they live in the **GitHub Release `v0.1`** tarball
`typeshi-weights-v0.1.tar.gz`. Needed only for training, the torch ensemble
(`scripts/predict.py`, `typeid2elem.api`), or re-exporting to CoreML/Metal.
Unpack it into `runs/` (`checkpoints/`→`runs/production/`, `onnx/`+`coreml/`
→`runs/export/`); the code prefers `weights/` and falls back to `runs/`.
