# Reproducing TypeSHI

Every step from a clean checkout to a working predictor: environment, data acquisition,
preprocessing, training, calibration, export, evaluation. Raw data is ~120 GB and is not in
the repository. Approximate times are for a 16-core box with one RTX 4090.

## 0. Environment

```bash
uv sync          # Python 3.12, all deps
```

Gotchas baked into `pyproject.toml` (do not "fix" them):

- **torch is pinned to cu128 wheels** (`[tool.uv.sources]`). Default torch wheels need CUDA 13;
  driver 575.x supports CUDA ≤ 12.9 and fails with them.
- **matscipy** does the neighbor lists. ASE's Python implementation is ~80× slower and turns
  preprocessing from an hour into days.
- Background training runs should set `CUDA_VISIBLE_DEVICES=0` explicitly — an inherited empty
  value produces a misleading "you might not have a CUDA gpu".

Optional experiment tracking: put a wandb API key in a repo-root file named `WANDB_API_KEY`
(sic — the filename typo is load-bearing, scripts look for exactly this; the file is
gitignored). Without it, training logs to CSV only.

## 1. Data acquisition (→ `Data/`, gitignored)

| dataset | role | how to get it |
|---|---|---|
| MPTrj (`MPtrj_2022.9_full.json`, 12 GB) | training | figshare article 23713842 ("Materials Project Trajectory Dataset"). Stream with ijson only — never json.load it. |
| COSMOS DBS (`dbs_total.extxyz`) | training | COSMOS dataset release → `Data/COSMOS/DBS/`. Also ships an LPSC r2SCAN MD trajectory used as an eval case. |
| COD — experimental crystals | training (biggest single lever) | `rsync -a rsync://www.crystallography.net/cif/ Data/COD/cif/` — ~535k CIFs / 111 GB, several hours, anonymous. |
| OpenLAM / DPA-2 domains (Alloy, Anode, Cluster, FerroEle) | training | AIS-Square (aissquare.com). The site is an SPA; direct links come from `https://backend.aissquare.com/dpa/detail/datasets?type=datasets&id=<id>` → `store.aissquare.com/...`. Extract tarballs under `Data/openLAM/<Domain>/`. |
| DPA-2 HfO₂ domain | training (targeted fix) | `https://store.aissquare.com/datasets/4561a31f-db9c-11ee-9b22-506b4b2349d8/HfO2.tar.gz` → `Data/openLAM/HfO2/` (114 systems, 57k frames). |
| WBM | evaluation | initial structures: HF dataset `nimashoghi/wbm` (parquet). Relaxed structures: figshare file id 40344463 (`https://ndownloader.figshare.com/files/40344463`, 66 MB) — the 2022 upload; the 2024 re-uploads sit in figshare cold storage and 202 for hours. |

**DeePMD format warning (cost us a poisoned training run):** some OpenLAM domains use the
MIXED-TYPE layout — real species live in `set.*/real_atom_types.npy` (virtual type −1 =
padding) and `type.raw` is meaningless. `scripts/preprocess_deepmd.py` handles both layouts.

**CIF warning:** experimental CIFs contain dummy species (`X`, Z=0) and partial occupancies.
The preprocessor skips both — two Z=0 records among 710k were enough to crash training with an
async CUDA device-side assert.

## 2. Preprocessing (structures → descriptor shards, CPU)

```bash
uv run python scripts/preprocess_cosmos.py Data/COSMOS/DBS/dbs_total.extxyz \
    --out Data/processed/cosmos_v3 --n-aug 2 --workers 12          # 401,310 records, ~5 min
uv run python scripts/preprocess_mptrj.py Data/MPtrj/MPtrj_2022.9_full.json \
    --out Data/processed/mptrj_v3 --n-aug 1 --frames-per-material 2 --workers 12
                                                                   # 543,712 records, ~10 min
uv run python scripts/preprocess_deepmd.py Data/openLAM/Alloy Data/openLAM/Anode \
    Data/openLAM/Cluster Data/openLAM/FerroEle \
    --out Data/processed/openlam_v3 --max-frames-per-sys 4 --n-aug 1   # 19,070 records
uv run python scripts/preprocess_deepmd.py Data/openLAM/HfO2 \
    --out Data/processed/hfo2_v3 --prefix hfo2 --max-frames-per-sys 20 --n-aug 1   # 3,518
uv run python scripts/preprocess_cod.py Data/COD/cif \
    --out Data/processed/cod_v3_full --n-aug 1 --workers 12         # 709,840 records, ~40 min

uv run python scripts/mine_costats.py    # element co-occurrence prior -> Data/processed/costats.npz
```

Shards are fp16 npz bucketed by type count; ~3 GB total; the whole set loads into RAM at
training time (~25 GB with the env arrays).

## 3. Training (GPU)

The six production members differ by recipe on purpose — heterogeneous recipes ensemble well,
same-recipe seeds do not:

```bash
T="uv run python scripts/train.py"
$T --data Data/processed/cosmos_v3 Data/processed/mptrj_v3 --name soft15  --epochs 15
$T --data Data/processed/cosmos_v3 Data/processed/mptrj_v3 --name sharp30 --epochs 30
$T --data Data/processed/cosmos_v3 Data/processed/mptrj_v3 \
   Data/processed/cosmos_aug2 Data/processed/mptrj_aug2    --name aug2sharp30 --epochs 30
$T --data Data/processed/cosmos_v3 Data/processed/mptrj_v3 Data/processed/openlam_v3 \
                                                           --name openlam_sharp30 --epochs 30
$T --data Data/processed/cosmos_v3 Data/processed/mptrj_v3 Data/processed/openlam_v3 \
   --use-env                                               --name env_sharp30 --epochs 30
$T --data Data/processed/cosmos_v3 Data/processed/mptrj_v3 Data/processed/openlam_v3 \
   Data/processed/hfo2_v3 Data/processed/cod_v3_full --use-env \
                                                           --name env_codfull_sharp30 --epochs 30
```

~1.5 h each on a 4090 for the full mix (shorter for the smaller mixes). Expected validation
top-1 (val = 10% of element sets of that run's own data mix — numbers are only comparable
within the same mix): openlam mix ≈ 0.57 without env, ≈ 0.61 with env; full COD mix ≈ 0.76.
Copy each `runs/<name>/version_*/checkpoints/best-*.ckpt` to `runs/production/<name>.ckpt`.

If you only train one model, make it `env_codfull_sharp30` — solo it matches the ensemble on
the real-file bench.

## 4. Calibration & export

```bash
uv run python scripts/calibrate.py --data Data/processed/cosmos_v3 \
    Data/processed/mptrj_v3 Data/processed/openlam_v3 --sample 8000 --coverage 0.9
# expect: temperature ~3.6, coverage ~91.7%, median set size 7 -> runs/production/calib.npz

CUDA_VISIBLE_DEVICES="" uv run python scripts/export_model.py       # ONNX + CoreML, CPU
uv run python - <<'EOF'                                             # int8 (1.9 MB)
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("runs/export/env_codfull_sharp30.onnx",
                 "runs/export/env_codfull_sharp30.int8.onnx", weight_type=QuantType.QInt8)
EOF
```

ONNX parity is asserted < 1e-3 (measured ~3e-5). CoreML conversion runs on Linux; executing
the .mlpackage requires macOS.

## 5. Evaluation

```bash
uv run python scripts/evaluate_bench.py runs/production/*.ckpt --decode          # real files
uv run python scripts/evaluate_lite.py runs/export/env_codfull_sharp30.int8.onnx # ONNX parity
uv run python scripts/evaluate_wbm.py runs/production/*.ckpt \
    --parquet Data/WBM/wbm_0.parquet Data/WBM/wbm_1.parquet --sample 4000        # large scale
```

The bench (`BENCH` list in `scripts/evaluate_bench.py`) references files from our own working
directories plus copies under `Data/eval_cases/`; adapt the list to your files — any structure
whose true elements you know is a case. Two hard-won rules for building your own: (1) verify
ground truth against stoichiometry, not just metadata — one of our files has the wrong mass on
its Mn type; (2) prefer final/relaxed frames (CONTCAR over POSCAR).

Expected result with the released production set: **78–81 / 106–108 / 120–122 of 131** top-1/3/5
(±: env-draw pooling), WBM ≈ 27.5 / 45 / 53.5.

## 6. Deployment notes

- **Packaging for release:** `uv run python scripts/package_release.py --version v0.1` writes the
  committed 2 MB `weights/` runtime bundle (int8 model + `costats.npz` + `calib.npz`) and a
  `dist/typeshi-weights-v0.1.tar.gz` of the full set (optimizer-stripped checkpoints ~7.4 MB each,
  per-member ONNX, CoreML packages). Upload the tarball as a GitHub Release asset; `dist/` is
  gitignored.
- `scripts/predict_lite.py` is the minimal-footprint path (no torch): 166 MB RSS, 0.8 s/file.
- The `typeid2elem.api.predict_structure(positions, type_ids, cell)` function is the
  integration point for other tools — no file I/O.
- Planned (not yet in repo): Metal descriptor kernel ("InaccurateRDFCalculator") + CoreML for
  on-device Mac inference; the descriptor math is deliberately plain histogramming to make
  that port mechanical.
