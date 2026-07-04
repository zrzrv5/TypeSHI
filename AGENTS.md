# AGENTS.md — orientation for coding agents

TypeSHI predicts top-K element candidates per atom **type id** from geometry only
(`positions`, `type_ids`, optional `cell`). Two-stage design: per-type-pair descriptors
(partial RDFs + robust scalars + sampled neighbor environments) → small set-transformer
(1.85M params, 94-way per type). Read `docs/MODEL.md` for the why of every piece before
changing the model or the features.

## Commands

Everything runs from the repo root as `uv run python ...` (Python 3.12, uv-managed).

```bash
uv run python scripts/predict.py <file> --conformal      # torch ensemble inference
uv run python scripts/predict_lite.py <file>             # ONNX, no torch
uv run python scripts/evaluate_bench.py runs/production/*.ckpt --decode   # real-file bench
uv run python scripts/train.py --data Data/processed/<shards...> --name X --epochs 30 --use-env
uv run python scripts/export_model.py                    # ONNX + CoreML (CPU only)
```

Full pipeline (data → shards → training → calibration → export): `docs/REPRODUCE.md`.

## Hard rules

- **Never feed masses (or any file metadata) to the model.** Masses are a printed baseline and
  a source of eval ground truth only. The owner considers anything else cheating — and one
  eval file has *wrong* masses, which is the point of the project.
- **Never scale-normalize or scale-augment geometry.** Absolute ångström distances are the
  physical signal; unit inference (`src/typeid2elem/units.py`) depends on this.
- **torch stays pinned to cu128 wheels** in `pyproject.toml` (driver supports CUDA ≤ 12.9;
  default wheels break). Do not "upgrade" this.
- **Neighbor lists via matscipy only** — ASE's is ~80× slower and once turned preprocessing
  into a multi-day stall.
- wandb key lives in a repo-root file named `WANDB_API_KEY` 
  `.gitignore` matches every spelling + `*_API_KEY`). Never print or commit it.
- `Data/` and `runs/` are gitignored (large); don't add data files to git. The committed
  `weights/` bundle (2 MB: int8 deploy model + `costats.npz` + `calib.npz`) is the exception —
  inference assets resolve from `weights/` first, then fall back to `runs/`/`Data/`
  (`src/typeid2elem/assets.py`). Full checkpoints/ONNX/CoreML are a GitHub Release tarball built
  by `scripts/package_release.py`, not committed.

## Documentation conventions

- `docs/EXPERIMENTS.md` — **append-only** lab log (date, config, metrics, conclusion). Every
  experiment gets an entry, including negative results.
- `docs/PLAN.md` — design decisions + decision log; update when a decision changes.
- `docs/DATA.md` — dataset stats and acquisition notes.

## Traps already hit (don't rediscover)

- DeePMD MIXED-TYPE format: real species in `set.*/real_atom_types.npy`, `type.raw` is
  padding; virtual type −1 must be stripped (`scripts/preprocess_deepmd.py` handles it).
- Experimental CIFs contain dummy species `X` (Z=0) and partial occupancies — both must be
  filtered; a Z=0 label crashes CE with an async CUDA device-side assert.
- Empty `CUDA_VISIBLE_DEVICES` gives a misleading "you might not have a CUDA gpu"; export
  scripts set it empty on purpose (CPU), training must set `=0` explicitly.
- Environment sets are 16-atom samples: predictions must pool ≥4 draws or near-ties flip
  with the seed.
- Interior-crop for huge files: per-type densities/fractions must come from the FULL system,
  not the crop (composition bias rescales whole RDF rows).
- VASP CONTCARs may carry trailing predictor-corrector blocks that break `ase.io.vasp`.
- Benchmark ground truths must be verified against stoichiometry, not just metadata or
  collection-agent reports (~1 error per collected batch historically).

## Evaluation etiquette

The bench (`BENCH` in `scripts/evaluate_bench.py`) is the project's compass; current standing
numbers live at the end of `docs/EXPERIMENTS.md`. When comparing models, keep the fusion mode
and denominator fixed; WBM (`scripts/evaluate_wbm.py`) is the large-N tiebreaker. Val top-1 is
only comparable between runs trained on the same data mix (val = 10% of that mix's element sets).
