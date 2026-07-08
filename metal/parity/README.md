# metal/parity/ — golden reference for the Metal port

`golden.json` is the source of truth the Swift `parity-check` diffs against. It is produced
by the **Python** pipeline, so it's authoritative and platform-independent.

- `gen_reference.py` — builds a small, fully-specified rocksalt structure (64 atoms, both
  types), computes reference descriptors with `typeid2elem.descriptors.compute_features`,
  runs the committed int8 ONNX model, and writes `golden.json`.
- `golden.json` — `input` (positions/type_ids/cell the Swift test rebuilds), `expected`
  (deterministic `rdf`/`pair_extra`/`frac`/`glob` the kernel must reproduce), `env_seed0`
  (sampled env for one seed — structural check only), and `log_probs`/`top1_per_type`.

## Decode fixture

- `gen_decode_golden.py` — runs the Python `CompositionPrior.marginals` + RAPS conformal on
  `golden.json`'s committed `log_probs` (using the same baked PMI/calib the Swift side loads)
  and writes `decode_golden.json`. `parity-check` reproduces it from the same `log_probs` —
  an on-device unit test for `Decode.swift` that needs no CoreML model.

## Verification scripts (no-torch venv: `numpy scipy ase onnxruntime matscipy`)

The full training env can't install on macOS (torch is cu128-pinned), so these validate the
model/eval legs through the committed **int8 ONNX** deploy model instead of CoreML:

- `onnx_bridge.py <dump.json>` — pushes Metal descriptors (from `parity-check … --dump`)
  through the deploy ONNX and checks top-1 vs `golden.json` (the end-to-end acceptance test).
- `eval_metal_local.py` — runs the Metal `desc-dump` path on every committed eval file
  (`Data/eval_cases/*`), diffs descriptors vs Python, and compares top-1 vs Python & truth.
- `check_edge_cases.py` — open-box (cell-less) + triclinic descriptor paths the eval files
  don't exercise.
- `eval_lite_local.py` — lite (ONNX + decode) accuracy on the committed eval subset.

## Regenerate

```bash
# any platform, no Metal needed (use the no-torch venv or `uv run` on Linux):
python metal/parity/gen_reference.py         # golden.json
python metal/parity/gen_decode_golden.py     # decode_golden.json
python metal/tools/gen_decode_assets.py      # ../Sources/.../Resources/decode_assets.json
```

Keep `golden.json` / `decode_golden.json` committed and regenerate whenever the descriptor
definition, the deploy model, or the decode/calibration stats change — a stale golden
silently passes a wrong port.
