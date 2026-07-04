# M6 — deployment export (ONNX / CoreML)

`scripts/export_model.py` exports a trained `TypeSetClassifier` checkpoint to:

- `runs/export/<name>.onnx` — ONNX (opset 17), parity-checked against PyTorch with onnxruntime.
- `runs/export/<name>.mlpackage` — CoreML `mlprogram`, fp16 weights, iOS16+ deployment target.
- `runs/export/<name>_fp32.mlpackage` — CoreML fp32 variant, saved only when the fp16
  conversion pass raises a warning (see "CoreML fp16 warning" below). Currently saved
  for all 4 production checkpoints.

CoreML has no runtime on Linux, so the CoreML step is convert-only — there's no way to
run a parity check for it here. It must be checked on a Mac/iOS.

## Running it

```
uv run python scripts/export_model.py                # exports all runs/production/*.ckpt
uv run python scripts/export_model.py --ckpt runs/production/sharp30.ckpt
uv run python scripts/export_model.py --out-dir runs/export --n-parity 20 --seed 0
```

CPU only — do not run with a GPU visible (training may be using it). No CUDA ops are
used in the script; run with `CUDA_VISIBLE_DEVICES=` if you want to be extra sure.

## Input / output spec

Fixed shapes, batch size 1, type-token axis padded to `T = MAX_TYPES = 8`
(`src/typeid2elem/augment.py:MAX_TYPES`) regardless of how many types the real
structure has (1..8, the training-time cap):

| name | shape | dtype | meaning |
|---|---|---|---|
| `rdf` | (1, 8, 8, 64) | float32 | partial RDF `g_ab(r)`, 64 bins over 0..8 Å |
| `pair_extra` | (1, 8, 8, 8) | float32 | log1p(coordination numbers) + NN-distance stats + peak position/height, see `descriptors.py` |
| `frac` | (1, 8) | float32 | stoichiometric fraction per type |
| `glob` | (1, 3) | float32 | [log number density, has_cell flag, 1/T] |
| `mask` | (1, 8) | float32 (0.0/1.0) | 1.0 for real type slots, 0.0 for padding |

Output:

| name | shape | dtype | meaning |
|---|---|---|---|
| `log_probs` | (1, 8, 94) | float32 | log-softmax over 94 element classes (class `i` = atomic number `i+1`) per type slot |

Padding convention matches `src/typeid2elem/data.py:collate` — real type slots come
first (`rdf[:, :t, :t]`, `frac[:, :t]`, `mask[:, :t] = 1`), the rest is zero-padded with
`mask = 0`. **Only read `log_probs[0, :t]`** for a structure with `t <= 8` real types;
padded-slot outputs are not meaningful (their queries still run through the network,
they're just never used). If a real structure has `t > 8` types, it's out of scope for
these checkpoints — training capped at `MAX_TYPES = 8` (`augment.py`).

**Descriptor computation is not part of the export.** Turning raw `positions` /
`type_ids` / `cell` into `rdf`, `pair_extra`, `frac`, `glob` must be reimplemented
on-device from scratch — the reference implementation is
`src/typeid2elem/descriptors.py:compute_features` (uses `matscipy.neighbours.neighbour_list`
for the pairwise cutoff search, `R_MAX=8.0` Å, `N_BINS=64`, Gaussian smearing
`SMEAR_BINS=1.0`, coordination-number checkpoints at `CN_RADII=(2,3,4,6)` Å, per-atom
median/10th-percentile nearest-neighbor distances). Multi-frame fusion
(`descriptors.py:average_features`) is a plain per-key mean across frames and can be
done on-device too if needed, but is not part of the exported graph.

## Checkpoint-driven behavior

The export wrapper (`ExportModel` in `scripts/export_model.py`) reads each
checkpoint's saved hparams and adapts:

- `hparams.two_pass` — if `True`, the exported graph runs the model's second
  attention pass (conditioned on first-pass soft element probabilities via
  `elem_embed`) and returns *that* output, matching `TypeSetClassifier.predict_probs`.
  If `False` (all 4 current production checkpoints), only the first pass runs.
- `hparams.use_env` — **not supported**. `ExportModel.__init__` raises
  `NotImplementedError` if a checkpoint has `use_env=True`. See "What breaks
  with `use_env`" below. All 4 current production checkpoints have `use_env=False`
  (it's a newly-added hparam; old checkpoints don't have the key at all and get
  the `__init__` default of `False`).

## Exporter workaround: manual attention reimplementation

`nn.TransformerEncoder(..., src_key_padding_mask=...)` exports to ONNX fine as-is
(opset 17, legacy `torch.onnx.export`, `dynamo=False` — the new `dynamo=True` path
needs the `onnxscript` package, not installed, and wasn't needed). But tracing it for
CoreML fails:

```
ERROR - converting 'int' op (located at: 'encoder/0/self_attn/...'):
TypeError: only 0-dimensional arrays can be converted to Python scalars
```

`F.multi_head_attention_forward`'s fast path records `aten::size`/`aten::Int` ops on
batch/seq-length for reshapes around the key-padding mask; even with fully static
`.expand()`-based shapes, coremltools' torch frontend can't resolve these to
constants. Fix: `ManualEncoderLayer` in `scripts/export_model.py` reimplements
`nn.TransformerEncoderLayer(norm_first=True, activation=gelu)`'s forward with explicit
`torch.matmul` attention (splitting `in_proj_weight`/`in_proj_bias` into q/k/v by hand,
reusing the *same* trained weight tensors — no retraining or weight copying) and an
additive mask bias instead of a boolean key-padding-mask kwarg. This is used for both
ONNX and CoreML exports (one code path, one thing to verify), even though ONNX didn't
strictly need it.

Verified against the original `TypeSetClassifier.forward` in plain PyTorch (no
export/trace involved) on random full-T=8 inputs: max abs diff of log-probs
= 5.7e-6–7.6e-6 across the 4 checkpoints, well under the 1e-4 target.

### Mask-fill value: -1e4, not -1e9 or -inf

The original model fills invalid pair-pool positions with `-1e9` before a masked max
(`model.py: h.masked_fill(~pmask.expand_as(h), -1e9).amax(2)`), and PyTorch's attention
uses `-inf` internally for padding. Both overflow float16
(max representable ≈ 6.55e4), which triggers a `RuntimeWarning: overflow encountered
in cast` during CoreML's fp16 weight/constant-folding pass. The export wrapper uses
`-1e4` everywhere instead — comfortably outside the range of real post-LayerNorm
activations (order 1-10) so it's still "effectively -inf" for masking purposes, but
representable in fp16.

### CoreML fp16 warning

Even with `-1e4`, the fp16 conversion pass for **every** one of the 4 checkpoints still
emits a `RuntimeWarning: overflow encountered in cast` from
`coremltools/converters/mil/mil/ops/defs/iOS15/elementwise_unary.py:889` during the "MIL
default pipeline" stage — some other internal constant (not one we control directly)
also overflows fp16 range during a cast. The fp32 conversion (`compute_precision=
ct.precision.FLOAT32`) is clean (no warnings). Per the task spec, the script treats any
warning caught during the fp16 `ct.convert()` call as a trigger to also produce the fp32
variant, so `<name>_fp32.mlpackage` exists for all 4 checkpoints. The fp16 `.mlpackage`
still converts successfully (this is a warning, not an error) — whether the fp16
precision loss is acceptable is untested here since CoreML can't run on Linux; if you
see accuracy regressions on-device, use the fp32 package.

## Results (all 4 production checkpoints, 2026-07-03)

All checkpoints share the same architecture (`d_pair=128, d_model=256, n_heads=8,
n_blocks=2, two_pass=False, use_env=False`), so param count and file sizes are
identical across checkpoints (only weights differ).

Params (`ExportModel`, i.e. the exported graph — matches the trained model minus the
`class_weights` buffer, which isn't part of inference): **1,790,174**.

| checkpoint | wrapper-vs-torch diff | ONNX parity max diff (n=20, T=1..8) | ONNX size | CoreML fp16 size | CoreML fp32 size |
|---|---|---|---|---|---|
| aug2sharp30 | 7.6e-06 | 2.67e-05 | 6.9 MB | 3.5 MB | 6.9 MB |
| openlam_sharp30 | 5.7e-06 | 2.10e-05 | 6.9 MB | 3.5 MB | 6.9 MB |
| sharp30 | 5.7e-06 | 2.48e-05 | 6.9 MB | 3.5 MB | 6.9 MB |
| soft15 | 5.7e-06 | 1.53e-05 | 6.9 MB | 3.5 MB | 6.9 MB |

All well under the required thresholds (wrapper self-check < 1e-4, ONNX parity < 1e-3).
CoreML trace-vs-torch diff (PyTorch JIT trace output vs. the eager `ExportModel`, both
in fp32, no CoreML runtime involved) was exactly 0.0 for all 4 — tracing this graph is
lossless, all remaining error budget is in the fp16 CoreML runtime itself which is
untestable here.

## What would break with a `use_env` checkpoint

`TypeSetClassifier` gained a `use_env` branch (gather over type tokens + masked-mean
pooling over sampled-atom neighbor environments — `model.py` lines ~87-103) alongside
this export work. It reads `batch["env_d"]`/`batch["env_t"]` (per-sampled-atom RBF
distances and gathered partner-type indices), which `collate()`/`features_to_batch()`
in `src/typeid2elem/data.py` **do not currently produce** — so the branch is dead code
for every checkpoint trained so far (`hparams.use_env` defaults to `False`, and the 4
production checkpoints predate the hparam entirely).

If a future checkpoint sets `use_env=True`, exporting it needs, at minimum:

1. `env_d`/`env_t` added as fixed-shape export inputs (shape `(1, T, M, K)` for some
   fixed sampled-atoms-per-type `M` and neighbors-per-atom `K` — currently unbounded/
   ragged in the training-time data pipeline, so a padding+masking convention would need
   to be defined and documented here, analogous to the `T`-padding done for `rdf` etc.)
2. The on-device descriptor pipeline would need to reimplement per-atom neighbor
   sampling and RBF expansion (`R_ENV=6.0` Å, `n_rbf` centers,
   `descriptors.py` doesn't currently expose this — it would need a new function
   alongside `compute_features`), not just the pair-marginal RDF it computes today.
3. The `torch.gather` in the env branch (`torch.gather(pemb, 1, idx.reshape(...))`)
   is dynamic-index gather over the type-token axis; worth checking early whether
   ONNX/CoreML export chokes on it the same way the attention fast-path did — gather
   ops are usually fine in ONNX but CoreML's torch frontend has historically been
   pickier about dynamic index tensors, so budget time to hit a similar workaround.
4. `ExportModel.__init__` currently hard-`raise`s on `use_env=True` rather than
   silently producing a wrong graph — that guard should stay until 1-3 are done and
   verified, and should be removed/updated as part of that work, not before.

None of this was implemented here — out of scope per the M6 task, which only covers
the 4 existing production checkpoints (all `use_env=False`).
