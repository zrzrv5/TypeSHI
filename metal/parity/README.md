# metal/parity/ — golden reference for the Metal port

`golden.json` is the source of truth the Swift `parity-check` diffs against. It is produced
by the **Python** pipeline, so it's authoritative and platform-independent.

- `gen_reference.py` — builds a small, fully-specified rocksalt structure (64 atoms, both
  types), computes reference descriptors with `typeid2elem.descriptors.compute_features`,
  runs the committed int8 ONNX model, and writes `golden.json`.
- `golden.json` — `input` (positions/type_ids/cell the Swift test rebuilds), `expected`
  (deterministic `rdf`/`pair_extra`/`frac`/`glob` the kernel must reproduce), `env_seed0`
  (sampled env for one seed — structural check only), and `log_probs`/`top1_per_type`.

Regenerate (any platform, no Metal needed):

```bash
uv run python metal/parity/gen_reference.py
```

Keep `golden.json` committed and regenerate it whenever the descriptor definition or the
deploy model changes — a stale golden silently passes a wrong port.
