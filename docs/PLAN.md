# Plan & Design Decisions

**Goal:** given (positions N×3, type_ids N×1, optional cell 3×3, optionally many frames) from an MD
file, output top-K element candidates per type id.

## Why descriptor + small set-network (not atom GNN)

- Input size N is unbounded (10⁴–10⁶ atoms); a per-atom model is expensive and hard to deploy.
- The *label* lives at the **type** level, not the atom level → aggregate atoms into per-type
  statistics first. This makes inference O(N) neighbor search once, then a tiny fixed-cost network.
- Deployable: descriptor computation is plain histogram math (portable to Swift/C++); the network is
  a small MLP-family model → trivially exportable (CoreML/ONNX). GNN scatter ops are not needed.

## Descriptors (v1 spec)

For a snapshot with T types, compute for every **ordered** type pair (a,b):

- **Partial RDF** g_ab(r): r ∈ (0, 8 Å], 64 bins (Δr = 0.125 Å), Gaussian-smeared (σ ≈ 0.5·Δr),
  normalized by ideal-gas expectation using partial number density ρ_b, so g→1 at large r for
  homogeneous systems. PBC handled by neighbor lists (minimum image / supercell). Without a cell,
  use a density estimate from the point cloud's convex hull volume and set `has_cell=0`.
- **Per-pair scalars:** first-peak position & height, running coordination number of b around a at
  r ∈ {2.0, 3.0, 4.0, 6.0} Å.
- **Per-type scalars:** stoichiometric fraction x_a.
- **Global scalars:** total number density (Å⁻³), has_cell flag, T (number of types).

Rationale: bond lengths + coordination + stoichiometry are exactly what a human expert uses to guess
chemistry. Partial RDFs contain all of these; scalars are shortcuts that ease learning.
Absolute distance scale is the signal → **no scale normalization, no scale augmentation.**

## Model (v1): type-graph DeepSets

Each type a is classified using the *set* of its pair channels {(g_ab, scalars_ab, x_b, is_self=δ_ab) : b = 1..T}:

```
h_ab = φ([g_ab bins, pair scalars, x_b, δ_ab])          # shared MLP, ~128-d
ctx_a = mean_b h_ab  ⊕  max_b h_ab                       # permutation-invariant over partners
logits_a = ψ([ctx_a, x_a, global scalars])               # MLP → ~89 element classes
```

- Handles any T (T is just the set size). T ≤ ~10 in practice → negligible cost.
- Optional v2: one round of attention between type tokens (tiny transformer) if DeepSets underfits.
- < 1M params. Loss: cross-entropy per type. Metric: top-1 / top-3 / top-5 accuracy per type.

## Augmentations (critical — bridges DFT training → dirty MD reality)

1. **Type splitting:** with p≈0.5 pick an element with ≥2·m atoms and partition its atoms into
   m ∈ {2,3} type ids (random partition; later also spatially-clustered partition). Both new types
   keep the same element label. This teaches the model that x_a and g_aa change under splits.
2. **Thermal jitter:** Gaussian position noise, σ ~ U(0, 0.25 Å) (MPTrj already has off-equilibrium
   frames; jitter still helps for high-T MD).
3. **Small random strain** (±3%) on cell+positions — mimics NPT fluctuation, keeps bond lengths ~physical.
4. **Supercell replication** for tiny cells (< ~1.5×r_cut) so RDFs are well-sampled — also mimics the
   large-N regime of real MD files.

## Multi-frame inference

- Descriptors: partial RDFs averaged over frames = standard time-averaged RDF → smoother input, free.
- Predictions: also average per-frame log-probs (log-pool). Both supported; compare on eval.

## Composition-aware decoding (v2, optional)

Per-type softmax is independent; real chemistry couples types (charge neutrality, co-occurrence).
Later: beam search over top-K per type scored by sum of log-probs + a composition prior
(element co-occurrence statistics mined from MP). Keep v1 simple.

## Data plan

- **COSMOS DBS** (`dbs_total.extxyz`): ASE `iread` streaming → per-structure descriptor extraction.
- **MPTrj** (12GB json): `ijson` streaming; sample ≤ ~3 frames per mp-id (diversity over volume).
- Offline preprocessing (multiprocessing, CPU) → `Data/processed/*.npz` shards:
  ragged arrays of (pair features, type labels, structure metadata).
- Split by **material/composition**, not by frame, to avoid leakage (same formula in train & val).
- Class imbalance (O, Li dominate MP): weighted sampling or class-balanced loss; measure per-element recall.

## Baselines

1. Mass lookup when `Masses` present (exact; sanity check only).
2. Classical scorer: match first-neighbor distances against covalent/ionic radii sums + stoichiometry
   vs MP composition statistics. If ML can't beat this, rethink.

## Milestones

- [x] M0 Environment + data inventory
- [x] M1 Research report (agent) + data stats (agent) → docs/RESEARCH.md, docs/DATA.md
- [x] M2 Parsers (LAMMPS data, extxyz, MPTrj json) + descriptor pipeline + tests on eval files
- [x] M3 Preprocess COSMOS (small, fast) → first training run → sanity metrics
- [x] M4 Preprocess MPTrj subset → full training → top-K eval on 3 real LAMMPS files + LPSC MD
      (E2: top-1 12/19, top-3 16/19, top-5 17/19 — see docs/EXPERIMENTS.md)
- [x] M4b Checkpoint ensemble (E4): top-1 14/19, top-3 17/19, top-5 18/19 on real targets
- [x] M5 Fix open problems round: OMol-box fix (kept), decode prior (weak keep), debias/TTA/deep
      ensembles (rejected), dilute augmentation (kept as ensemble member) → E5-E7
- [ ] M7 Round 2 (research-driven, see docs/RESEARCH2.md): openLAM training data (in progress),
      WBM large-scale test, two-pass soft-embedding refinement (implemented, training queued),
      MLIP verifier best-of-N (prototyped; discriminativity check running), then RAFT distillation.
      RL verdict: best-of-N + rejection-sampling distillation first; policy gradient only if
      headroom remains (see RESEARCH2 Q5).
- [x] M6 Export path (ONNX + CoreML .mlpackage, scripts/export_model.py, docs/EXPORT.md):
      all 4 production ckpts exported, ONNX parity < 3e-5, 3.5 MB fp16 mlpackage, 1.79M params.
      Still open: env-branch export, packaged CLI, calibration (RAPS conformal top-K).
- [x] M8 Round 3 (E12/E13): env encoder (+3.8 val top-1, solo=quad on WBM; in production),
      HfO2 DPA domain (HfBaO-pog Hf rank 46->1), relaxed-WBM eval (≈ initial; E11 caveat
      withdrawn), hot-MD fine-tune (neutral at 364 trajs). Production = 6-model ensemble:
      bench 32/43/46, WBM 27.5/45.1/53.9.
- [x] M9 Round 4 (E14/E15): conf-fusion default (specialist rescue, Hf 8.9%→41%), unit
      inference + not-atoms gate (units.py; DEM gated; Angstrom prior after TT1 lesson),
      RAPS conformal sets (T=3.6, 91.7% coverage; predict.py --conformal), COD ingestion
      (55k CIFs → env_cod member, best solo + best unseen-WBM; swapped for env_hfo2).
- [x] M10 full-COD retrain (E16): 535k CIFs → 710k records, 42% of the mix. Single
      `env_codfull` model matches the 6-model ensemble on the real-file bench
      (78-81/106-108/120-122 of 131); swapped into production. Ensemble now mostly buys WBM
      stability. Biggest single accuracy lever in the project.
- [x] M11 deployment (E17): no-torch lite path (`predict_lite.py`, 166 MB/0.8 s), int8 model
      (1.94 MB, accuracy-neutral), interior-crop cap (10⁶ atoms 1.1 s/0.5 GB), 4-draw env
      pooling (kills the sampling-seed lottery). Array API `predict_structure(pos, type_ids,
      cell)`. ONNX + CoreML export (`export_model.py`).
- [x] M12 GitHub packaging: committed `weights/` runtime bundle (2 MB: int8 model +
      costats + calib) with an asset resolver (`assets.py`) so a bare clone runs; full
      artifacts (stripped ckpts, ONNX, CoreML) as a Release tarball (`package_release.py`);
      eval cases committed; docs (README/AGENTS/REPORT/MODEL/REPRODUCE) written.
- [ ] M13 on-device (Apple/Metal): `metal/` Swift package — `InaccurateRDFCalculator`
      descriptor kernel → CoreML. Skeleton + parity golden landed; remaining = envSample
      kernel + 4-draw pooling, cell-list neighbor search, atomic-min for nn_p10, port of
      composition decode + RAPS conformal, triclinic PBC. See `metal/README.md`.
- [ ] M14 accuracy/coverage open: learned gating/stacking over ensemble members (conf-fusion
      is only a partial specialist rescue); more AIS-Square domains (ZrO2, W, CuZr glass, SSE,
      perovskites); family-aware metrics for the f-block; packaged inference CLI / GUI.

## Decision log

- 2026-07-03: descriptor+set-net over GNN (deployability, N-independence). Partial RDF as core
  descriptor; SOAP/ACSF rejected for v1 (per-atom cost, species-dependent parameterization is
  circular when species are unknown — SOAP needs species to define channels; our channels are
  *type ids*, which is fine for RDF but breaks pretrained per-element models).
- 2026-07-03: subagent usage — web survey → sonnet agent; MPTrj/COSMOS stats → sonnet agent;
  architecture/design/implementation kept in main (fable) session.
- 2026-07-03: train/val split key = md5 hash of the sorted **element set** (`group_id`), val = gid%10==0.
  Stronger than per-material split: all Li-P-S structures land on one side, and it also deduplicates
  across COSMOS↔MPTrj overlap.
- 2026-07-03: molecules (no cell / pbc=F) are **included** via a non-periodic descriptor path
  (convex-hull density, KD-tree neighbors). Slabs with mixed pbc are treated as clusters — known
  v1 approximation.
- 2026-07-03: types with **zero atoms** in a file (NaCBH type 2 = C) are unidentifiable from geometry
  by construction; report them as such, fall back to mass lookup when Masses present.
- 2026-07-03: added LPSC (Li6PS5Cl r2SCAN MD, shipped inside the COSMOS zip) as a held-out finite-T
  eval + multi-frame fusion testbed. Eval harness: `scripts/evaluate_real.py`.
- 2026-07-03 (round 3): RAFT distillation REJECTED for now. RAFT distills verifier-chosen labels,
  but every corpus we are allowed to train on already has true element labels, so verifier labels
  are strictly noisier than what we own; the verifier adds value only at inference on unlabeled
  user files (kept as optional verify_rerank). What the RL/verifier discussion actually exposes is
  the near-equilibrium-DFT -> hot-MD domain gap, attacked directly by scripts/make_md_frames.py:
  MACE-MP-0 NVT (300-900 K) on train-split MPTrj materials -> real thermal ensembles with free
  true labels -> fine-tune. Val element-sets (gid%10==0) excluded at generation time.
- 2026-07-03 (round 3): descriptor v3 adds per-type environment sets (env_d/env_t: M=16 sampled
  atoms x K=16 nearest neighbors within 6 A as (distance, partner type id)); model gains a
  learned species-blind encoder branch (use_env) that pools (RBF(d), partner type token) over
  neighbors then atoms -- restores the per-atom JOINT coordination that pair-marginal RDFs
  discard (e.g. "every Li sees 4 O and 2 Cl", not just marginal Li-O and Li-Cl histograms).
- 2026-07-04 (E18): generative output head (discrete diffusion / discrete flow matching)
  REJECTED. Controlled head swap on an identical trunk (masked absorbing-state diffusion ==
  discrete flow matching under the masking interpolant): baseline independent-softmax 71.5/85.1
  /89.2 vs diffusion joint-decode 68.4/82.7/87.3 (val, same data/epochs). ~3 pts worse, the
  joint decode never beat its own marginal (attention already models sibling structure, cf.
  the E10 two-pass null result), and never reached the free hand-built PMI beam. Cause: the
  masked objective leaks sibling labels and weakens the geometry→element encoder. Also breaks
  single-pass ONNX/CoreML export (T sequential passes). The ceiling is geometric identifiability,
  not output-model expressiveness. Kept `model_diffusion.py` + `scripts/exp_diffusion.py` as
  record; retained the `TypeSetClassifier.encode()` trunk refactor.
- 2026-07-04: distribution split — a 2 MB `weights/` bundle is COMMITTED (int8 deploy model +
  costats + calib) so a fresh clone runs `predict_lite.py` with no download; the full artifacts
  (optimizer-stripped ckpts ~7.4 MB, per-member ONNX, CoreML packages) are a GitHub Release
  tarball, not committed. `src/typeid2elem/assets.py` resolves each asset from `weights/` first,
  then falls back to `runs/`/`Data/` so both a bare clone and a full working tree run unconfigured.
- 2026-07-04: `Data/eval_cases/` (MLFF POSCARs + MDRun LAMMPS data, 1.2 MB) is committed so the
  real-file benchmark is self-contained; the rest of `Data/` (training corpora) stays gitignored.
  Public release goes out with a permissive posture (repo public; see README license line).
- 2026-07-04: on-device roadmap committed as the `metal/` Swift package skeleton
  (`InaccurateRDFCalculator` → CoreML). "Inaccurate" is a design license: the GPU RDF may
  subsample centers / use a coarse neighbor search because chemistry is decided by bond-length
  peaks + coordination, both subsampling-robust. Parity is defined by predictions (top-1 + 90%
  set), not bit-exact descriptors; `metal/parity/golden.json` is the Python-generated target.
