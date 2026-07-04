# Experiment Log (append-only)

## 2026-07-03 — E0: smoke test (pipeline validation)

- **Data:** 5k COSMOS frames + 2k MPTrj materials (test shards), 1 clean + 2/1 aug variants → ~22k records.
- **Model:** TypeSetClassifier d_pair=128, d_model=256, 2 attention blocks, 1.8M params.
- **Train:** 2 epochs, bs 256, bf16, RTX 4090. Class-weighted CE (1/sqrt(freq), clip 10).
- **Result:** val top-1 = 0.392 (94 classes). Pipeline works end-to-end (train → ckpt → predict CLI).
- **Predict CLI on TT1.data (Li Al Cl O):** wrong, as expected for a smoke model, but type 2 → O@28%
  and Al-ish confusion patterns already visible.
- **Conclusion:** proceed to full-data training.

## 2026-07-03 — E0b: neighbor-list bottleneck

- Full COSMOS preprocessing stalled: ASE `primitive_neighbor_list` (pure Python) takes 0.5–2 s/frame on
  OC20 slab frames (vacuum cells). Whole-file ETA exploded.
- **Fix:** switched `descriptors.py` to `matscipy.neighbours.neighbour_list` (C implementation).
  80× faster on the slow block (200 frames: 24.2 s → 0.3 s; LiSiPS 2429 atoms: 0.27 s → 0.02 s),
  output verified identical (closest-approach distances match to 2 decimals, no NaNs).
- Lesson: always use matscipy for neighbor lists in this project.

## 2026-07-03 — E1: COSMOS-only, features v1 → real-world failure analysis

- **Data:** full COSMOS DBS, 401,310 records (1 clean + 2 aug). First attempt NaN'd: 6 `inf` RDF values
  (fp16 overflow — near-isolated molecules have tiny ideal-gas density ⇒ astronomically large g).
  Fixed by clipping g ≤ 1e4 at source + gradient_clip_val=1.0; shards repaired in place.
- **Result:** val top-1 = **76.3%** / 94 classes (held-out element-sets) after 15 epochs.
- **Real-world eval (scripts/evaluate_real.py): top-1 2/19, top-3 7/19.** Failure modes:
  1. **Corrupted bond-length feature at finite T.** "Closest approach" = first non-empty histogram
     bin is an extreme-value statistic: at 400–600K one hot pair among thousands shortens it by
     ~0.4 Å (LiSiPS P–S read 1.56 Å ⇒ model says "P–O bond" ⇒ predicts O for S with 92% confidence;
     same on LPSC). This single feature explains the systematic S→O flips.
  2. **Alkali confusion** Li→Ag/Sr/Na (similar radii & coordination in sulfides) — geometrically
     honest; needs composition context (MPTrj has the Li-P-S chemistry; COSMOS DBS is molecule-heavy)
     and joint decoding priors (v2 roadmap).
  3. Multi-frame fusion changed little — biases are systematic, not noise (as theory predicts).
- **Conclusion → features v2:** replace closest-approach with per-atom nearest-neighbor distance
  **median + 10th percentile** per type pair (robust to thermal outliers) + smoothed-RDF peak
  position/height; N_PAIR_EXTRA 5→8. Jitter σ_max 0.25→0.35 Å. Verified on LiSiPS 400K: P–S median
  NN = 1.98 Å, peak = 2.06 Å (correct; v1 said 1.56 Å).

## 2026-07-03 — E2: combined COSMOS+MPTrj, features v2

- **Data:** 401,310 COSMOS + 543,712 MPTrj records (v2 features), train 868,184 / val 76,838.
- **Train:** 15 epochs bs 1024, class-weighted CE, wandb run `combined`.
- **Val:** top-1 = 62.0% (94 classes; harder val mix than E1 — not comparable). Best ckpt = last epoch
  ⇒ undertrained.
- **Real-world eval: top-1 12/19, top-3 16/19, top-5 17/19** (E1: 2/19, 7/19, 7/19).
  - LiSiPS: Li✓ (58.8%), P✓ (68.9%), S✓ (99.9%); Si rank 10 (only 1.9% of atoms — dilute-type problem).
  - TT1 (38 atoms, no masses): Cl✓ (75.1%), O✓ (48.7%), Al rank 2, Li rank 6 (tiny-cell statistics).
  - NaCBH: B✓ (100%), H✓ (96.2%), Na rank 2 (Ag 72% — classic Na/Ag radius confusion).
  - LPSC 1 frame: Li✓ P✓ S✓, Cl rank 3 → all four in top-3.
  - **Multi-frame fusion (8 frames) slightly HURT** (P r=1→2, Cl r=3→4). Log-pool over correlated
    frames sharpens wrong modes; needs investigation (maybe drop per-frame pooling, keep only
    descriptor averaging).
- **Conclusions:** robust NN-distance features fixed the S→O catastrophe (S now 96-99.9%). Remaining
  weaknesses: dilute types (<2% fraction), alkali/noble-metal radius twins (Na/Ag, Li/Ag),
  tiny systems. Next: E3 longer training; later: composition-prior decoding, dilute-type
  augmentation (drop-fraction augmentation), fusion ablation.

## 2026-07-03 — E3: E2 + 30 epochs

- Same data/config as E2, 30 epochs, wandb run `combined_e30`. Val top-1 = 62.9% (best = last epoch
  again — LR schedule ends there; longer training may still help marginally).
- **Real-world: top-1 13/19, top-3 17/19, top-5 17/19.**
  - Huge sharpening: LiSiPS Li 97.6%, P 98.7%, S 95.5%, Si rank 10→3; NaCBH now all-correct
    (Na 90.4%, B 100%, H 95.8%); LPSC Li 99.5% / P 98.5% / S 96.3%.
  - **But TT1 regressed** (Li r6→12, Al r2→16): the sharper model commits to BiOCl/FeOCl chemistry —
    a chemically seductive wrong basin for a 38-atom Li-Al-Cl-O cell. Overconfidence on tiny samples.

## 2026-07-03 — E4: checkpoint ensemble (E2 ⊕ E3, mean log-probs)

- **Real-world: top-1 14/19, top-3 17/19, top-5 18/19.** Best so far. Soft E2 + sharp E3 average out
  each other's failure modes (NaCBH Na stays correct; TT1 Al back into top-5 at r=5; LPSC/LiSiPS keep
  E3's sharpness). Only TT1-Li (r=10, Bi confusion) remains outside top-5.
- `scripts/evaluate_real.py` now accepts multiple checkpoints and ensembles them.
- **Open problems (priority order):**
  1. Tiny-sample overconfidence (TT1): candidate fixes — deep ensemble (3 seeds), jitter-TTA,
     composition-prior joint decoding (charge neutrality strongly disfavors all-cation Bi+Fe+Cl+O
     assignments... actually BiOCl is neutral — need the Li signal: no Bi-Bi at 3.5 Å etc.).
  2. Anion-site confusion Cl vs S in argyrodite (LPSC Cl r=3) — Cl sits on S-like sites; genuinely hard.
  3. Dilute types (LiSiPS Si at 1.9%, r=3) — drop-fraction augmentation.

## 2026-07-03 — E5: extended benchmark + inference-side ideas (tried & root-caused)

**Extended benchmark** (`scripts/evaluate_bench.py`): 13 systems / 43 scoreable type-ids, built from
files a subagent collected across ~/Documents (see docs/EVAL_CASES.md): original 4 targets + TT1 at
4750 atoms (pdb) + LACO 150-atom + LPS glasses + NaBH 300K + NiTi alloy + CHO & P2S6 molecules +
elemental-Na MD with 3 split type-ids. Model always geometry-blind; truth from symbols/masses/dirnames.

Ideas tried this round (config → top1/top3/top5 of 43):
- **E2+E3 ensemble baseline: 27/36/39** (63%/84%/91%).
- **OMol-box fix (kept):** cell-less molecules were catastrophic (P2S6 rank ~50) because ALL training
  molecules have OMol vacuum boxes (extent+10 Å, pbc TTT) and the convex-hull no-PBC path was OOD.
  compute_features now reproduces the OMol convention for cell-less input. CHO 2→3 top-1, P2S6 0→2 top-5.
  No retraining needed (training frames all have cells).
- **Debias (rejected):** hypothesis that 1/√freq class weights tilt posterior toward N (O→N, C→N
  confusions) — post-hoc logit adjustment γ·log(w) at γ∈{0.3,0.5,1} did NOT help (27→27→26 top-1).
  The O→N confusions are geometric (RMC-fitted structures have stretched bonds); top-3 catches them.
- **Composition-prior decode (weak keep):** PMI + bounded charge-neutrality reranking
  (`src/typeid2elem/decode.py`, stats mined from 60,836 unique training element sets by
  `scripts/mine_costats.py`). +1 top-3 / −1 top-5; neutrality term too weak vs 2-nat model confidence
  gaps (TT1-38at: Li-Al-Cl-O is exactly neutral, Bi-Fe-Cl-O is not, still loses). Off by default.
- **Deep ensemble (negative result):** +2 same-recipe seeds (val 62.8/63.0%) → 4-model top-1 DROPS
  27→22. Homogeneous sharp seeds outvote the complementary soft E2. Diversity > size under domain
  shift; keep E2(15ep)+E3(30ep) as the production pair.
- **TTA jitter (no gain):** −1 to −2 top-1 at every setting tried.

**Failure taxonomy after E5** (what's left):
1. Dilute/minority & uneven-split types (Na-metal split types r=61/70, LiSiPS Si r=3) → train-time fix (E6).
2. Intermetallic sublattices (NiTi B2: geometrically near-identical sites; both types get the same
   ~flat distribution) — information-limited; top-5 acceptable.
3. 38-atom TT1 Li — small-sample limit (4 Li atoms); solved at 4750 atoms (Li 75% top-1).
4. RMC-fit structures read stretched (O→N) — source-domain artifact, top-3 catches.

## 2026-07-03 — E6: dilute-type augmentation retrain

- augment v2: Dirichlet(0.3) uneven type splits (33% of type instances now <5% fraction) +
  new `subsample_type` (delete 50-95% of one type). Reprocess → `Data/processed/{cosmos,mptrj}_aug2`,
  train `aug2_soft` (15ep, val 60.3%) + `aug2_sharp` (30ep, val 61.2%).
- **aug2 pair alone: mixed/negative** (26/34/37 vs 27/36/39 for E2+E3) — fixed its targets
  (TT1-5x5x5 → 3/4 top-1, P2S6 top-1) but over-skewed training off stoichiometric reality.
- **E2+E3+aug2_sharp trio: 29/36/38 → best.** Heterogeneous *recipes* ensemble well where
  same-recipe seeds (E5) failed. +decode: 29/37/38.
- **Production config** (`runs/production/{soft15,sharp30,aug2sharp30}.ckpt`, decode ON):
  `scripts/predict.py` defaults to it.

## 2026-07-03 — E7: benchmark corrections (user) + HfBaO cases → final round numbers

- User corrections: dropped DimTest case (nonsensical); test_Na is actually **Na-S-P-O**
  (per its input.lammps masses) — the model's "S" prediction for type 2 had been correct and was
  mis-scored under the wrong "all-Na" truth; LACO/RMC TT3GR+TT3RR added as Li Al Cl O;
  added 2 HfBaO (Ba-doped amorphous HfO2, truth from masses: Hf/Ba/O).
- **Final corrected benchmark (16 systems, 52 type-ids), production trio + decode:**
  **top-1 32/52 (62%), top-3 43/52 (83%), top-5 46/52 (88%).**
- New failure surface from HfBaO: amorphous heavy-metal oxides — Hf→Sb/Sn/Te (r=4 and r=46 on two
  files), Ba→Rb/K (r=5). Radius-twin confusions + amorphous strain; MPTrj is crystalline-biased.
- decode win observed: NiTi Ti→top-1 via Ni-Ti co-occurrence PMI.
- Remaining priorities: (1) amorphous heavy-metal oxides (data gap — could add melt-quench-like
  augmentation or heavier strain/jitter), (2) dilute P in NaSPO (r=61), (3) neutrality weight
  tuning on a held-out synthetic benchmark rather than by hand.

## 2026-07-03 — E8: MLIP physics-verifier (best-of-N) prototype

- `src/typeid2elem/verifier.py`: MACE-MP-0-small force-RMS on the observed geometry per candidate
  joint assignment (from decode's beam), combined score = logp − λ·log1p(fRMS/f0).
- **Discriminativity check** (`scripts/verifier_check.py`, 57 labeled COSMOS structures, radius-twin
  swaps): **truth wins 86%** overall — O→N 38/41, Li→Bi & Ba→Rb & K→Cs & Ti→Zr all correct;
  **Si→P only 4/8 (coin flip — too chemically similar)**.
- Case studies: LiSiPS — verifier sharpens correct answers (Li 98%, S 100%) but can't fix dilute Si.
  **TT1 — INVERTED**: MACE force-RMS is LOWER for the Bi-Fe trap (1.57 eV/Å) than for the true
  Li-Al-Cl-O (2.74 eV/Å); TT1 is an RMC-fitted structure with genuinely stretched bonds, so the
  model's Bi call is geometrically rational. **HfBaO — candidate-pool-limited**: Hf at rank 46 never
  enters the beam, verifier can't select what's not proposed.
- Verdict: viable as optional "slow mode" for near-equilibrium inputs; not a cure-all. Next per
  research plan: RAFT distillation once candidate pools contain the right answers.

## 2026-07-03 — E9: openLAM training data (user-added)

- Parsed DeePMD npy trees (`scripts/preprocess_deepmd.py`). **Pitfall found & fixed:** Anode domain
  uses DeePMD *mixed-type* format (real species in `set.*/real_atom_types.npy`, virtual type −1
  padding; `type.raw` meaningless) — first training run was label-poisoned and was killed/redone.
- 19,070 records (Alloy/Anode/Cluster/FerroEle; Drug skipped — organics already abundant). Element
  profile: Pt/Au/Pd/Ag/Cu/Ni alloys + Ba/Pb/Ti/O. Oversampled 3× in training.
- `openlam_sharp` (30ep): alone 28/38/40 of 52. **Quad ensemble (trio + openlam_sharp): 33/41/46;
  +decode: 34/42/44 — new top-1 best (65%).** NiTi: Ti✓ top-1, Ni r=2 (was flat/clueless).
  HfBaO Hf still unfixed — our tarballs lack the DPA-2 HfO2 domain (only 36 Hf instances); getting
  that domain from AIS-Square is the obvious next data move.

## 2026-07-03 — E10: two-pass soft-embedding refinement

- `two_pass=True` in TypeSetClassifier: pass 2 re-encodes type tokens + soft element embeddings
  (softmax-weighted, differentiable), joint loss on both passes. Trained 30ep on aug2+openlam
  (`twopass_sharp`, val 57.4% vs 57.3% plain — marginal).
- Bench: alone 25/39/42 (trades top-1 for top-3 vs plain openlam_sharp 28/38/40); quint ensemble
  33/42/46 vs quad 34/42/44. **Verdict: neutral at this data/model scale** — differences are ±1-2
  of 52 (noise). Keep the code; revisit after the learned species-blind encoder (RESEARCH2 Q1),
  where richer per-pass features give refinement more to work with. The 52-type bench cannot
  resolve <±4% effects — WBM-scale eval needed for such comparisons.
- **Production updated to the quad** (soft15 + sharp30 + aug2sharp30 + openlam_sharp30, decode ON):
  34/52 top-1, 42/52 top-3, 44/52 top-5.

## 2026-07-03 — E11: WBM large-scale test (initial structures)

- figshare thaw blocked the official extxyz for hours → used HF mirror `nimashoghi/wbm` parquet
  (256,964 rows incl. `initial_structure` + `unique_prototype`; relaxed structures NOT mirrored —
  slow poller running for figshare 48169600). Evaluator: `scripts/evaluate_wbm.py --parquet ...`.
- **Production quad on 4,000 unique-prototype initial structures (12,158 type-ids):
  top-1 27%, top-3 45%, top-5 53%.** Crucially, seen vs unseen element-set buckets are equal
  (27.0% vs 27.5%) → no composition memorization; the element-set split is doing its job.
- **Caveats making this a LOWER bound:** WBM initial structures are element-substituted prototypes
  BEFORE relaxation — bond lengths still reflect the original elements, so geometry↔label pairs are
  partially inconsistent by construction. Relaxed-WBM is the fair test (pending figshare thaw).
- **Diagnostics:** accuracy RISES with type count (T=2: 21% top-1 → T=4: 40% → T=5: 53%) — more
  context, more constraints. Hardest elements = f-block (Np/Pa/Pu/Th ~0-6% top-3; Gd/Tb/Dy/Ho ~11%)
  + interchangeable +3 cations (Y, Mn, Cr) — lanthanide contraction makes these geometrically
  near-degenerate (~0.01 Å radius differences); irreducible without electronic signals. Easiest:
  anions (O 94%, I 95%, Br 94%, F 92%, B 91% top-3).
- Implication: report "family-aware" metrics (right lanthanide vs exact lanthanide) and use
  abstention/conformal sets for the degenerate families rather than chasing exact-match there.

### Pipeline facts worth remembering
- Descriptor throughput: ~45-55 structures/s/8 workers (COSMOS ~44 f/s incl. 2 augs; MPTrj ~53 f/s incl. 1 aug).
- Full COSMOS DBS (134k frames, n-aug 2) ≈ 45 min @ 14 workers → ~400k records.
- NaCBH eval file declares 4 types but type 2 (C) has **zero atoms** → geometry can never identify it;
  only 3 types are scoreable there. Eval counts only present types.
- LiSiPS + NaCBH have Masses sections (mass lookup = exact); TT1 has none → the true ML test case.
- LPSC (Li6PS5Cl) r2SCAN MD trajectory extracted to
  `Data/COSMOS/benchmark_results/LPSC_MD/r2scan_DFT.extxyz` (416 atoms/frame) — held-out finite-T eval;
  also used to test multi-frame fusion.

## 2026-07-03 — E12: relaxed WBM + learned env encoder + HfO2 domain (round 3)

**Relaxed WBM (kills the E11 caveat).** Agent found the warm 2022-10-19 figshare file
(`wbm-computed-structure-entries.json.bz2`, id 40344463; the frozen 48169600 was only the 2024
re-upload). Verified genuinely relaxed (mean |Δr| 1.67 Å vs initial on spot check).
`evaluate_wbm.py --cse-json ...`, production quad, 4,000 unique prototypes (12,079 type-ids):
**top-1 24.2%, top-3 43.6%, top-5 54.0%** — statistically the same as initial structures
(27/45/53; different random subsample). The "initial structures are a lower bound because of
geometry-label mismatch" hypothesis did NOT materialize: our descriptors are robust to
relaxation-scale geometry changes, and WBM errors are chemistry-identifiability limits
(radius-degenerate cations, f-block), not unrelaxed-geometry noise.

**Learned species-blind env encoder (v3 features, `use_env`).** Per type: M=16 sampled atoms ×
K=16 nearest neighbors ≤ 6 Å stored as (distance, partner type id); model branch embeds each
neighbor as [RBF16(d), partner-type token] → masked DeepSets over neighbors → over atoms →
additive type-token update (+56k params). Restores per-atom JOINT coordination that pair-marginal
RDFs discard. All shards regenerated as `*_v3` (964k records, counts identical to v2).
- val top-1: **0.611 vs 0.573** for same-data non-env (openlam_sharp) → **+3.8 pts**.
- WBM-initial solo: **27.2%** vs 26.2% non-env solo → +1.0 pt; a single env model ≈ the whole
  4-model ensemble (27.0%). Zero seen/unseen gap.
- Real bench solo: 26/38/46 (over 52) — solo top-1 below quad ensemble but top-5 above it.
- Gain is distribution-dependent: biggest on diverse val (molecules/alloys/MD-ish), small on
  ordered WBM crystals where pair marginals already suffice.

**HfO2 DPA domain (AIS-Square id 145, 114 systems / 57k frames).** `hfo2_v3` = 3,518 records
(max 20 frames/sys). env+hfo2 model (val 0.612): **HfBaO-pog Hf rank 46 → rank 1 (99.6%)** —
targeted data collection works. Amorphous HfBaO-amor still hard (Hf r=6, Ba r=5).
Bench solo 28/37/43.

**Ensembles (bench top1/top3/top5 over 52; WBM-initial top1/3/5):**
| set | bench | WBM |
|---|---|---|
| quad (old production) | 34/42/44 | 27.0/45/53 |
| quad+env | 33/42/46 | 27.4/45.0/53.8 |
| quad+env+hfo2env (6) | 32/43/46 | **27.5/45.1/53.9** |
| trio+env+hfo2env (5) | 30/42/48 | — |

Env members trade ~1-2 bench top-1 (n=52, noise-level) for +2 top-5 and the best WBM numbers.
NOTE: log-prob averaging dilutes specialists — solo env+hfo2 has Hf@99.6% on HfBaO-pog, but 5
Hf-ignorant co-members drag it to rank 3. Weighted/gated ensembling is an open idea.

**Export (M6 closed).** `scripts/export_model.py` + `docs/EXPORT.md`: ONNX (opset 17) + CoreML
.mlpackage for all production ckpts, parity ≤ 3.1e-5; CoreML needed a manual re-implementation of
TransformerEncoderLayer (tracer chokes on src_key_padding_mask) and python-constant shapes in the
env gather. Env ckpts export too (7.4 MB, extra fixed-shape inputs env_d/env_t (1,8,16,16)).

**RAFT verdict recorded in PLAN decision log:** rejected — all trainable corpora have true labels;
verifier labels strictly noisier. Replacement experiment: MACE-MP-0 NVT (300-900 K) hot frames on
train-split MPTrj materials (`scripts/make_md_frames.py`), fine-tune with true labels (E13).

## 2026-07-03 — E13: hot-MD fine-tune (RAFT replacement) — NEUTRAL

- `scripts/make_md_frames.py`: MACE-MP-0-small NVT Langevin (300-900 K, 2 fs, 300 steps,
  3 frames/traj) on every 50th MPTrj material, train-split element sets only, melted/exploded
  guard. Snapshot at 364 trajectories = 1,092 frames -> `hotmd_v3` (3,276 records, n-aug 2).
- Fine-tune (`train.py --init-ckpt`, new flag): env_hfo2_sharp30 + 2 epochs lr 1e-4 on base v3 mix
  with hotmd oversampled 20x (~6% of steps). val 0.610 (from 0.612).
- Bench: 28/36/44 vs 28/37/43 pre-FT; the four finite-T MD systems are IDENTICAL except LiSiPS
  top-5 3->4. **Neutral** — robust NN-median features + jitter augmentation already cover thermal
  disorder at this trajectory count. Not promoted. Generation continues to 1,200 trajs
  (Data/MDgen/) for a possible larger-scale retry (train-time aug rather than fine-tune).

## 2026-07-03 — PRODUCTION UPDATE (round 3)

`runs/production/` = 6 ckpts: soft15, sharp30, aug2sharp30, openlam_sharp30, **env_sharp30,
env_hfo2_sharp30** (+ decode, default in predict.py):
- real bench **32/52 top-1, 43/52 top-3, 46/52 top-5** (old quad: 34/42/44 — traded 2 top-1
  [n=52 noise] for +1 top-3/+2 top-5; product surfaces top-K, so coverage wins)
- WBM-initial **27.5% / 45.1% / 53.9%** (best recorded), relaxed-WBM equivalent.
- Single-model deployment pick: env_hfo2_sharp30 (solo 28/37/43 bench, 7.4 MB ONNX / 3.8 MB
  CoreML fp16, exported in runs/export/).

## 2026-07-03 — E14: ensemble fusion, unit inference gate, conformal sets (round 4)

**Fusion rules** (`inference.fuse_logp`, `--fuse` on evaluators; production = 6-model ensemble):
| rule | bench (52) | WBM (12,158) |
|---|---|---|
| logmean (old) | 32/43/46 | 27.5/45.1/53.9 |
| probmean | 34/40/45 | 27.4/44.9/53.4 |
| **conf** (adopted) | 32/42/47 | 27.4/44.8/53.7 |
| median | 30/41/45 | — |

Globally all within noise; `conf` (per-type confidence-weighted arithmetic prob mean) adopted as
default because it repairs the demonstrated specialist-dilution failure at no global cost:
HfBaO-pog Hf 8.9% r3 → 40.9% r2; NiTi Ti → rank 1. Full specialist rescue would need gating that
knows WHICH model to trust (stacking on val) — deferred.

**Unit inference + not-atoms gate** (`src/typeid2elem/units.py`, auto in predict.py). NN-distance
window test ((0.6, 3.6) A) across candidate units (A, Bohr, nm, um, cm, m = LAMMPS unit styles);
no candidate fits → "NOT ATOMISTIC" (exit 2) — catches DEM/granular data (tested: mm-grain packing
in SI units → gated, message names the reason). Ambiguous candidates (A vs Bohr overlap for
organic-scale NN) tie-broken by model confidence under each rescaling.
LESSON: pure confidence tiebreak mis-called RMC-distorted TT1 as Bohr (distorted-but-A structures
look "more familiar" rescaled); fixed with an Angstrom prior — non-A needs +0.30 confidence margin
when A is window-plausible. Regression: LiSiPS/TT1/NaCBH × {as-is → angstrom, ÷0.529 → bohr} all
correct. Known edge case: true-Bohr organics may still resolve to A (rare in practice; --units
overrides).

**Conformal abstention** (`scripts/calibrate.py`, `predict.py --conformal`). Pipeline is heavily
overconfident: temperature = 3.6 (NLL 2.85 → 1.73) on 33k val type-ids. RAPS (lam .02, k_reg 4)
at 90%: empirical coverage 91.7%, set size median 7 / p90 9 → honest uncertainty for the
radius-degenerate families. calib.npz ships with production; sets print under each prediction.

## 2026-07-03 — E15: COD experimental data (round 4, partial download)

- COD rsync (anonymous, rsync://www.crystallography.net/cif/) — first 55,360 CIFs (of ~500k;
  download continues). `scripts/preprocess_cod.py`: skip partial-occupancy/disordered sites,
  2..2000 atoms, >=2 elements → 69,790 records (63% yield). Chemistry: silicates, perovskites,
  intermetallics, organics, rare earths — exactly the experimental diversity DFT sets lack.
- env_cod_sharp30 (base v3 + hfo2 + cod, use_env, 30 ep): val 0.624 (val now incl. COD sets —
  not comparable to 0.611). **Best solo model:** bench 29/39/41 top-1; WBM 27.2/44.3/52.7 with
  the best unseen-set bucket recorded (27.8 solo / 28.1 in ensemble).
- Ensembles: as 7th member neutral (31/42/47 bench). **Swapped for env_hfo2_sharp30** (whose
  training data it supersedes): production stays 6 models — bench 31/42/47,
  WBM 27.6/45.1/53.7, unseen 28.1/44.7/52.7, HfBaO-pog Hf 41.0% r2 preserved. All deltas within
  noise; swap chosen for best unseen generalization + strictly-superset data + slot economy.
- Recalibrated (T=3.6, qhat 0.9653, coverage 91.7%, median set 7). env_cod exported
  (ONNX parity 1.7e-5). Single-model deployment pick is now env_cod_sharp30.
- TODO when rsync completes (~500k CIFs): full re-preprocess + retrain this member; expect the
  COD contribution to grow ~9x.

## 2026-07-03 — bench expansion: MLFF cathode cases (52 -> 102 type-ids)

12 POSCAR cases from the user's MLFF share (see docs/EVAL_CASES.md; truths hand-verified, one
agent misreport caught). Production 6-model ensemble on the full 102: **56/82/94** (CONTCAR final geometries; initial-POSCAR variant scored 58/81/91)
(57%/79%/89%). New-case subtotal 27/39/44 of 50. Baseline for future rounds uses the 102-id bench.

## 2026-07-03 — bench expansion 2: MDRun cases + CONTCAR swap (102 -> 131 type-ids)

MLFF cases moved to CONTCAR final geometries (56/82/94 of 102). Added 6 MDRun cases (29 ids,
see docs/EVAL_CASES.md; notable: wrong-mass LFMP file where geometry beats mass lookup, and a
core-shell polarizable model scoring 4/6 despite duplicate near-zero-separation particles).
**Standing baseline: 78/107/122 of 131 (60%/82%/93%), production 6-model + conf fuse + decode.**

## 2026-07-03 — E16: full COD (535k CIFs) — solo model now matches the ensemble

- Full COD rsync: 534,816 CIFs / 111 GB → `cod_v3_full` = 709,842 records (10x the partial run;
  COD = 42% of the 1.52M-record training mix).
- TRAP: 2 of 710k records contained dummy species 'X' (Z=0 → label -1) and crashed training with
  an async CUDA device-side assert (surfacing in optimizer-to-device, not CE). Filter
  `zs.min() < 1` added to preprocess_cod.py; bad shard repaired in place. Also: a stray empty
  CUDA_VISIBLE_DEVICES kills GPU init with a misleading "you might not have a CUDA gpu" — use
  explicit CUDA_VISIBLE_DEVICES=0 for background training runs.
- env_codfull_sharp30 (val 0.763 — NOT comparable to earlier 0.61-0.62: val now 42% COD crystals).
- **Solo: bench 79/108/117 of 131 — a single 1.85M-param model beats the whole 6-model
  production ensemble (78/107/122) on top-1/top-3.** WBM solo 26.8/44.2/53.1.
- Swapped for env_cod_sharp30 in production (superset data). New production: bench
  **79/106/120 of 131**, WBM 27.5/45.0/53.5, best unseen top-1 recorded (28.3%). Recalibrated
  (T=3.6, qhat 0.9681, 91.6%); exported (ONNX parity 1.7e-5).
- Takeaway: experimental-crystal scale (COD) was the biggest single data lever so far; ensemble
  members now add little on top of the best member — next accuracy moves are learned gating or
  a bigger/longer single model, not more members.

## 2026-07-03 — E17: deployment: lite path, int8, interior-crop, env-draw pooling

- **predict_lite.py** (onnxruntime + numpy/scipy/matscipy; no torch/lightning): full pipeline
  (unit gate w/ ONNX-confidence tiebreak, decode, conformal). Footprint on LiSiPS (2.4k atoms):
  **166 MB peak RSS / 0.8 s** vs torch 6-model predict.py 946 MB / 2.5 s.
- **int8 dynamic quantization**: 7.42 MB -> **1.94 MB**, bench 81/106/120 vs fp32 79/108/117
  (noise) — accuracy-free 3.8x shrink. fp32 ONNX matches torch exactly (79/108/117).
- **Interior-crop cap** (descriptors.crop_for_inference, default 100k atoms): fractional-space
  crop, only complete-neighborhood atoms are RDF centers. TRAP FOUND: per-type density/fractions
  MUST come from the full system — crop boundaries cut sublattices unevenly and the composition
  bias rescales whole RDF rows. 952k-atom system: 10.6 s / 3.77 GB -> **1.1 s / 541 MB**,
  identical predictions after fix.
- **Env sampling variance discovered**: env sets = 16 sampled atoms/type; near-tie predictions
  (LiSiPS type 3: P vs B) flip with the sampling seed (P at seed 0, B at seeds 1-4!).
  Fixed by log-pooling over 4 env draws (now default in predict.py + predict_lite.py):
  removes the seed lottery; bench 78/107/120 (±1 vs single-draw = noise). Affects all
  use_env checkpoints; WBM evals remain single-draw (relative comparisons unaffected).

## 2026-07-04 — E18: discrete-diffusion / flow-matching joint decoder (rejected)

- **Question:** the production head scores each type id with an INDEPENDENT softmax and couples
  types afterward with a hand-built composition beam (PMI + charge neutrality). Would a *learned*
  joint p(y_1..y_T | descriptors) do better? Tested masked (absorbing-state) discrete diffusion
  == discrete flow matching under the masking interpolant (Campbell 2024), so one model stands in
  for both families. (Continuous Gaussian diffusion on a 94-way simplex is strictly worse — not run.)
- **Setup (controlled head swap):** `src/typeid2elem/model_diffusion.py` reuses the IDENTICAL trunk
  (`TypeSetClassifier.encode`, refactored out for this) and replaces only the head with a
  label-embedding + 2-block denoiser + MDLM 1/t-weighted masked-CE objective. Baseline and diffusion
  trained from scratch on the SAME shard subset (40 mptrj_v3 + 12 cod_v3), SAME 14 epochs, no env.
  Eval on a fixed 26,692-type-id val subset (`scripts/exp_diffusion.py`). Inference = confidence-
  ordered iterative unmasking, recording each type's posterior at the step it is unmasked.
- **Result (val top-1/3/5):**
  | method | top-1 | top-3 | top-5 |
  |---|---|---|---|
  | A baseline marginal (production head) | **71.5** | **85.1** | **89.2** |
  | B baseline + PMI/charge beam | 71.6 | 85.1 | 89.2 |
  | C diffusion joint decode | 68.4 | 82.7 | 87.3 |
  | D diffusion marginal (all-masked shot) | 68.4 | 83.2 | 87.8 |
- **Verdict: rejected.** Diffusion is ~3 pts worse at top-1, and C ≈ D (iterative sibling
  conditioning added nothing over its own single-shot marginal; greedy commitment slightly *hurt*
  top-3/5 via error propagation). The learned joint never reached even the free hand-built beam (B).
- **Why worse, not just neutral:** the masked objective leaks sibling labels during training, so the
  descriptor→element map — the thing we actually want — gets weaker gradient pressure than a baseline
  that must predict every token from geometry alone every step. The joint task dilutes the geometry
  encoder. Consistent with E10 (explicit two-pass soft-label feedback also added nothing): the
  attention trunk already captures the sibling structure, and the real ceiling is geometric
  identifiability, not output-model expressiveness.
- **Deployment cost even if it had helped:** iterative decode = T sequential forward passes, breaking
  the single-pass ONNX/CoreML export (E17). Kept `model_diffusion.py` + `exp_diffusion.py` as the
  record; the `TypeSetClassifier.encode` refactor is retained (harmless, reused nowhere else yet).
