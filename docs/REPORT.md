# TypeSHI — project report

*Type-id → Species Hypothesis Inference: your MD file forgot its elements; TypeSHI guesses
them back from geometry alone.*

MD files routinely store atoms as anonymous integer types — `1, 2, 3, 4` — with no element
symbols and often no usable metadata. TypeSHI reads nothing but **geometry** (positions, type
ids, an optional cell) and returns ranked element candidates for every type id, with
calibrated uncertainty, in under a second, from a 2 MB model.

**Headline numbers:** 60% top-1 / 82% top-3 / 93% top-5 over 131 real type-ids · a 1.9 MB int8
model matching a 6-model ensemble · 0.8 s and 166 MB RAM per file on CPU · 1.1 s even at 10⁶
atoms.

## 1. Why this is hard, and why it matters

Anyone who has inherited a LAMMPS `.data` file, a bare dump, or a decade-old trajectory knows
the ritual: guess the elements from folder names, dig for the input script, hope the `Masses`
section exists. And force-field conventions split one element across many type ids
(core–shell models, charge states, site labels).

No prior model does this task. The nearest published work masks an element and predicts its
period/group from a *known* crystal graph; nothing infers species for *anonymous* types in
*dirty, finite-temperature* simulation data. The information genuinely is in the geometry —
bond lengths, coordination, stoichiometry are exactly what a human expert reads — but the
expert doesn't scale.

## 2. Method: descriptors first, tiny network second

We deliberately rejected end-to-end atomistic GNNs. The label lives at the *type* level, so we
aggregate atoms into per-type-pair statistics before any learning — model cost becomes
independent of atom count, and the network small enough to ship anywhere.

```
parse file → unit gate (infer Å/Bohr/nm…; refuse non-atomistic data)
           → per-type-pair descriptors → set network over types
           → composition decode → ranked elements + 90% coverage sets
```

- **Descriptors** (per ordered type pair): smeared partial RDFs g_ab(r) (64 bins to 8 Å,
  ideal-gas normalized) + robust scalars — per-atom nearest-neighbor distance medians and 10th
  percentiles (extreme values collapse at 400–600 K; medians don't), running coordination
  numbers, peak position/height, stoichiometric fractions. Absolute ångström distances are
  *the* signal — no scale normalization anywhere, which is also what makes wrong-unit
  detection possible.
- **Environment sets** (learned channel): per type, 16 sampled atoms × 16 nearest neighbors as
  (distance, partner type), DeepSets-encoded. Restores the per-atom *joint* coordination
  ("every Li sees 4 O **and** 2 Cl") that pair marginals discard.
- **Network**: shared MLP per pair channel → permutation-invariant pooling → one token per
  type → 2 transformer blocks (types negotiate mutual exclusivity and composition) → 94-way
  head. **1.85 M parameters**; any number of types by construction. Trained on ~1.5 M records
  (MPTrj, COSMOS, OpenLAM/DPA-2, COD), split by *element set*, with augmentations that mirror
  real-file dirt: type splitting, thermal jitter, strain, dilution.
- **Honest uncertainty**: the raw pipeline is overconfident — softmax temperature scaling
  needs a factor of ≈3.6(!). We ship RAPS conformal sets (90% coverage, median 7 candidates)
  instead of pretending.

## 3. What we tried, in order

| idea | verdict | what happened |
|---|---|---|
| closest-approach distance as bond feature | rejected | extreme-value statistic; thermal contacts read P–S as 1.56 Å instead of 2.0. Medians fixed it: 2/19 → 12/19 top-1 |
| class weighting, post-hoc debias, TTA jitter | mixed | weighting kept; the rest did nothing |
| same-recipe deep ensembles | rejected | seeds agree too much; heterogeneous *recipes* ensemble well |
| composition-prior decoding (PMI + charge neutrality beam) | kept | a confident O drags partners toward plausible oxide chemistry |
| two-pass soft refinement | neutral | the attention blocks already do it |
| MLIP force-consistency verifier (best-of-N) | shelved | 86% correct on radius-twin swaps (apparently novel), but inverts on RMC-fitted structures; optional flag |
| RL / RAFT distillation | rejected | every trainable corpus already has true labels; verifier labels are strictly noisier |
| hot-MD fine-tuning (MACE trajectories, free labels) | neutral | robust features + jitter had already closed the thermal gap; clean negative result |
| learned environment sets | kept | +3.8 val top-1; one env model matched the then-ensemble on WBM |
| targeted domain data (DPA-2 HfO₂) | kept | a known failure went from rank 46 to rank 1 at 99.6% |
| **experimental crystals at scale (COD, 535k CIFs)** | **kept** | **the single biggest lever; one small model now matches the entire ensemble** |
| confidence-weighted ensemble fusion | kept | log-averaging let 5 ignorant models outvote 1 informed specialist (99.6% → rank 3); conf-fusion repairs most of it free |
| unit inference + not-atoms gate | kept | NN distances live in a hard physical window; scan candidate units, or refuse DEM data. Trap: distorted-but-valid structures look "more familiar" under wrong rescaling — fixed with an Å prior |
| deployment: ONNX/CoreML, int8, interior-crop, draw pooling | kept | see §5; includes fixing a nondeterminism where 16-atom env samples flipped near-tie calls |

## 4. Results

**Real-file bench — 22 systems, 131 type-ids** (electrolytes at 400–600 K, doped cathodes with
O-vacancies, amorphous oxides, molecules, alloys, glasses, a core–shell model, RMC fits;
collected from real working directories, not curated for friendliness):

| configuration | top-1 | top-3 | top-5 |
|---|---|---|---|
| TypeSHI, single 1.9 MB int8 model | 81/131 | 106/131 | 120/131 |
| TypeSHI, 6-model ensemble + decode | 78/131 | 107/131 | 122/131 |

The single small model and the full ensemble are statistically indistinguishable — the
ensemble era ended when the training data got diverse enough. Failures concentrate exactly
where geometry is degenerate: Ni↔Co↔Mn (~0.01 Å radius differences), Si↔P, heavily distorted
structures.

**Large-scale — WBM, 12,158 type-ids:** 27.5% / 45.0% / 53.5%, with **zero gap between seen
and unseen element sets** (generalization, not memorization). Accuracy *rises* with type count
(21% at T=2 → 53% at T=5): more types, more mutual constraints. Relaxed and unrelaxed
structures score identically. The f-block is geometrically unidentifiable (lanthanide
contraction) — a physics fact the conformal sets state honestly.

## 5. Deployment

| path | deps | model | peak RAM | wall/file |
|---|---|---|---|---|
| full (torch, 6-model) | torch + lightning | 6 × 7.4 MB | 946 MB | 2.5 s |
| lite (onnxruntime) | numpy · scipy · matscipy · onnxruntime | **1.9 MB int8** | **166 MB** | **0.8 s** |

Million-atom files stay cheap through an interior-crop trick (only atoms with a complete 8 Å
neighborhood act as RDF centers; densities from the full system — crop boundaries cut
sublattices unevenly): 952k atoms go from 10.6 s / 3.8 GB to **1.1 s / 0.5 GB** with identical
predictions. CoreML packages (3.8 MB fp16) export cleanly; descriptor math is plain
histogramming, portable to Swift/C++/Metal.

## 6. Take-home points

1. **Put the physics in the features, not the network.** Per-type-pair descriptors made model
   size independent of atom count, trivially exportable, and unit-aware. Absolute bond lengths
   are the signal — never normalize scale, and wrong-unit detection comes free.
2. **Robust statistics beat sufficient statistics on dirty data.** The largest early gain was
   replacing a minimum with a median. Thermal noise destroys extreme values first.
3. **Data diversity beat every architectural idea.** The env encoder earned +3.8 points;
   535k experimental crystals let one small model match a 6-model ensemble. Given the choice
   between a cleverer model and a broader corpus, take the corpus.
4. **Train for the file, not the physics.** Real files split elements across type ids, dilute
   them, and lie in their metadata. Type-splitting augmentation transferred straight to a
   core–shell polarizable model and a file whose mass metadata was simply wrong.
5. **Know your identifiability limits and price them in.** Radius twins and the f-block cannot
   be separated by geometry. Calibrated top-K sets (softmax temperature 3.6 of overconfidence!)
   turn an impossible exact-match demand into a useful shortlist.
6. **When labels are free, RL isn't the answer.** Best-of-N with a physics verifier at
   inference, maybe; distilling verifier labels over corpora with ground truth, no.
7. **Ensembles dilute specialists.** Log-prob averaging is a consensus operation; the one
   model that knows Hf gets outvoted. Confidence-weighted fusion recovers most of it; learned
   gating is the open next step.
8. **Chase nondeterminism before shipping.** A 16-atom sampled feature quietly flipped
   near-tie predictions between runs. Pooling four draws costs milliseconds and makes the tool
   answer the same question the same way twice.

---

*Stack: Python 3.12 · PyTorch + Lightning (training) · ONNX Runtime / CoreML (inference) ·
matscipy neighbor lists · one RTX 4090. Data: MPTrj, COSMOS, OpenLAM & DPA-2 domains
(AIS-Square), Crystallography Open Database, Wang–Botti–Marques (evaluation). Full lab log:
`docs/EXPERIMENTS.md` E0–E17.*
