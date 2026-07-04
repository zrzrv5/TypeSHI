# TypeSHI: model documentation

What the model is, and — more importantly — why every piece is the way it is. The failure
stories behind these decisions are in `docs/EXPERIMENTS.md` (E0–E17); this document gives the
final design with its rationale.

## 1. Problem statement

Input: `positions (N,3)`, `type_ids (N,)`, optional `cell (3,3)`, optionally several frames.
Output: for each **type id** (not each atom), a ranked distribution over 94 elements
(Z = 1..94), plus a calibrated 90%-coverage candidate set.

Two facts shape everything:

1. **The label lives at the type level.** All atoms of type 3 share one answer. So aggregate
   atoms into per-type statistics *first*, and learn on those — instead of running a per-atom
   network over 10⁴–10⁶ atoms and pooling at the end.
2. **Absolute distances are the signal.** Bond lengths identify chemistry (Si–O ≈ 1.6 Å,
   P–S ≈ 2.0 Å). Therefore: no scale normalization, no scale augmentation, ever. A useful side
   effect: a structure in the wrong length unit looks like *no* chemistry, which is exactly how
   the unit-inference gate works (§7).

Why not an end-to-end atom GNN: deployment. Descriptors are plain histogram math (portable to
Swift/C++/Metal); the network that follows is 1.85 M parameters and exports to ONNX/CoreML
without scatter ops. Model cost is independent of N and of the number of types.

## 2. Descriptors (v3)

Computed per **ordered type pair** (a←b), from one neighbor-list pass (matscipy, cutoff 8 Å):

| channel | shape | why |
|---|---|---|
| smeared partial RDF g_ab(r) | (T,T,64), Δr = 0.125 Å | the workhorse: bond lengths, shells, order/disorder in one object; ideal-gas normalized so g→1 at large r; values clipped ≤1e4 (isolated molecules otherwise overflow fp16) |
| running coordination numbers at r ∈ {2,3,4,6} Å | (T,T,4) | "how many b around a" — the second thing a human expert checks |
| per-atom nearest-neighbor distance: median + 10th percentile over atoms | (T,T,2) | **robust** bond-length estimates. The naive minimum is an extreme-value statistic: at 400 K a single thermal close contact reads P–S as 1.56 Å instead of 2.0 and poisons the prediction (E1). Medians survive temperature. |
| RDF peak position + height (of the smoothed g) | (T,T,2) | dominant-correlation summary, robust to lone contacts |
| stoichiometric fraction x_a | (T,) | composition is chemistry |
| global: log number density, has_cell flag, 1/n_types | (3,) | context. (n_types = the COUNT of type ids in the file. No thermal information exists anywhere in the pipeline — the model sees geometry only.) |

**Environment sets** (the learned channel): per type, 16 sampled atoms × their 16 nearest
neighbors within 6 Å, stored as (distance, partner type id). Pair-marginal RDFs cannot express
per-atom *joint* coordination — "every Li sees 4 O **and** 2 Cl" vs. "Li–O and Li–Cl exist" —
and that joint information separates chemistries the marginals confuse. Worth +3.8 points of
validation top-1 (E12).

Molecules (no cell): reproduced OMol-style vacuum boxes (extent + 10 Å, treated as periodic)
because that is what the training data contains — a convex-hull density path was tried and was
out-of-distribution (E5).

Huge periodic systems: `crop_for_inference` takes a fractional-space crop (~10⁵ atoms); only
atoms whose full 8 Å neighborhood lies inside the crop act as RDF centers. Per-type densities
and fractions must come from the **full** system — crop boundaries cut crystal sublattices
unevenly, and a biased composition rescales entire RDF rows (E17). 10⁶ atoms: 1.1 s / 0.5 GB,
predictions identical to the uncapped computation.

## 3. Network

```
pair channel (a<-b): [log1p(g_ab) | scalars | x_b | is_self]  --phi MLP-->  h_ab   (128)
type token a:  [mean_b h_ab | max_b h_ab | x_a | global]      --proj MLP--> tok_a  (256)
   + env branch: RBF16(d) x partner-type-token --DeepSets over nbrs, atoms--> +tok_a
2 transformer encoder blocks over the <=8 type tokens (padding masked)
linear head -> 94 logits per type
```

- DeepSets pooling over pair channels handles a *variable number of types* by construction.
- The transformer lets types negotiate: mutual exclusivity, composition context. This is
  measurable — WBM accuracy rises monotonically with type count (21% top-1 at T=2 → 53% at
  T=5). An explicit second pass feeding soft element identities back in ("once type 2 is easy,
  type 1 gets easier") added nothing — attention already does it (E10).
- 1,846,558 parameters. Loss: class-weighted cross-entropy (1/√freq, clipped at 10, absent
  classes zeroed). OneCycle LR, 30 epochs, bf16, ~1.5 h on one RTX 4090 for the full mix.

## 4. Training data and split

≈1.52 M training records from: MPTrj (DFT relaxation trajectories, 544 k records), COSMOS DBS
(401 k), **COD — experimental crystal structures (710 k records from 535 k CIFs; the single
biggest accuracy lever in the whole project, E16)**, OpenLAM/DPA-2 domains incl. a targeted
HfO₂ set (22 k). Records are precomputed descriptor shards (npz), not structures.

Train/val split is by **element set** (md5 of the sorted symbols, 10% to val): every Li–P–S
structure lands on one side. Stronger than per-structure splits — it also deduplicates across
datasets and makes "unseen chemistry" measurable. WBM shows no seen/unseen gap, i.e. no
composition memorization.

**Augmentations bridge DFT training to dirty MD reality** (all at preprocessing time):

| augmentation | why real files need it |
|---|---|
| type splitting (random, uneven Dirichlet sizes) | force fields split one element across type ids: charge states, core–shell models, site labels. Verified transfer: a core–shell polarizable file scores 4/6 top-1 with both O types correct. |
| subsample a type to 5–50% | dopants and dilute species |
| thermal jitter σ ≤ 0.35 Å | training structures are near-equilibrium; user files are at 400–900 K |
| ±3% strain | NPT fluctuations |

What was tried and did **not** help: hot-MD fine-tuning (MACE trajectories with free true
labels — the robust features + jitter had already closed that gap), RAFT/verifier-label
distillation (every corpus we may train on already has true labels; verifier labels are
strictly noisier), post-hoc debias, TTA jitter, same-recipe deep ensembles.

## 5. Ensemble, fusion, decoding

Production = 6 checkpoints with **heterogeneous recipes** (loss softness, epochs, augmentation
mix, data mix; homogeneous seed ensembles do not help). Fusion is confidence-weighted
arithmetic prob-mean (`fuse="conf"`): plain log-averaging is a consensus operation that lets
five Hf-ignorant members outvote the one member that knows Hf at 99.6% (it landed at rank 3;
conf-fusion restores rank 2 at 41%). Since COD, a single model matches the ensemble on the
real-file bench — the ensemble mostly buys WBM stability now.

Composition decode: a beam search over top-10 per type re-scored with element co-occurrence
PMI (mined from 60 k training element sets) + fraction-weighted charge neutrality, then
re-marginalized. Couples the per-type posteriors ("if type 3 is O at 99%, its partners are
probably cations").

Environment sets are 16-atom samples — a deliberately high-variance feature. At inference we
log-pool **4 env draws**; without this, near-tie calls flip with the sampling seed (E17).

## 6. Calibration

The pipeline is heavily overconfident: **softmax** temperature scaling needs a factor of 3.6
on 33 k validation type-ids (a statistics term — nothing to do with thermal temperature). Shipped
answer: RAPS conformal sets at 90% coverage (empirical 91.7%, median size 7). For
geometrically degenerate families (lanthanide contraction: Gd/Tb/Dy differ by ~0.01 Å) the
truthful output *is* "one of these" — exact-match there is physically impossible from
geometry, and the sets say so instead of bluffing.

## 7. Unit inference & the not-atoms gate

Nearest-neighbor distances of atomistic matter live in a hard window (~0.6–3.6 Å). For each
candidate unit (Å, Bohr, nm, µm, cm, m — the LAMMPS unit styles) rescale and score the
fraction of NN distances inside the window; no candidate fits → **"NOT ATOMISTIC"** (this is
what happens to DEM/granular files). Å-vs-Bohr can be genuinely ambiguous (×1.89); survivors
are tie-broken by model confidence under each rescaling, with an ångström prior — a
distorted-but-valid structure (e.g. an RMC fit) can look *more* familiar under a wrong
rescaling, so non-Å must win decisively (+0.30 confidence margin).

## 8. Known limits

- **Identifiability**: radius twins (Si/P, Ni/Co/Mn, Cl/S) and the f-block cannot be separated
  by geometry alone. Expect them inside the conformal set, not at rank 1.
- Zero-atom types (declared but unpopulated) are unidentifiable by construction and reported
  as such.
- RMC-fitted structures have bonds that genuinely favor the wrong elements; both the model and
  an MLIP force-consistency verifier are misled (E9).
- The optional MACE verifier (`--verify` style best-of-N re-ranking, `src/typeid2elem/verifier.py`)
  wins 86% of radius-twin swaps but needs a large MLIP at inference — off by default.
