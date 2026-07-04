# TypeSHI : guess the chemical elements from geometry alone

TypeSHI (**T**ype-id → **S**pecies **H**ypothesis **I**nference) is a for-fun experiment I asked Fable to conduct that may end up used in my other [project](https://github.com/zrzrv5/NAIIVE).
It turns a structure into per-type-pair descriptors (partial RDFs + robust bond/coordination
statistics + sampled neighbor environments) and feeds them to a tiny set-transformer (1.85M
params) that scores 94 elements per type id. 

If you landed here looking for something else:

- It's **not masked element prediction** (recovering a hidden atom in an otherwise *labeled* graph) — check [Mole-BERT](https://arxiv.org/abs/2211.03563) ([code](https://github.com/junxia97/Mole-BERT)).
- It's **not force-field atom-typing** (assigning sub types to atoms of *known* elements) — check [atom typing via graph representation learning(JCP 2022)](https://pubs.aip.org/aip/jcp/article-abstract/156/20/204108/2841271/Atom-typing-using-graph-representation-learning?redirectedFrom=fulltext).
- It's **not a materials/structure predictor from experimental measurements** (phase or composition from XRD/PDF spectra) — check [XRD-AutoAnalyzer](https://github.com/njszym/XRD-AutoAnalyzer) and [DeepStruc](https://github.com/EmilSkaaning/DeepStruc).

TypeSHI's input is a bare simulation snapshot whose per-atom *identities themselves* are the
unknowns.

> **TLDR** — current best: a single small model trained on ~1.5M records (MLFF dataset + crystals from COD, with augmentation), plus a composition prior and conformal top-K sets at inference. 
> On 131 real type-ids from files lying around my machines: 60% top-1, 82% top-3, 93% top-5 — from a 1.9 MB int8 model in <1 s on CPU.

> **Problem & Motivation**: files from my work often carry no element names — sometimes the format can't store them, sometimes (mostly) past-me was lazy — and every visualization session starts with re-assigning types by hand.
> Parsing metadata (e.g. a LAMMPS `Masses` section) should be the correct way to this,
>  but I just wondering whether the geometry alone is enough.

> **Important**: the problem is ill-defined, so the output is ranked *hypotheses* with calibrated coverage sets, never a verdict. Also, much of the work was done by Fable, so I would suggest reading the [Report](docs/REPORT.md) — every claim there traces back to the lab log in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) — and taking it with a grain of salt.

[TOC]

## Structure of this project

- `src/typeid2elem/` — library: `io` (LAMMPS/ASE parsing), `descriptors`, `model`, `inference`,
  `decode` (composition prior), `units` (unit inference + not-atoms gate), `api`
- `scripts/` — `preprocess_*` (one per data source), `train`, `evaluate_*`, `predict`,
  `predict_lite` (no-torch ONNX path), `calibrate`, `export_model`, `mine_costats`
- `weights/` — committed 2 MB runtime bundle: the single int8 deploy model +
  decode/conformal stats, so a fresh clone runs `predict_lite.py` with no download
- `metal/` — (future) Metal + Swift code for on-device inference (`InaccurateRDFCalculator`
  descriptor kernel → CoreML)
- `docs/`
  - `REPORT.md` — the human-readable project report: method, what we tried, results, take-homes
  - `MODEL.md` — what the model is and *why* every piece is the way it is
  - `REPRODUCE.md` — data acquisition → preprocessing → training → evaluation, step by step
  - `EXPERIMENTS.md` — append-only lab log E0–E17: everything tried, kept, and rejected
  - `EXPORT.md` — ONNX/CoreML export details; `PLAN.md` — decision log; `DATA.md` — dataset notes

Training data (~120 GB raw) is **not** in the repo; `docs/REPRODUCE.md` has download commands
for every source. The 2 MB int8 deploy model ships in `weights/` (so the lite path runs on a
bare clone); the full set — torch checkpoints, per-member ONNX, and the CoreML packages for the
Metal export — is a GitHub Release tarball built by `scripts/package_release.py`.

## Inference

### So, does it work?

Well — on test cases from files I had around, it works alright. Keep in mind the problem
itself is ill-posed (non-atomistic files like DEM, core–shell models where one atom is two
particles, and you could just randomly shuffle labels — no model can undo that), so if it
worked *too* well that would just be wizardry and magic. Some of what I tested — ✓ means the
true element is top-1 (with its probability); misses show what was predicted instead and where
the truth ranked. Temperatures below describe how the files were *made*; the model never sees
temperature, masses, or any metadata — geometry only.

| system | atoms | true types | prediction (per type) | comments |
|---|---|---|---|---|
| my LiSiPS solid electrolyte (400 K MD) | 2,429 | Li Si P S | Li ✓ 94% · **Si → P 54%** (Si r8, in 90% set) · P ✓ 90% · S ✓ 99.5% | Si/P is a radius twin; the coverage set is the honest answer |
| Na₂B₁₂H₁₂ (600 K MD) | 5,200 | Na C B H | Na ✓ 79% · **C: zero atoms declared** → reported unidentifiable · B ✓ 100% · H ✓ 96% | geometry can't name a type with no atoms in it |
| Li₆PS₅Cl DFT-MD (r²SCAN) | 416 | Li P S Cl | Li ✓ 95% · P ✓ 95% · S ✓ 93% · **Cl → S 95%** (Cl r3) | argyrodite Cl sits on S-like sites — anion twin |
| NiTi shape-memory alloy | 6,750 | Ni Ti | Ti ✓ 23% · **Ni → Ti 23%** (Ni r2, 14%) | metallic twins: both types share one shortlist, attention can't break the tie |
| CHO molecule (no cell) | 63 | H C O | H ✓ 100% · C ✓ 96% · **O → N 51%** (O r2, 48%) | vacuum-box path; O/N near-tie |
| LiFePO₄ **core–shell** model | 1,296 | Fe Li O P (+Fe,O shells) | Li ✓ 69% · O ✓ 97% · O(shell) ✓ 95% · P ✓ 100% · **Fe core+shell → Li 26%** (Fe r6) | each Fe/O is *two* particles at ~0 Å; both O types still resolve |
| amorphous Hf-Ba oxide | 96 | Hf Ba O | O ✓ 97% · **Hf → Sb 78%** (Hf r48) · **Ba → K 72%** (Ba r6) | amorphous heavy-metal oxides are the standing hard case |
| RMC-fitted LACO structure | 38 | Li Al Cl O | O ✓ 32% · **Cl → S 50%** (Cl r2, 48%) · **Li → Bi 71%** (Li r5) · **Al → Fe 54%** (Al r65) | reverse-Monte-Carlo bonds *genuinely* favor the wrong elements — even an MLIP verifier agrees with the wrong answer here |
| DEM granular packing saved as LAMMPS data | 512 | (not atoms) | **refused**: "NOT ATOMISTIC" | no length unit puts its nearest-neighbor distances in the chemistry window |

Full benchmark (22 systems / 131 type-ids): 60% top-1, 82% top-3, 93% top-5. On WBM
(12,158 type-ids of element-substituted crystal prototypes): 27.5 / 45.0 / 53.5 — with zero
gap between seen and unseen element sets, i.e. it generalizes composition rather than
memorizing it.

### Try it yourself

There's a simple inference script (with a LAMMPS data-file reader written by Fable; anything
ASE reads works too via `--format ase`):

```bash
uv run python scripts/predict.py <file> [--conformal] [--units bohr] [--top-k 5]
# minimal footprint (no torch; 1.9 MB int8 model from weights/, ~170 MB RAM, <1 s):
uv run python scripts/predict_lite.py <file> [--conformal]
```

`predict_lite.py` runs on a bare clone (model in `weights/`). `predict.py` runs the 6-model
torch ensemble and needs the checkpoints from the Release tarball unpacked into `runs/`.

```
$ uv run python scripts/predict.py 400K.run.data --conformal

Inferred length unit: angstrom (NN in-window 100%)

System: 2429 atoms, 4 types, 1 frame(s), 6-model ensemble, composition decode
  type 1: Li 94.0%, Ca 1.8%, Zn 1.6%, Sr 1.2%, Ag 0.6%
      90%-coverage set (8): {Li, Ca, Zn, Sr, Ag, Na, Cd, Mg}
  type 2: P 54.2%, Sc 24.0%, N 5.3%, Fe 3.1%, Mn 2.4%     <- actually Si: a radius twin;
      90%-coverage set (9): {P, Sc, N, Fe, Mn, Zr, Co, Si, O}   the honest answer is the set
  ...
```

Or pass it the raw numpy arrays directly — no files involved:

```python
from typeid2elem.api import predict_structure

result = predict_structure(positions, type_ids, cell=cell)   # any length unit
for t in result["types"]:
    print(t["label"], t["candidates"][:3], t["set90"])
```

It infers the length unit (Å / Bohr / nm / µm / cm / m), refuses non-atomistic input, and
time-averages multiple frames of the same system if you pass several. I will maybe include
this in my other project later this year...

## For Development

Ask your favorite coding agent to read the `AGENTS.md` file.

### Install and setup

```bash
uv sync        # Python 3.12, full environment (training + inference)
```

Two environment gotchas are pinned in `pyproject.toml` on purpose: torch uses **cu128 wheels**
(default cu13 wheels break on driver ≤ CUDA 12.9), and neighbor lists use **matscipy** (ASE's
is ~80× slower). Optional wandb logging reads a repo-root key file — see `docs/REPRODUCE.md` §0.
Inference is CPU-only; training wants a GPU (everything here was trained on one RTX 4090).

### Download data

I use a lot of data for training — ~120 GB raw across MPTrj, COSMOS, OpenLAM/DPA-2 domains,
and the Crystallography Open Database — none of it is included in the repo. `docs/REPRODUCE.md`
has the exact download commands per source, the preprocessing commands with expected record
counts, all six training recipes, and the format traps we hit (DeePMD mixed-type layout,
CIF dummy species) so you don't have to rediscover them.

## Roadmap

- **On-device (Apple/Metal)**: descriptor computation as a Metal kernel
  (`InaccurateRDFCalculator`: subsampled/approximate partial-RDF transform; the interior-crop
  trick maps directly to GPU) feeding the CoreML model (`runs/export/*.mlpackage`, 3.8 MB fp16).
- Learned gating over ensemble members (confidence-weighted fusion is a partial fix for
  specialist dilution).
- More DPA-2/AIS-Square domains (ZrO₂, W, metallic glasses, electrolytes).

## License

TBD before public release.
