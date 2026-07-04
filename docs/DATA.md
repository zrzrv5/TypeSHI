# Data inventory & stats

Factual stats on the two training datasets, gathered to plan preprocessing/training for the
type-id → element model (see `docs/PLAN.md`). All numbers below come from full (non-sampled)
scans of both files, produced with the scripts in `scripts/inspect/` (paths given per section).
Run everything as `uv run python scripts/inspect/<script>.py` from the repo root.

---

## Dataset 1: `Data/COSMOS/DBS/dbs_total.extxyz` (~780 MB, extended XYZ)

### Method

- Verified the extxyz frame layout with `ase.io.iread` on the first 5 frames (line 1 = atom
  count `N`, line 2 = comment/info line of `key=value` pairs, lines 3..N+2 = `<element> x y z fx
  fy fz magmom`).
- Wrote a manual line-parser, `scripts/inspect/scan_extxyz.py`, that does **one full pass** over
  the file without using ASE (ASE's per-frame object construction is unnecessary overhead for pure
  counting/histogramming). It reads only the header line, regex-scans the comment line for
  `key=` tokens, and reads the first whitespace token of each atom line as the element symbol.
- This is a **full scan**, not a sample — it completed in ~4 seconds (single pass, no per-frame
  Python object construction), so no sampling was needed.
- Raw output: `scripts/inspect/extxyz_scan_result.json`.

### Frame count

**134,327 frames total.**

### Comment-line (info) fields

Every single frame (134,327/134,327) has exactly these 8 keys in its comment line, with no
missing values observed anywhere in the file:

| key | meaning |
|---|---|
| `Lattice` | 3×3 cell matrix (always present — see below) |
| `Properties` | column spec for atom lines; always `species:S:1:pos:R:3:forces:R:3:magmoms:R:1` |
| `pbc` | periodic boundary flags, always present, always `"T T T"` in the frames sampled |
| `db_label` | source-dataset tag + index, e.g. `oc20_random_10333` (see grouping below) |
| `energy`, `free_energy` | total energy (float) |
| `stress` | 3×3 stress tensor |
| `magmom` | scalar total magnetic moment (per-atom magmoms are the 4th atom-line column) |

There is **no `config_type` field** — grouping/provenance is carried entirely by `db_label`.

- `Lattice`/cell: present in **134,327/134,327** frames (100%).
- `pbc`: present in **134,327/134,327** frames (100%), always `T T T` — fully periodic, no
  molecules-in-vacuum / mixed-pbc frames observed.

### `db_label` and grouping (important for train/val split)

`db_label` decomposes as `<source>_<random|spare|elem>_<index>`. There are **18 distinct
source/category prefixes**, and the file is laid out as 18 large **contiguous blocks**, one per
prefix (verified: db_label-prefix run-length analysis found exactly 18 runs across the whole
file, i.e. the file is fully sorted/grouped by source, not interleaved):

| Source prefix | Frames | % of file |
|---|---|---|
| omol25_random | 72,694 | 54.12% |
| qcml_random | 24,482 | 18.23% |
| oc20_random | 17,144 | 12.76% |
| omol25_spare | 6,991 | 5.20% |
| oc22_random | 3,898 | 2.90% |
| qcml_spare | 2,449 | 1.82% |
| oc20_spare | 1,735 | 1.29% |
| odac23_random | 1,729 | 1.29% |
| omol25_elem | 634 | 0.47% |
| qcml_elem | 594 | 0.44% |
| odac23_elem | 462 | 0.34% |
| oc22_spare | 387 | 0.29% |
| oc22_elem | 296 | 0.22% |
| oc20_elem | 287 | 0.21% |
| odac23_spare | 173 | 0.13% |
| matpes_random | 145 | 0.11% |
| matpes_elem | 132 | 0.10% |
| matpes_spare | 95 | 0.07% |

These prefixes correspond to well-known foundation-model source datasets: **MatPES** (MatPES
elemental/random DFT), **OC20**/**OC22** (Open Catalyst surfaces), **ODAC23** (Open DAC MOFs),
**OMol25** (Open Molecules), **QCML** (quantum-chemistry ML). `omol25_random` alone is >54% of
all frames — a strong source imbalance to be aware of independent of the element imbalance.

**Full `db_label` values are unique** — every one of the 134,327 labels appears exactly once
(checked directly), so there are **no repeated/duplicate-frame trajectories keyed by db_label**.
Looking at **composition** (the multiset of elements) instead: consecutive-frame runs of
identical composition have median run length 1 and mean 1.09 (max 10) — i.e. the file is
essentially a sequence of independent single-point structures, not long MD trajectories. This
matches the source datasets (mostly single relaxation/random snapshots, not trajectories).

**Implication for splitting:** because the file is grouped into contiguous per-source blocks
(not shuffled), a naive sequential train/tail split (e.g. first 80% train / last 20% val) would
put entire sources only in one split (e.g. val could be 100% `matpes_*` with zero `omol25_*`).
There is no meaningful "trajectory leakage" risk (frames are not repeated / are not dense-in-time
snapshots of the same structure), so a plain **stratified random split by `db_label` source
prefix** (shuffle within each of the 18 blocks, then split with a fixed fraction from each) is
sufficient and avoids both leakage and source-distribution mismatch between train/val.

### n_atoms distribution

n = 134,327 frames. Mean **43.97**, median **22**, min **1**, max **615**.

| percentile | n_atoms |
|---|---|
| 5th | 8 |
| 25th | 14 |
| 50th (median) | 22 |
| 75th | 62 |
| 90th | 101 |
| 95th | 126 |
| 99th | 207 |

Distribution is heavily right-skewed: a mode around 12-16 atoms, a long tail out to 615 atoms
(full histogram in `scripts/inspect/extxyz_scan_result.json` under `n_atoms_hist`).

### Distinct elements per frame

| distinct elements | frames | % |
|---|---|---|
| 1 | 69 | 0.05% |
| 2 | 3,585 | 2.67% |
| 3 | 25,771 | 19.19% |
| 4 | 54,189 | 40.34% |
| 5+ | 50,713 | 37.76% |

**97.3% of frames have ≥3 distinct elements** — this dataset is dominated by multi-element
structures, which is exactly the regime our type→element model cares about most.

### Overall element histogram (atom-instance counts, not frame counts)

89 distinct elements across 5,906,286 total atom instances. Top 30 (91.3% of all atoms) below;
remaining 59 elements are the tail.

| Rank | Element | Count | % of atoms |
|---|---|---|---|
| 1 | H | 2,099,961 | 35.55% |
| 2 | C | 1,340,662 | 22.70% |
| 3 | O | 613,752 | 10.39% |
| 4 | N | 278,909 | 4.72% |
| 5 | S | 127,530 | 2.16% |
| 6 | P | 58,970 | 1.00% |
| 7 | Si | 58,023 | 0.98% |
| 8 | Al | 54,975 | 0.93% |
| 9 | Se | 51,074 | 0.86% |
| 10 | Cl | 50,792 | 0.86% |
| 11 | Ti | 48,008 | 0.81% |
| 12 | Ga | 43,159 | 0.73% |
| 13 | Te | 38,225 | 0.65% |
| 14 | Pd | 37,417 | 0.63% |
| 15 | Ca | 34,243 | 0.58% |
| 16 | Hf | 34,215 | 0.58% |
| 17 | As | 33,827 | 0.57% |
| 18 | Zn | 33,081 | 0.56% |
| 19 | Cu | 32,551 | 0.55% |
| 20 | Pt | 32,252 | 0.55% |
| 21 | F | 31,448 | 0.53% |
| 22 | Sn | 30,993 | 0.52% |
| 23 | In | 30,602 | 0.52% |
| 24 | Ge | 29,725 | 0.50% |
| 25 | Ag | 29,203 | 0.49% |
| 26 | Na | 28,674 | 0.49% |
| 27 | Zr | 28,308 | 0.48% |
| 28 | Rh | 27,634 | 0.47% |
| 29 | V | 27,591 | 0.47% |
| 30 | Sb | 27,014 | 0.46% |
| tail | 59 more elements | 513,468 | 8.69% |

Rarest tail elements (from `element_hist` in the raw JSON): Pu (6), Ac (8), Pa (8), Ar (27), Ne
(30), He (39), Pm (41), Kr (52) — actinides/noble gases are vanishingly rare, as expected for
organic/inorganic materials-chemistry data. H/C/O/N alone are 73.4% of all atoms (organic-heavy —
consistent with OMol25/QCML dominance).

---

## Dataset 2: `Data/MPtrj/MPtrj_2022.9_full.json` (12 GB, nested JSON)

### Method

- Structure verified first with a manual `ijson` probe: top level is `{mp_id: {frame_id: {...}}}`;
  each frame dict has a `structure` key holding a pymatgen `Structure`-as-dict
  (`@module`, `@class`, `charge`, `lattice`, `sites`) plus energies/forces/etc. Each `sites[i]` is
  `{'species': [{'element': ..., 'occu': ...}], 'abc': [...], 'xyz': [...], 'label': ..., 'properties': {}}`.
- **Fastest access pattern**: `ijson.kvitems(f, '', use_float=True)` over the file opened in
  binary mode. This streams top-level `(mp_id, material_dict)` pairs — each `material_dict` (all
  frames for one material) is fully materialized in memory, but that's small (a handful of frames,
  each ≤ a few hundred atoms), so peak memory stays bounded while the top-level dict is never
  fully built. `ijson` auto-selected its C backend (`yajl2_c`), which is available and fast.
  `use_float=True` is important — without it every numeric leaf (all `force`/`stress`/`magmom`
  entries) is built as a `Decimal`, which is much slower to construct than `float` at this scale.
- This is a **full scan** (not a sample): `scripts/inspect/scan_mptrj.py` processed the entire
  12 GB file in **242 seconds** (~4 minutes), well inside the 15-minute budget.
- Raw output: `scripts/inspect/mptrj_scan_result.json`.

### Counts

**145,923 materials, 1,580,395 frames** (avg **10.83 frames/material**). These numbers match the
published MPtrj (2022.9) statistics exactly, confirming the scan is correct and complete.

### Frames-per-material distribution

| frames per material | materials | % |
|---|---|---|
| 1 | 18,512 | 12.69% |
| 2 | 6,348 | 4.35% |
| 3 | 4,031 | 2.76% |
| 4 | 7,720 | 5.29% |
| 5 | 7,829 | 5.36% |
| 6 | 9,613 | 6.59% |
| 7 | 7,025 | 4.81% |
| 8 | 7,457 | 5.11% |
| 9 | 6,580 | 4.51% |
| 10 | 6,455 | 4.42% |
| 11–19 | 46,291 | 31.72% |
| ≥20 | 19,062 | 13.06% |

Most materials contribute several frames (relaxation-trajectory steps toward the relaxed
structure), not just one — **only 12.7% of materials are single-frame**. This is the key reason
splits must be done **by `mp_id`, not by frame**: frames within one material are highly correlated
(same composition, often near-identical geometry at different relaxation steps), so a random
frame-level split would leak near-duplicate structures across train/val. `docs/PLAN.md` already
specifies "split by material/composition, not by frame" — this scan confirms why that matters
quantitatively (87% of materials have ≥2 frames to leak).

### n_atoms distribution

n = 1,580,395 frames. Mean **31.19**, median **22**, min **1**, max **444**.

| percentile | n_atoms |
|---|---|
| 5th | 4 |
| 25th | 12 |
| 50th (median) | 22 |
| 75th | 40 |
| 90th | 68 |
| 95th | 88 |
| 99th | 144 |

Smaller cell sizes than the COSMOS DBS file on average (median 22 vs 22 — same median, but a much
shorter tail: 99th pctile 144 vs 207, max 444 vs 615) — consistent with MPtrj being built from
Materials Project DFT relaxations (typically small unit cells / primitive-ish cells) rather than
large slab/molecule snapshots.

### Distinct elements per frame

| distinct elements | frames | % |
|---|---|---|
| 1 | 10,031 | 0.63% |
| 2 | 220,220 | 13.93% |
| 3 | 689,617 | 43.64% |
| 4 | 469,950 | 29.74% |
| 5+ | 190,577 | 12.06% |

**85.4% of frames have ≥3 distinct elements** — somewhat lower than COSMOS DBS (97.3%), but still
the large majority; ternary compositions (3 elements) are the single largest bucket (43.6%).

### Overall element histogram

89 distinct elements across 49,295,660 total atom instances (structures are typically inorganic
Materials Project compounds — oxide-heavy, unlike the organic-heavy COSMOS DBS set). Top 30
(85.4% of all atoms) below; remaining 59 elements are the tail.

| Rank | Element | Count | % of atoms |
|---|---|---|---|
| 1 | O | 18,231,921 | 36.98% |
| 2 | H | 3,022,802 | 6.13% |
| 3 | F | 1,853,060 | 3.76% |
| 4 | S | 1,671,273 | 3.39% |
| 5 | P | 1,286,729 | 2.61% |
| 6 | N | 1,270,141 | 2.58% |
| 7 | Li | 1,256,745 | 2.55% |
| 8 | Mg | 1,236,354 | 2.51% |
| 9 | C | 1,062,189 | 2.15% |
| 10 | Si | 1,044,992 | 2.12% |
| 11 | Cl | 941,874 | 1.91% |
| 12 | Fe | 747,611 | 1.52% |
| 13 | Se | 735,063 | 1.49% |
| 14 | B | 673,705 | 1.37% |
| 15 | Mn | 666,675 | 1.35% |
| 16 | Al | 607,254 | 1.23% |
| 17 | Na | 565,195 | 1.15% |
| 18 | V | 483,914 | 0.98% |
| 19 | K | 465,558 | 0.94% |
| 20 | Co | 462,520 | 0.94% |
| 21 | Cu | 443,103 | 0.90% |
| 22 | Ca | 432,675 | 0.88% |
| 23 | Ti | 379,620 | 0.77% |
| 24 | Br | 378,868 | 0.77% |
| 25 | I | 378,474 | 0.77% |
| 26 | Ba | 377,213 | 0.77% |
| 27 | Zn | 369,598 | 0.75% |
| 28 | Te | 356,260 | 0.72% |
| 29 | Ni | 349,059 | 0.71% |
| 30 | Sr | 340,053 | 0.69% |
| tail | 59 more elements | 7,205,162 | 14.62% |

Rarest elements: Ne (7), Ar (19), He (394), Kr (1,118), Pa (3,024), Ac (3,210), Pm (6,100), Np
(11,502), Xe (12,681), Pu (15,847) — noble gases and several actinides are essentially absent
(MP's DFT+U inorganic-compound coverage rarely includes noble-gas or transuranic phases).

### Non-structure keys per frame

Every one of the 1,580,395 frames has **exactly the same 12 top-level keys**, all always present
(no missing/optional keys observed):

`structure` (dict), `uncorrected_total_energy` (float), `corrected_total_energy` (float),
`energy_per_atom` (float), `ef_per_atom` (float), `e_per_atom_relaxed` (float),
`ef_per_atom_relaxed` (float), `force` (list), `stress` (list), `magmom` (list), `bandgap`
(float), `mp_id` (str).

Two data-integrity checks:
- `frame['mp_id']` **always equals** the top-level material key it's nested under (0 mismatches
  in 1,580,395 frames) — the authoritative material id is reliable to use directly.
- `structure['charge']` is **always `null`** across every frame scanned — not a usable field.
- Note: the dict *keys* used as frame ids (e.g. `mp-1012897-0-0` nested under top-level
  `mp-1005792`) do not always share the same leading mp-number as their parent — this is cosmetic
  (frame-id strings can reference a different/related mp-id from grouping in the original MP task
  graph), but `frame['mp_id']` is always consistent, so use that field, not the frame-id string,
  if you need the material id.

---

## Implications for training

1. **Class imbalance is severe and dataset-dependent.** COSMOS DBS is H/C/O/N-heavy (73.4% of
   atoms are H+C+O+N; H alone is 35.6%) reflecting its OC20/OMol25/QCML organic-chemistry/molecule
   provenance. MPtrj is O-heavy (37.0% of atoms) with far more metals (Li, Mg, Fe, Mn, V, Co, Ni,
   Zn...) reflecting Materials Project's inorganic-compound coverage. Combined, both datasets have
   long-tail rare elements (noble gases, actinides: single-digit to low-hundreds atom counts) that
   will need class-balanced loss / weighted sampling per `docs/PLAN.md`'s existing plan — this scan
   supplies the actual per-element counts needed to set those weights (`element_hist` in both
   `*_scan_result.json` files).
2. **Multi-element structures dominate, which is exactly the target regime.** 97.3% of COSMOS DBS
   frames and 85.4% of MPtrj frames have ≥3 distinct elements (COSMOS DBS: 40.3% have exactly 4,
   37.8% have 5+; MPtrj: 43.6% have exactly 3, 29.7% have exactly 4). Single- and two-element
   frames are a small minority in both sets (COSMOS DBS: 2.7% combined; MPtrj: 14.6% combined) —
   good, since the model's hard cases (disambiguating type ids across several co-occurring
   elements) are the common case in training data, not an edge case needing special oversampling.
3. **Split strategy must differ per dataset, for different reasons:**
   - *MPtrj*: split by `mp_id` (material), never by frame — 87.3% of materials contribute ≥2
     frames (avg 10.8/material) that are relaxation-trajectory steps of the same composition/near-
     identical geometry; a frame-level random split leaks near-duplicates across train/val. This
     matches the existing `docs/PLAN.md` decision; the scan quantifies the leakage risk.
   - *COSMOS DBS*: split by `db_label` source prefix (18 contiguous blocks: MatPES, OC20, OC22,
     ODAC23, OMol25, QCML × {random,spare,elem}), stratified, since the file is stored as large
     contiguous per-source blocks (not shuffled) and one source (`omol25_random`) alone is 54% of
     frames. There is no frame-level trajectory-leakage risk here (composition-run-length median is
     1 — frames are essentially independent single-point structures), so the only real risk is a
     sequential/positional split accidentally holding out entire sources.
4. **Combined coverage**: both datasets separately cover 89 distinct elements (COSMOS DBS's 89 and
   MPtrj's 89 element sets overlap heavily but are not identical — chalcogen/pnictogen-heavy vs.
   oxide/halide-heavy — a full element-set diff was not computed here but both lists are in the
   raw JSON `element_hist` keys if needed for an exact set operation).
5. **n_atoms ranges are compatible** (COSMOS DBS median 22 atoms, max 615; MPtrj median 22 atoms,
   max 444) — both fit comfortably within typical partial-RDF/type-pair descriptor computation
   budgets (`docs/PLAN.md`'s descriptor pipeline is O(N) per frame for neighbor search), no dataset
   requires special-casing for extreme structure sizes beyond what the other already has.
6. **Both files have 100%-complete metadata for the fields inspected** — no missing cell/pbc
   (COSMOS DBS) and no missing/heterogeneous frame keys (MPtrj) were found, so the preprocessing
   pipeline does not need defensive handling for absent fields in either dataset (aside from
   MPtrj's always-null `charge`, which simply isn't a usable signal).

## Round-3 additions (2026-07-03)

- **WBM relaxed structures**: `Data/WBM/2022-10-19-wbm-computed-structure-entries.json.bz2`
  (figshare id 40344463, 66 MB, md5-verified; pandas-JSON with `material_id`,
  `computed_structure_entry`). The 2024 re-uploads (e.g. 48169600) sit in figshare cold storage
  (HTTP 202 for hours) — the 2022 uploads of the same data are warm. AIS-Square-style direct URL:
  `https://ndownloader.figshare.com/files/40344463`.
- **HfO2 DPA domain**: `Data/openLAM/HfO2/` (AIS-Square dataset id 145, `HfO2_DPA_v1_0`,
  133 MB tar). 114 DeePMD systems, type_map (Hf, O), 57,154 frames. Direct links discoverable via
  `backend.aissquare.com/dpa/detail/datasets?type=datasets&id=<id>` → `store.aissquare.com/...`.
  Other candidate domains noted there: ZrO2-PBEsol (214), Perovskite_oxides-PBEsol (116),
  W_DPA_v1_0 (136), CuZr_metallicglass (354), Solid_State_Electrolyte (217), Electrolyte (216).
- **Hot-MD frames**: `Data/MDgen/hot_frames.extxyz` — MACE-MP-0-small NVT Langevin (300-900 K,
  2 fs, 300 steps, 3 frames/trajectory) on every 50th MPTrj material (train-split element sets
  only, <=200 atoms, melted/exploded guard). `scripts/make_md_frames.py`.
- **v3 processed shards** (`Data/processed/*_v3`): add `env_d` (T,16,16 fp16) + `env_t`
  (T,16,16 int8, -1 pad) per-type neighbor-environment sets for the `use_env` encoder;
  otherwise identical to v2 (964,092 base records + hfo2_v3 3,518 + hotmd_v3 3,276).
