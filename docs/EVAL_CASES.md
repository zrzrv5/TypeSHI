# LAMMPS & Simulation File Evaluation Cases for ML Element Prediction

> **Corrections (2026-07-03, from user):** DimTest/true_dpnpy.xyz removed (not a meaningful case).
> test_Na/* is **Na-S-P-O** (types 1-4 per test_Na/input.lammps masses), not elemental Na.
> LACO/RMC/TT3GR.data and TT3RR.data are **Li Al Cl O** (types 1-4).
> Added: HfBaO/amorGen (Ba-doped amorphous HfO2; Masses -> Hf/Ba/O).
> The scored benchmark lives in `scripts/evaluate_bench.py` (16 systems / 52 type-ids).

This document catalogs diverse simulation files suitable for evaluating an ML model that predicts element types from atomic geometry. Files are drawn from multiple formats (LAMMPS data, VASP, XYZ, PDB, etc.) and represent varied chemical systems and structure sizes.

## Summary Statistics

- **Total files found and screened**: 40+
- **Files with element symbols built into file format** (XYZ, PDB, etc.): 9
- **Files with element composition from Masses section** (LAMMPS ≥ 0.5 amu tolerance): 4  
- **LAMMPS data files without Masses** (need ground-truth mapping): 5

---

## Files with Built-in Element Symbols (Perfect Ground Truth)

These files contain element symbols directly in the file format. No interpretation needed.

| Path | Format | n_atoms | Elements | Chemical System | Notes |
|------|--------|---------|----------|-----------------|-------|
| `/home/zrzrv5/Documents/CHO/iter6.xyz` | XYZ | 21 | C, H, O | CHO | Small organic structure |
| `/home/zrzrv5/Documents/LiSiPS_Opt/templates/frag/Li.xyz` | XYZ | 1 | Li | LiSiPS | Single atom |
| `/home/zrzrv5/Documents/LiSiPS_Opt/templates/frag/P2S6.xyz` | XYZ | 8 | P, S | LiSiPS | Molecule (P₂S₆) |
| `/home/zrzrv5/Documents/RMC/TT1/TT1_5x5x5.pdb` | PDB | 4750 | Al, Cl, Li, O | Ionic | Large structure |
| `/home/zrzrv5/Documents/RMC/TT1/restart.pdb` | PDB | 4750 | Al, Cl, Li, O | Ionic | Same system, restart snapshot |
| `/home/zrzrv5/Documents/RMC_ppt/Fit2/TT3G.pdb` | PDB | 4640 | Al, Cl, Li, O | Ionic | RMC fit variant |
| `/home/zrzrv5/Documents/RMC_ppt/Fit2/trajectory.xyz` | XYZ | 4640 | Al, Cl, Li, O | Ionic | MD trajectory snapshot |
| `/home/zrzrv5/Documents/RMC_ppt/atomicNiTi/system.pdb` | PDB | 6750 | Ni, Ti | NiTi | Metallic alloy |

---

## Files with Masses Mapping to Elements (Reliable Ground Truth)

These files contain a Masses section in LAMMPS format where each mass maps to a unique element within 0.5 amu tolerance.

| Path | Format | n_atoms | Element Composition | Chemical System | Notes |
|------|--------|---------|-------------------|-----------------|-------|
| `/home/zrzrv5/Documents/LPS_dp/Data/67Li2S_33P2S5.data` | LAMMPS | 8020 | Li, P, S | Li₂S-P₂S₅ mixture | Large system |
| `/home/zrzrv5/Documents/LPS_dp/Data/70Li2S_30P2S5.data` | LAMMPS | 8069 | Li, P, S | Li₂S-P₂S₅ mixture | Large system variant |
| `/home/zrzrv5/Documents/LPS_dp/Data/75Li2S_25P2S5.data` | LAMMPS | 8020 | Li, P, S | Li₂S-P₂S₅ mixture | Third composition |
| `/home/zrzrv5/Documents/PlumedPG/DIFPG/EQ.300K.data` | LAMMPS | 936 | Na, B, H | Na₂B₁₂H₁₂ | Borohydride |
| `/home/zrzrv5/Documents/PlumedPG/DIFPG/Na2B12H12LoM_3x2x3.data` | LAMMPS | 936 | Na, B, H | Na₂B₁₂H₁₂ | Supercell variant |

---

## LAMMPS Data Files Without Masses (Need Ground-Truth Mapping)

These contain valid LAMMPS atom type information but no Masses section. Composition inferred from filename or directory.

| Path | Format | n_atoms | Inferred Composition | Chemical System |
|------|--------|---------|-------------------|-----------------|
| `/home/zrzrv5/Documents/LACO/RMC/TT3GR.data` | LAMMPS | 4640 | Unknown | Unknown |
| `/home/zrzrv5/Documents/LACO/RMC/TT3RR.data` | LAMMPS | 4640 | Unknown | Unknown |
| `/home/zrzrv5/Documents/test_Na/025.0000.lmp` | LAMMPS | 128 | Na | Na metal |
| `/home/zrzrv5/Documents/test_Na/confs/000.0000.lmp` | LAMMPS | 48 | Na | Na metal |
| `/home/zrzrv5/Documents/test_Na/confs/025.0000.lmp` | LAMMPS | 128 | Na | Na metal |

---

## LAMMPS Dump/Trajectory Files (Require Sibling Data Files for Masses)

| Path | Format | n_atoms | Notes |
|------|--------|---------|-------|
| `/home/zrzrv5/Documents/ZnO/MeltRun/Min.dump` | LAMMPS dump | Variable | Check `/home/zrzrv5/Documents/ZnO/MeltRun/*.data` for masses |
| `/home/zrzrv5/Documents/ZnO/MeltRun/94@1500K.dump` | LAMMPS dump | Variable | ZnO melt trajectory |
| `/home/zrzrv5/Documents/ZnO/MeltRun/Melt.dump` | LAMMPS dump | Variable | ZnO melt run |
| `/home/zrzrv5/Documents/test_Na/traj/120.lammpstrj` | LAMMPS traj | 128 | Na MD trajectory |
| `/home/zrzrv5/Documents/test_Na/traj/25000.lammpstrj` | LAMMPS traj | 128 | Na MD at later timestep |

---

## Chemistry Coverage by System

### Oxides (1 system, 1 file with symbols)
- **NiTi**: 6,750-atom metallic alloy (PDB with symbols)

### Borohydrides (1 system, 2 files)
- **Na₂B₁₂H₁₂**: 936 atoms, 3 elements — both with reliable masses

### Lithium Conductors (1 main system, 3 files)
- **LPS (Li₂S-P₂S₅ mixtures)**: 8,000–8,069 atoms, three compositions — all with reliable masses

### Ionic Materials (1 system, 5 files)
- **Mixed Li-Al-Cl-O**: 4,640–4,750 atoms — all with element symbols (PDB/XYZ)

### Elemental & Metals (1 system, 4 files)
- **Na metal**: 48–128 atoms — LAMMPS format, need ground truth

### Small Molecules & Fragments (2 systems, 3 files)
- **CHO**: 21 atoms — XYZ with symbols
- **P₂S₆, Li, P₂S₆**: 1–8 atoms — XYZ with symbols

### Mixed/Complex (1 file)
- **7-element mixture** (Al, C, Cl, H, Li, Na, O): 131 atoms — XYZ with symbols

---

## Recommended Priority Evaluation Set

**For a balanced, diverse baseline, use these 10 files:**

1. **CHO/iter6.xyz** (21 atoms, 3 elem, symbols)  
   → Test small organic structure
2. **DimTest/data_gen/true_dpnpy.xyz** (131 atoms, 7 elem, symbols)  
   → Test high-diversity multi-element system
3. **RMC_ppt/atomicNiTi/system.pdb** (6,750 atoms, 2 elem, symbols)  
   → Test large metallic system
4. **RMC/TT1/TT1_5x5x5.pdb** (4,750 atoms, 4 elem, symbols)  
   → Test large ionic system
5. **LPS_dp/Data/70Li2S_30P2S5.data** (8,069 atoms, 3 elem, masses)  
   → Test very large lithium-ion system
6. **PlumedPG/DIFPG/Na2B12H12LoM_3x2x3.data** (936 atoms, 3 elem, masses)  
   → Test borohydride/cluster system
7. **LiSiPS_Opt/templates/frag/P2S6.xyz** (8 atoms, 2 elem, symbols)  
   → Test small molecular fragment
8. **test_Na/traj/25000.lammpstrj** (128 atoms, Na, trajectory)  
   → Test simple metal trajectory (if ground truth provided)
9. **RMC_ppt/Fit2/trajectory.xyz** (4,640 atoms, 4 elem, symbols)  
   → Test MD trajectory with known elements
10. **RMC_ppt/Fit2/TT3G.pdb** (4,640 atoms, 4 elem, symbols)  
    → Test secondary ionic structure variant

---

## Format Distribution

| Format | Count | Perfect Ground Truth |
|--------|-------|----------------------|
| XYZ | 5 | ✓ (element symbols) |
| PDB | 4 | ✓ (element symbols) |
| LAMMPS data | 9 | 5 with masses, 4 without |
| LAMMPS dump/traj | 5 | Partial (sibling data) |

---

## Size Distribution

| Category | Count | Size Range |
|----------|-------|------------|
| **Tiny** (≤ 10 atoms) | 2 | 1–8 |
| **Small** (11–200 atoms) | 2 | 21–131 |
| **Medium** (201–1,000 atoms) | 2 | 936 |
| **Large** (1,001–5,000 atoms) | 5 | 4,640–4,750 |
| **Very Large** (> 5,000 atoms) | 4 | 6,750–8,069 |

---

## Notes for Model Development

1. **Element Symbol Sources (Ideal)**: 9 files provide ground truth via XYZ/PDB/CIF formats with built-in element symbols. Use these as your most reliable test set.

2. **Mass-Based Sources**: 5 LAMMPS data files have Masses that reliably map to elements. These are appropriate if your model can handle type IDs rather than raw symbols.

3. **Large Structures**: Most large systems (> 5,000 atoms) are from LPS and NiTi; use for scalability tests.

4. **Trajectory Data**: Several dump/lammpstrj files available in `test_Na/traj/` and `ZnO/MeltRun/` if you need temporal/dynamic evaluation. Pair with sibling `.data` files for ground truth.

5. **Chemical Diversity**: 7 distinct element types (Na, Li, Al, P, S, B, H, C, O, Ni, Ti, Zn) represented across 10+ unique chemical systems.

---

## Files Excluded (Pre-existing)

- `/home/zrzrv5/Documents/LiSiPS_Opt/Run_Jan2/490942_64507963/02_MSD/Data/400K.run.data`
- `/home/zrzrv5/Documents/LACO/Relax_TT1/TT1.data`
- `/home/zrzrv5/Documents/NaCBH/iter58/Na2B12H12.Lo/Data/600K.data`

## MLFF folder additions (2026-07-03, agent-collected + hand-verified)

Source: SMB share `smb://yolonas.local/mainstorage/SWD/MLFF` (POSCARs copied into
`Data/eval_cases/MLFF/` — the gvfs mount is session-dependent, don't reference it in BENCH).
12 cases, truth = POSCAR element headers (all verified by hand; the collection agent misreported
one: "NCM-poly" has NO Zr — header is Li Ni O Co Mn, 84 atoms). Chemistry: binary oxides
(Li2O, MnO2, NiO, Co3O4) + NCM cathodes (811/333 stoichiometries, O-vacancy variants at 1200K,
Zr/Al/Fe dopants, a no-Li variant, one 10K MD snapshot). Bench grows 52 -> 102 scoreable type-ids.
Production (6-model, conf fuse, decode) on the new 50: 27/39/44 (54%/78%/88%) — Ni/Co/Mn
radius-twin confusions dominate top-1 misses, top-5 nearly saturates; hard cases: Li2O (0/2 top-1),
Al-doped no-Li NCM (1/5 top-1).

**Update (same day):** per user request the 6 MD/defect cases now use CONTCAR (final
configuration) instead of POSCAR (initial); CONTCARs carried trailing predictor-corrector blocks
that break ase.io.vasp velocity parsing — local copies truncated to header+coordinates. Bench on
final geometries: **56/82/94 of 102** (top-1 -2 from thermalized 1200K structures, top-5 +3;
new-case subtotal 25/40/47 of 50). This is the standing baseline. MDRun folder collection pending.

## MDRun folder additions (2026-07-03, agent-collected + hand-verified)

Source: `smb://yolonas.local/mainstorage/SWD/MDRun`, copied to `Data/eval_cases/MDRun/`.
6 curated cases (agent proposed 12; 6 were duplicate representations of the same systems):
LFP333.min (Li Fe P O, Masses), LFMMP.min (+ Mn Mg, 6 types, Masses), LFMP333.93 (M-site SPLIT
Fe0.3/Mn0.7 across types 1/5; input script has the WRONG mass on the Mn type -- stoichiometry
32:76 settles truth; mass lookup would fail here), LFP core-shell (polarizable model, Fe/O each
split into core+shell particle pairs at near-zero separation -- pathological geometry on purpose),
NMC523 (Masses with element comments), LiNiO2-5000 (3 of 5 declared types present; stoichiometry
1296:1296:2592 confirms Li Ni O).

Bench now **131 scoreable type-ids**. Production (6-model, conf, decode): **78/107/122**
(60%/82%/93%). MDRun subtotal 22/25/28 of 29 — classical-potential phosphates are friendly;
core-shell case scores 4/6 top-1 with both O types and Li/P correct (split-type augmentation
transfers), Fe core+shell at rank 5.
