# TypeSHI — type-id → element inference from geometry

Given an MD file that stores only integer atom **type ids** (no element symbols, often no usable
metadata), predict top-K element candidates per type id from geometry alone — `positions`,
`type_ids`, optional `cell`, optionally multiple frames. Public name **TypeSHI** = Type-id →
Species Hypothesis Inference. Programmatic entry: `typeid2elem.api.predict_structure`.

**Read `AGENTS.md` first.** It carries the working knowledge — commands, the hard rules with
their rationale, the traps already hit, and eval etiquette. This file is only the inviolable
minimum; everything else lives in `AGENTS.md` and `docs/`.

## Inviolable (violating these breaks the project's premise)
- **Never feed masses — or any file metadata — to the model.** They are a printed baseline and a
  source of eval ground truth only; the owner considers anything else cheating (one eval file has
  *wrong* masses on purpose).
- **Never scale-normalize or scale-augment geometry.** Absolute ångström distances are the signal.
- The repo-root **wandb key file** is gitignored (via `*_API_KEY`) — never print or commit it.

## Environment
`uv` project, Python 3.12, RTX 4090; run everything as `uv run python ...` from repo root. torch
pinned to **cu128** wheels; neighbor lists via **matscipy** only. The why, plus all commands, are
in `AGENTS.md`; full data→train→export pipeline in `docs/REPRODUCE.md`.

## Where things are
`src/typeid2elem/` (library) · `scripts/` · `weights/` (2 MB committed runtime bundle) ·
`runs/`, `Data/` (gitignored). Docs: `AGENTS.md`, `README.md`, and `docs/` — `MODEL.md` (design +
why), `REPORT.md`, `REPRODUCE.md`, `EXPERIMENTS.md` (append-only log), `EVAL_CASES.md` (the real
user files + expected elements), `PLAN.md`, `DATA.md`, `EXPORT.md`.
