"""Mine element co-occurrence statistics from processed shards -> costats.npz.

P(element), P(element pair in same structure) over unique element SETS (each
gid = element set counted once, so relaxation frames/augmentations don't skew).

Usage: uv run python scripts/mine_costats.py Data/processed/cosmos_dbs Data/processed/mptrj
"""

import sys
from pathlib import Path

import numpy as np

N_EL = 94
seen: dict[int, frozenset] = {}
for d in sys.argv[1:]:
    for f in sorted(Path(d).glob("*.npz")):
        z = np.load(f)
        for gid, labels in zip(z["gid"], z["labels"]):
            if gid not in seen:
                seen[int(gid)] = frozenset(int(l) for l in labels)

single = np.zeros(N_EL)
pair = np.zeros((N_EL, N_EL))
for els in seen.values():
    for a in els:
        single[a - 1] += 1
        for b in els:
            if b > a:
                pair[a - 1, b - 1] += 1
                pair[b - 1, a - 1] += 1

n_sets = len(seen)
np.savez(Path("Data/processed/costats.npz"), single=single, pair=pair,
         n_sets=np.array(n_sets))
print(f"mined {n_sets} unique element sets; "
      f"most common: {np.argsort(single)[::-1][:8] + 1} (Z)")
