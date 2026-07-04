"""One-time repair: clip non-finite / oversized RDF values in existing shards.

(The MPTrj preprocessing that generated them was running with the pre-clip
descriptor code; cheaper to repair than regenerate.)

Usage: uv run python scripts/repair_shards.py Data/processed/cosmos_dbs [dirs...]
"""

import sys
from pathlib import Path

import numpy as np

for d in sys.argv[1:]:
    fixed = 0
    for f in sorted(Path(d).glob("*.npz")):
        z = dict(np.load(f))
        rdf32 = z["rdf"].astype(np.float32)
        n_bad = int((~np.isfinite(rdf32)).sum() + (rdf32 > 1e4).sum())
        if n_bad:
            z["rdf"] = np.nan_to_num(rdf32, posinf=1e4, neginf=0.0).clip(0, 1e4).astype(np.float16)
            np.savez_compressed(f, **z)
            fixed += 1
    print(f"{d}: repaired {fixed} shard(s)")
