"""Full streaming scan of Data/MPtrj/MPtrj_2022.9_full.json.

Structure (verified with a manual ijson probe first):
    { mp_id: { frame_id: { structure: {@module,@class,charge,lattice,sites},
                            uncorrected_total_energy, corrected_total_energy,
                            energy_per_atom, ef_per_atom, e_per_atom_relaxed,
                            ef_per_atom_relaxed, force, stress, magmom,
                            bandgap, mp_id }, ... }, ... }

Each `sites` entry looks like:
    {'species': [{'element': 'Sm', 'occu': 1}], 'abc': [...], 'xyz': [...],
     'label': 'Sm', 'properties': {}}

Access pattern: ijson.kvitems(f, '', use_float=True) with the C backend
(yajl2_c, auto-selected) streams top-level (mp_id -> value) pairs; each value
is a small, fully-materialized dict (one material's frames), so memory stays
bounded. use_float=True avoids slow Decimal construction for every numeric
leaf (force/stress/magmom arrays are large per frame).

This does a FULL pass (measured ~1400 materials/sec on a warm-cache probe of
the first 2000 materials => full file of O(100k) materials should finish in a
few minutes, well under the 15 min budget).
"""
import ijson
import json
import sys
import time
from collections import Counter

PATH = "Data/MPtrj/MPtrj_2022.9_full.json"


def main():
    t0 = time.time()
    n_materials = 0
    n_frames = 0
    frames_per_material = Counter()
    n_atoms_counter = Counter()
    n_distinct_elem_counter = Counter()
    element_counter = Counter()
    frame_key_counter = Counter()  # non-structure keys seen in frames
    mismatched_mpid_examples = []  # frame_id's mp-id prefix != top-level mp_id, or frame['mp_id'] != top-level
    sample_frame_keys = None
    charge_nonnull = 0
    charge_null = 0

    with open(PATH, "rb") as f:
        parser = ijson.kvitems(f, "", use_float=True)
        for mp_id, mat in parser:
            n_materials += 1
            nf = len(mat)
            n_frames += nf
            bucket = nf if nf < 20 else ">=20"
            frames_per_material[bucket] += 1

            for frame_id, frame in mat.items():
                for k in frame.keys():
                    frame_key_counter[k] += 1
                if sample_frame_keys is None:
                    sample_frame_keys = {k: (type(v).__name__) for k, v in frame.items()}

                inner_mpid = frame.get("mp_id")
                if inner_mpid != mp_id and len(mismatched_mpid_examples) < 10:
                    mismatched_mpid_examples.append((mp_id, frame_id, inner_mpid))

                struct = frame.get("structure", {})
                if struct.get("charge") is not None:
                    charge_nonnull += 1
                else:
                    charge_null += 1

                sites = struct.get("sites", [])
                n = len(sites)
                n_atoms_counter[n] += 1

                elems = []
                for site in sites:
                    sp = site.get("species", [])
                    # a site can (rarely) have partial/mixed occupancy species list;
                    # take the majority (first / highest occu) element as "the" element
                    if sp:
                        if len(sp) > 1:
                            best = max(sp, key=lambda s: s.get("occu", 0))
                        else:
                            best = sp[0]
                        elems.append(best.get("element"))
                for e in elems:
                    element_counter[e] += 1
                distinct = len(set(elems))
                dbucket = distinct if distinct < 5 else "5+"
                n_distinct_elem_counter[dbucket] += 1

            if n_materials % 10000 == 0:
                elapsed = time.time() - t0
                print(f"...{n_materials} materials / {n_frames} frames scanned "
                      f"({elapsed:.0f}s elapsed, {n_materials/elapsed:.0f} mat/s)",
                      file=sys.stderr)

    elapsed = time.time() - t0
    result = {
        "n_materials": n_materials,
        "n_frames": n_frames,
        "elapsed_sec": elapsed,
        "frames_per_material_hist": {str(k): v for k, v in frames_per_material.most_common()},
        "n_atoms_hist": dict(sorted(n_atoms_counter.items())),
        "n_distinct_elem_hist": {str(k): v for k, v in n_distinct_elem_counter.items()},
        "element_hist": dict(element_counter.most_common()),
        "frame_key_counter": dict(frame_key_counter.most_common()),
        "sample_frame_keys_with_types": sample_frame_keys,
        "mismatched_mpid_examples": mismatched_mpid_examples,
        "charge_nonnull": charge_nonnull,
        "charge_null": charge_null,
    }
    with open("scripts/inspect/mptrj_scan_result.json", "w") as out:
        json.dump(result, out, indent=2)
    print(f"DONE. n_materials={n_materials} n_frames={n_frames} elapsed={elapsed:.0f}s")


if __name__ == "__main__":
    main()
