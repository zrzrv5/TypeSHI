"""Fast manual line-scanner for Data/COSMOS/DBS/dbs_total.extxyz.

extxyz frame format (verified against ase.io.iread on first 5 frames):
    line 1: integer N = number of atoms
    line 2: comment/info line, space-separated key=value pairs (quoted values may
            contain spaces), always includes Lattice=, Properties=, pbc=, db_label=,
            energy=, stress=, magmom=, free_energy=
    lines 3..N+2: "<element> x y z fx fy fz magmom"

This script does ONE pass over the file, parsing only:
  - the atom-count header line -> n_atoms per frame
  - the comment line -> presence of Lattice/pbc/db_label + which key= fields appear
  - the element (first whitespace token) of each atom line -> element histogram
      and per-frame distinct-element count
  - db_label value, to look at frame-to-frame grouping / trajectories

It avoids ase entirely so it can process the full ~780MB / all frames in a single
pass at ~text-scan speed.
"""
import re
import sys
import json
from collections import Counter, defaultdict

PATH = "Data/COSMOS/DBS/dbs_total.extxyz"

KEY_RE = re.compile(r'(\w+)=')

def main():
    n_atoms_counter = Counter()
    n_distinct_elem_counter = Counter()
    element_counter = Counter()
    frame_count = 0
    info_key_counter = Counter()
    has_lattice = 0
    has_pbc = 0
    missing_lattice_examples = []
    missing_pbc_examples = []
    db_label_prefix_counter = Counter()
    db_label_examples = defaultdict(list)

    # for consecutive-run analysis (same db_label "prefix_number" family & same composition)
    prev_prefix = None
    run_lengths = []
    cur_run = 0

    # composition-based consecutive check (multiset of elements identical to prev frame)
    prev_comp = None
    comp_run_lengths = []
    comp_cur_run = 0

    with open(PATH, "r") as f:
        line = f.readline()
        lineno = 1
        while line:
            line_stripped = line.strip()
            if line_stripped == "":
                line = f.readline()
                lineno += 1
                continue
            # header line: must be an integer
            try:
                n = int(line_stripped)
            except ValueError:
                raise RuntimeError(f"Expected atom-count header at line {lineno}, got: {line_stripped!r}")

            comment = f.readline()
            lineno += 1
            frame_count += 1
            n_atoms_counter[n] += 1

            # parse comment line keys
            keys_found = set(KEY_RE.findall(comment))
            for k in keys_found:
                info_key_counter[k] += 1
            if "Lattice" in keys_found:
                has_lattice += 1
            elif len(missing_lattice_examples) < 5:
                missing_lattice_examples.append((frame_count, comment.strip()[:200]))
            if "pbc" in keys_found:
                has_pbc += 1
            elif len(missing_pbc_examples) < 5:
                missing_pbc_examples.append((frame_count, comment.strip()[:200]))

            m = re.search(r'db_label=(\S+)', comment)
            if m:
                label = m.group(1)
                # split trailing _<number> if present
                pm = re.match(r'(.+)_(\d+)$', label)
                prefix = pm.group(1) if pm else label
                db_label_prefix_counter[prefix] += 1
                if len(db_label_examples[prefix]) < 3:
                    db_label_examples[prefix].append(label)
            else:
                prefix = None

            if prefix == prev_prefix:
                cur_run += 1
            else:
                if prev_prefix is not None:
                    run_lengths.append(cur_run)
                cur_run = 1
                prev_prefix = prefix

            # read N atom lines, grab element = first token
            frame_elements = []
            for _ in range(n):
                atom_line = f.readline()
                lineno += 1
                if not atom_line:
                    raise RuntimeError(f"Unexpected EOF reading atoms at line {lineno}")
                elem = atom_line.split(None, 1)[0]
                frame_elements.append(elem)
                element_counter[elem] += 1

            distinct = len(set(frame_elements))
            bucket = distinct if distinct < 5 else "5+"
            n_distinct_elem_counter[bucket] += 1

            comp = tuple(sorted(set(frame_elements)))
            if comp == prev_comp:
                comp_cur_run += 1
            else:
                if prev_comp is not None:
                    comp_run_lengths.append(comp_cur_run)
                comp_cur_run = 1
                prev_comp = comp

            line = f.readline()
            lineno += 1

            if frame_count % 200000 == 0:
                print(f"...{frame_count} frames scanned", file=sys.stderr)

    if prev_prefix is not None:
        run_lengths.append(cur_run)
    if prev_comp is not None:
        comp_run_lengths.append(comp_cur_run)

    result = {
        "frame_count": frame_count,
        "n_atoms_hist": dict(sorted(n_atoms_counter.items())),
        "n_distinct_elem_hist": {str(k): v for k, v in n_distinct_elem_counter.items()},
        "element_hist": dict(element_counter.most_common()),
        "info_key_counter": dict(info_key_counter.most_common()),
        "has_lattice": has_lattice,
        "has_pbc": has_pbc,
        "missing_lattice_examples": missing_lattice_examples,
        "missing_pbc_examples": missing_pbc_examples,
        "db_label_prefix_counter": dict(db_label_prefix_counter.most_common()),
        "db_label_examples": {k: v for k, v in list(db_label_examples.items())[:50]},
        "db_label_run_length_stats": {
            "n_runs": len(run_lengths),
            "mean": sum(run_lengths) / len(run_lengths) if run_lengths else 0,
            "max": max(run_lengths) if run_lengths else 0,
            "median": sorted(run_lengths)[len(run_lengths)//2] if run_lengths else 0,
            "n_runs_len1": sum(1 for r in run_lengths if r == 1),
        },
        "composition_run_length_stats": {
            "n_runs": len(comp_run_lengths),
            "mean": sum(comp_run_lengths) / len(comp_run_lengths) if comp_run_lengths else 0,
            "max": max(comp_run_lengths) if comp_run_lengths else 0,
            "median": sorted(comp_run_lengths)[len(comp_run_lengths)//2] if comp_run_lengths else 0,
        },
    }
    with open("scripts/inspect/extxyz_scan_result.json", "w") as out:
        json.dump(result, out, indent=2)
    print("DONE. frame_count =", frame_count)


if __name__ == "__main__":
    main()
