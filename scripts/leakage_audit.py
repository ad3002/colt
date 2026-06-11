#!/usr/bin/env python3
"""Train/test leakage audit, exact up to the Sudoku symmetry group.

For every (train split, test split) pair the paper evaluates on, reports:

  * exact puzzle overlap   (clues + solution string identical)
  * exact solution overlap (solution grid identical)
  * digit-orbit overlap    (solution grids equal up to a digit relabeling)
  * full-orbit overlap     (equal up to digit relabeling x row permutations
                            within bands x band permutations x column
                            permutations within stacks x stack permutations
                            x transpose when the box shape is square)

The full-orbit check is exact, not sampled: a digit-permutation-invariant
hash (per-digit segmented sum of per-cell random words, mixed and summed
commutatively over digits) collapses the digit factor of the group, and the
cell factor is enumerated exhaustively on the smaller side of the pair
(128 transforms at 4x4, 3,456 at 6x6, 3,359,232 at 9x9). Hash hits are then
verified exactly via first-occurrence canonical relabeling, so the reported
counts have no false positives; false negatives would require a 64-bit hash
collision and are likewise verified out.

Output: results/leakage_audit.json (cited by the paper's dataset-construction
paragraph).

Usage:
    python scripts/leakage_audit.py --out results/leakage_audit.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


def splitmix64(x: np.ndarray) -> np.ndarray:
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & MASK
    z = x
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & MASK
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & MASK
    return z ^ (z >> np.uint64(31))


def load_split(tsv: Path) -> tuple[int, np.ndarray, np.ndarray]:
    clues, sols = [], []
    with tsv.open() as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            clues.append([int(t) for t in parts[1].split(" ")])
            sols.append([int(t) - 1 for t in parts[2].split(" ")])
    n = int(round(len(sols[0]) ** 0.5))
    return n, np.array(clues, dtype=np.int8), np.array(sols, dtype=np.int8)


def cell_perm_factors(n: int, br: int, bc: int):
    """All row orders (band perms x within-band perms) and column orders."""
    def orders(block: int, count: int):
        out = []
        for block_perm in itertools.permutations(range(count)):
            for inner in itertools.product(
                    *[list(itertools.permutations(range(block))) for _ in range(count)]):
                order = []
                for b in block_perm:
                    order.extend(b * block + i for i in inner[b])
                out.append(order)
        return np.array(out, dtype=np.int64)
    return orders(br, n // br), orders(bc, n // bc)


def relabel(grid: np.ndarray) -> tuple:
    """First-occurrence canonical form of the digit-permutation orbit."""
    m, nxt, out = {}, 0, []
    for v in grid.tolist():
        if v not in m:
            m[v] = nxt
            nxt += 1
        out.append(m[v])
    return tuple(out)


def digit_invariant_hash(sols: np.ndarray, n: int, r1: np.ndarray) -> np.ndarray:
    """Digit-permutation-invariant 64-bit hash of each solution grid."""
    h = np.zeros(len(sols), dtype=np.uint64)
    for d in range(n):
        hd = ((sols == d) * r1[None, :]).sum(axis=1, dtype=np.uint64)
        h = (h + splitmix64(hd)) & MASK
    return h


def orbit_hashes_for_grid(sol: np.ndarray, n: int, row_orders, col_orders, r1table) -> np.ndarray:
    """Hash of every cell-transform of one grid; digit factor handled by invariance."""
    h = np.zeros((len(row_orders), len(col_orders)), dtype=np.uint64)
    grid = sol.reshape(n, n)
    for d in range(n):
        rows_d, cols_d = np.nonzero(grid == d)
        hd = np.zeros_like(h)
        for k in range(len(rows_d)):
            hd = (hd + r1table[row_orders[:, rows_d[k]][:, None],
                               col_orders[:, cols_d[k]][None, :]]) & MASK
        h = (h + splitmix64(hd)) & MASK
    return h


def audit_pair(name, train_tsv: Path, test_tsv: Path, rng: np.random.Generator) -> dict:
    n, train_clues, train_sols = load_split(train_tsv)
    n2, test_clues, test_sols = load_split(test_tsv)
    assert n == n2
    br = max(b for b in range(1, int(n**0.5) + 1) if n % b == 0)
    bc = n // br
    r1 = rng.integers(0, 2**63, size=n * n, dtype=np.uint64)
    r1table = r1.reshape(n, n)

    def key(c, s):
        return c.tobytes() + s.tobytes()

    train_puzzles = {key(c, s) for c, s in zip(train_clues, train_sols)}
    exact_puzzle = sum(key(c, s) in train_puzzles for c, s in zip(test_clues, test_sols))
    train_raw = {s.tobytes() for s in train_sols}
    exact_sol = sum(s.tobytes() in train_raw for s in test_sols)
    train_relab = {relabel(s) for s in train_sols}
    digit_orbit = sum(relabel(s) in train_relab for s in test_sols)

    row_orders, col_orders = cell_perm_factors(n, br, bc)
    n_cell_transforms = len(row_orders) * len(col_orders) * (2 if br == bc else 1)

    # Enumerate the cell group on the smaller side; look up invariant hashes
    # of the other side (plus its transpose when the box shape is square).
    small_is_train = len(train_sols) <= len(test_sols)
    small_sols = train_sols if small_is_train else test_sols
    big_sols = test_sols if small_is_train else train_sols
    big_list = [big_sols]
    if br == bc:
        big_list.append(np.array([s.reshape(n, n).T.ravel() for s in big_sols], dtype=np.int8))
    big_hash = np.concatenate([digit_invariant_hash(b, n, r1) for b in big_list])
    big_hash_sorted = np.sort(big_hash)

    t0 = time.perf_counter()
    candidate_pairs = []
    for si, sol in enumerate(small_sols):
        h = orbit_hashes_for_grid(sol, n, row_orders, col_orders, r1table).ravel()
        if np.searchsorted(big_hash_sorted, h, side="right").max(initial=0) > 0:
            hits = np.isin(h, big_hash_sorted)
            if hits.any():
                candidate_pairs.append(si)
    enum_s = time.perf_counter() - t0

    # Exact verification of candidates: full transform + canonical relabel.
    full_orbit_test_idx: set[int] = set()
    for si in candidate_pairs:
        sol = small_sols[si].reshape(n, n)
        small_forms = set()
        for ro in row_orders:
            sub = sol[ro, :]
            for co in col_orders:
                g = sub[:, co].ravel()
                small_forms.add(relabel(g))
                if br == bc:
                    small_forms.add(relabel(sub[:, co].T.ravel()))
        if small_is_train:
            for ti, ts in enumerate(test_sols):
                if relabel(ts) in small_forms:
                    full_orbit_test_idx.add(ti)
        else:
            for tr in train_sols:
                if relabel(tr) in small_forms:
                    full_orbit_test_idx.add(si)
                    break
    full_orbit = len(full_orbit_test_idx)

    res = {
        "train": str(train_tsv), "test": str(test_tsv),
        "n": n, "box": f"{br}x{bc}",
        "n_train": len(train_sols), "n_test": len(test_sols),
        "distinct_train_solutions": len(train_raw),
        "distinct_test_solutions": len({s.tobytes() for s in test_sols}),
        "cell_transforms_enumerated": int(n_cell_transforms),
        "exact_puzzle_overlap": int(exact_puzzle),
        "exact_solution_overlap": int(exact_sol),
        "digit_orbit_overlap": int(digit_orbit),
        "full_orbit_overlap": int(full_orbit),
        "full_orbit_overlap_frac_of_test": round(full_orbit / len(test_sols), 4),
        "enumeration_seconds": round(enum_s, 1),
    }
    print(name, json.dumps(res, indent=1), file=sys.stderr)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/leakage_audit.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(20260611)
    pairs = [
        ("sudoku4", "data/sudoku4/train.tsv", "data/sudoku4/test.tsv"),
        ("sudoku6", "data/sudoku6/train.tsv", "data/sudoku6/test.tsv"),
        ("sudoku6_hard_vs_sudoku6train", "data/sudoku6/train.tsv", "data/sudoku6_hard/test.tsv"),
        ("sudoku9_small", "data/sudoku9_small/train.tsv", "data/sudoku9_small/test.tsv"),
    ]
    out = {}
    for name, tr, te in pairs:
        if not Path(tr).exists() or not Path(te).exists():
            print(f"skip {name}: missing files", file=sys.stderr)
            continue
        out[name] = audit_pair(name, Path(tr), Path(te), rng)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
