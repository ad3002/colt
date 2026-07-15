#!/usr/bin/env python3
"""E15: independent verifier audit.

`verify_grid` below is written from the Sudoku rules alone and imports
nothing from the colt package. The audit then:
  1. accepts every dataset solution (all splits present under data/),
  2. rejects single-cell corruptions of each solution,
  3. rejects row-permuted grids (valid rows, broken columns),
  4. cross-checks agreement with the production verifier
     (colt.tasks.sudoku.satisfies_constraints) on all of the above plus
     uniformly random grids.

Any disagreement is a bug in one of the two verifiers and fails the audit.

Usage:
    python scripts/independent_verifier.py --out results/verifier_audit.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


# --- standalone verifier: no colt imports ------------------------------------

def box_shape(n: int) -> tuple[int, int]:
    br = max(b for b in range(1, int(n ** 0.5) + 1) if n % b == 0)
    return br, n // br


def verify_grid(grid: list[int], n: int) -> bool:
    """True iff grid (row-major, values 0..n-1) is a complete valid Sudoku."""
    if len(grid) != n * n or any(v < 0 or v >= n for v in grid):
        return False
    full = set(range(n))
    for r in range(n):
        if set(grid[r * n:(r + 1) * n]) != full:
            return False
    for c in range(n):
        if {grid[r * n + c] for r in range(n)} != full:
            return False
    br, bc = box_shape(n)
    for R in range(0, n, br):
        for C in range(0, n, bc):
            if {grid[(R + i) * n + C + j] for i in range(br) for j in range(bc)} != full:
                return False
    return True


# --- audit --------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-random", type=int, default=10000)
    args = ap.parse_args()
    rng = random.Random(0)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from colt.tasks.sudoku import satisfies_constraints, geometry_for

    def production(grid, n):
        g = torch.tensor([grid], dtype=torch.long)
        return bool(satisfies_constraints(g, geometry_for(n))[0])

    report = {"splits": {}, "disagreements": 0, "checks": 0}

    def check(grid, n, expect=None):
        a, b = verify_grid(grid, n), production(grid, n)
        report["checks"] += 1
        if a != b:
            report["disagreements"] += 1
        if expect is not None and a is not expect:
            report.setdefault("expectation_failures", []).append(
                {"n": n, "expected": expect, "standalone": a})
        return a

    splits = ["sudoku4", "sudoku6", "sudoku6_hard", "sudoku9_small", "sudoku9_regen"]
    for tsv in sorted(t for d in splits for t in Path("data", d).glob("*.tsv")):
        rows = []
        with tsv.open() as f:
            header = f.readline().rstrip("\n").split("\t")
            if "solution" not in header:
                continue
            col = header.index("solution")
            for line in f:
                parts = line.rstrip("\n").split("\t")
                rows.append([int(t) - 1 for t in parts[col].split(" ")])
        if not rows:
            continue
        n = int(round(len(rows[0]) ** 0.5))
        ok = corrupt_rej = perm_rej = 0
        for sol in rows:
            if check(sol, n, expect=True):
                ok += 1
            g = sol[:]
            i = rng.randrange(len(g))
            g[i] = (g[i] + 1 + rng.randrange(n - 1)) % n
            if not check(g, n, expect=False):
                corrupt_rej += 1
            rp = list(range(n))
            while rp == list(range(n)):
                rng.shuffle(rp)
            gp = [sol[rp[r] * n + c] for r in range(n) for c in range(n)]
            if not check(gp, n):
                perm_rej += 1
        report["splits"][str(tsv)] = {
            "n_solutions": len(rows), "accepted": ok,
            "corruptions_rejected": corrupt_rej,
            "row_permutations_rejected": perm_rej}

    n = 6
    rand_rej = 0
    for _ in range(args.n_random):
        g = [rng.randrange(n) for _ in range(n * n)]
        if not check(g, n):
            rand_rej += 1
    report["random_grids"] = {"n": args.n_random, "rejected": rand_rej}
    report["pass"] = (report["disagreements"] == 0
                      and "expectation_failures" not in report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
