#!/usr/bin/env python3
"""Measure, rather than assert, the classical-reference claim.

The paper states that a plain exact backtracking solver closes every test set
in this study quickly. This script turns that sentence into numbers: it runs an
MRV backtracking solver with constraint propagation over (a) every Sudoku test
split used in the paper and (b) freshly generated 3-coloring test instances at
each swept density (same generator and seeds as scripts/phase_transition.py),
recording per-instance wall time and decision counts.

Output: results/classical_reference.json, cited by the paper's
"What is being compared" paragraph and the boundary section.

Usage:
    python scripts/classical_reference.py --out results/classical_reference.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ----------------------------------------------------------------------------
# Sudoku: MRV backtracking with propagation (bitmask domains)
# ----------------------------------------------------------------------------


def sudoku_geometry(n: int):
    br = max(b for b in range(1, int(n**0.5) + 1) if n % b == 0)
    bc = n // br
    peers = [set() for _ in range(n * n)]
    for i in range(n * n):
        r, c = divmod(i, n)
        for cc in range(n):
            peers[i].add(r * n + cc)
        for rr in range(n):
            peers[i].add(rr * n + c)
        for j in range(n * n):
            jr, jc = divmod(j, n)
            if jr // br == r // br and jc // bc == c // bc:
                peers[i].add(j)
        peers[i].discard(i)
    return peers


def solve_sudoku(clues: list[int], n: int, peers) -> tuple[bool, int]:
    """clues: -1 blank else 0-indexed value. Returns (solved, decisions)."""
    full = (1 << n) - 1
    dom = [full] * (n * n)
    g = clues[:]
    stack = []
    for i, v in enumerate(g):
        if v >= 0:
            dom[i] = 1 << v
            stack.append(i)
    while stack:
        i = stack.pop()
        b = dom[i]
        for p in peers[i]:
            if dom[p] & b:
                dom[p] &= ~b
                if dom[p] == 0:
                    return False, 0
                if dom[p].bit_count() == 1 and g[p] < 0:
                    g[p] = dom[p].bit_length() - 1
                    stack.append(p)
    decisions = 0

    def bt() -> bool:
        nonlocal decisions
        best, bn = -1, n + 1
        for i in range(n * n):
            if g[i] < 0:
                c = dom[i].bit_count()
                if c < bn:
                    bn, best = c, i
        if best == -1:
            return True
        for v in range(n):
            if not dom[best] & (1 << v):
                continue
            decisions += 1
            sd, sg = dom[:], g[:]
            dom[best] = 1 << v
            g[best] = v
            ok = True
            prop = [best]
            while prop and ok:
                j = prop.pop()
                b = dom[j]
                for p in peers[j]:
                    if dom[p] & b:
                        dom[p] &= ~b
                        if dom[p] == 0:
                            ok = False
                            break
                        if dom[p].bit_count() == 1 and g[p] < 0:
                            g[p] = dom[p].bit_length() - 1
                            prop.append(p)
            if ok and bt():
                return True
            dom[:], g[:] = sd, sg
        return False

    return bt(), decisions


def load_split(tsv: Path) -> tuple[int, list[list[int]]]:
    rows = []
    with tsv.open() as f:
        f.readline()
        for line in f:
            clues = [int(t) - 1 for t in line.split("\t")[1].split(" ")]
            rows.append(clues)
    n = int(round(len(rows[0]) ** 0.5))
    return n, rows


# ----------------------------------------------------------------------------
# 3-COL instances (identical generator + seeds to phase_transition.py)
# ----------------------------------------------------------------------------

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "pt", Path(__file__).resolve().parent / "phase_transition.py")
pt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pt)


def stats_ms(times, decs):
    return {
        "n_instances": len(times),
        "all_solved": True,
        "time_ms_median": round(statistics.median(times) * 1e3, 3),
        "time_ms_p90": round(sorted(times)[int(0.9 * (len(times) - 1))] * 1e3, 3),
        "time_ms_max": round(max(times) * 1e3, 3),
        "decisions_median": int(statistics.median(decs)),
        "decisions_max": max(decs),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/classical_reference.json"))
    args = ap.parse_args()
    out = {"solver": "MRV backtracking with constraint propagation (pure Python)", "suites": {}}

    for split in ["data/sudoku4/test.tsv", "data/sudoku6/test.tsv",
                  "data/sudoku6_hard/test.tsv", "data/sudoku9/test.tsv",
                  "data/sudoku9_small/test.tsv"]:
        p = Path(split)
        if not p.exists():
            continue
        n, rows = load_split(p)
        peers = sudoku_geometry(n)
        times, decs = [], []
        for clues in rows:
            t0 = time.perf_counter()
            ok, d = solve_sudoku(clues, n, peers)
            times.append(time.perf_counter() - t0)
            decs.append(d)
            assert ok, f"unsolved instance in {split}"
        out["suites"][split] = stats_ms(times, decs)
        print(split, out["suites"][split], file=sys.stderr)

    for c in [3.0, 4.0, 4.5, 4.9]:
        rng = random.Random(42 + int(c * 10))
        times, decs = [], []
        made = 0
        while made < 60:
            m = int(round(c * pt.N_VERTICES / 2))
            pairs = [(u, v) for u in range(pt.N_VERTICES) for v in range(u + 1, pt.N_VERTICES)]
            edges = rng.sample(pairs, m)
            t0 = time.perf_counter()
            sols, decided = pt.sample_colorings(edges, 1, rng)
            dt = time.perf_counter() - t0
            if not decided or not sols:
                continue
            times.append(dt)
            decs.append(0)
            made += 1
        key = f"3col_n{pt.N_VERTICES}_c{c}"
        out["suites"][key] = {
            "n_instances": made,
            "time_ms_median": round(statistics.median(times) * 1e3, 3),
            "time_ms_p90": round(sorted(times)[int(0.9 * (len(times) - 1))] * 1e3, 3),
            "time_ms_max": round(max(times) * 1e3, 3),
        }
        print(key, out["suites"][key], file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
