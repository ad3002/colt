#!/usr/bin/env python3
"""Build a unique-solution Sudoku dataset for LDT (generic N×N).

Pipeline (deterministic given --seed):
  1. Generate a random complete solution by randomized backtracking.
  2. Dig holes in random order, keeping the solution UNIQUE at every step
     (uniqueness checked by a capped backtracking count, cap=2).
  3. Classify each puzzle as "requires guessing" iff a propagation-only solver
     (naked + hidden singles) cannot complete it.
  4. Optionally keep only guessing-required puzzles (--require-guessing), to
     approximate the paper's Sudoku-Extreme "backtracking-required" filter.
  5. Split unique puzzles into train/test and emit TSVs + MANIFEST.md5 + summary.

Token encoding (human-readable, 1-indexed): 0 = blank, 1..N = a given digit.
The model-side loader (ltd/datasets/sudoku.py) converts to 0-indexed candidates.

Output TSV columns:
  puzzle_id   clues   solution   n_clues   n_solutions

Cross-ref: paper §4 "Datasets" (Sudoku-Extreme). We GENERATE puzzles rather
than vendor the exact Tdoku/RRN corpus; that difference is logged in
DEVIATIONS.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path


# ----------------------------------------------------------------------------
# Geometry (mirrors ltd.datasets.sudoku, kept standalone so the builder has no
# torch dependency)
# ----------------------------------------------------------------------------


def factor_box(n: int) -> tuple[int, int]:
    best = (1, n)
    for br in range(1, int(n**0.5) + 1):
        if n % br == 0:
            best = (br, n // br)
    return best


@dataclass
class Geom:
    n: int
    br: int
    bc: int

    @property
    def C(self) -> int:
        return self.n * self.n

    def box_of(self, r: int, c: int) -> int:
        return (r // self.br) * (self.n // self.bc) + (c // self.bc)

    def peers(self) -> list[list[int]]:
        """For each cell, the list of peer cells (same row/col/box)."""
        n = self.n
        peers: list[list[int]] = []
        for i in range(self.C):
            r, c = i // n, i % n
            box = self.box_of(r, c)
            s = set()
            for cc in range(n):
                s.add(r * n + cc)
            for rr in range(n):
                s.add(rr * n + c)
            for j in range(self.C):
                if self.box_of(j // n, j % n) == box:
                    s.add(j)
            s.discard(i)
            peers.append(sorted(s))
        return peers


# ----------------------------------------------------------------------------
# Solvers (bitmask candidates; digit d uses bit (d-1))
# ----------------------------------------------------------------------------


def _units(geom: Geom) -> list[list[int]]:
    """All N*3 units (rows, cols, boxes) as cell-index lists."""
    n = geom.n
    units = []
    for r in range(n):
        units.append([r * n + c for c in range(n)])
    for c in range(n):
        units.append([r * n + c for r in range(n)])
    boxes: dict[int, list[int]] = {}
    for i in range(geom.C):
        boxes.setdefault(geom.box_of(i // n, i % n), []).append(i)
    units.extend(boxes.values())
    return units


def count_solutions(grid: list[int], geom: Geom, peers: list[list[int]], cap: int = 2) -> int:
    """Count solutions up to ``cap`` using MRV backtracking."""
    n = geom.n
    full = (1 << n) - 1
    cand = [full] * geom.C
    g = grid[:]
    # Initialize candidate masks from givens.
    for i in range(geom.C):
        if g[i]:
            cand[i] = 1 << (g[i] - 1)
    # Propagate givens.
    stack = [i for i in range(geom.C) if g[i]]
    while stack:
        i = stack.pop()
        bit = cand[i]
        for p in peers[i]:
            if cand[p] & bit:
                cand[p] &= ~bit
                if cand[p] == 0:
                    return 0
                if bin(cand[p]).count("1") == 1 and g[p] == 0:
                    g[p] = (cand[p].bit_length())
                    stack.append(p)

    count = 0

    def backtrack() -> bool:  # returns True to stop early (cap hit)
        nonlocal count
        # pick unfilled cell with fewest candidates (MRV)
        best = -1
        best_n = n + 1
        for i in range(geom.C):
            if g[i] == 0:
                cnt = bin(cand[i]).count("1")
                if cnt < best_n:
                    best_n, best = cnt, i
                    if cnt == 1:
                        break
        if best == -1:
            count += 1
            return count >= cap
        mask = cand[best]
        d = 1
        m = mask
        while m:
            if m & 1:
                # try digit d
                save_cand = cand[:]
                save_g = g[:]
                g[best] = d
                cand[best] = 1 << (d - 1)
                ok = True
                prop = [best]
                while prop and ok:
                    j = prop.pop()
                    bit = cand[j]
                    for p in peers[j]:
                        if cand[p] & bit:
                            cand[p] &= ~bit
                            if cand[p] == 0:
                                ok = False
                                break
                            if bin(cand[p]).count("1") == 1 and g[p] == 0:
                                g[p] = cand[p].bit_length()
                                prop.append(p)
                if ok and backtrack():
                    return True
                cand[:] = save_cand
                g[:] = save_g
            m >>= 1
            d += 1
        return False

    backtrack()
    return count


def solve_one(grid: list[int], geom: Geom, peers: list[list[int]], rng: random.Random) -> list[int] | None:
    """Return one complete solution (randomized digit order), or None."""
    n = geom.n
    full = (1 << n) - 1
    cand = [full] * geom.C
    g = grid[:]
    for i in range(geom.C):
        if g[i]:
            cand[i] = 1 << (g[i] - 1)

    def fill(i: int, d: int) -> bool:
        cand[i] = 1 << (d - 1)
        g[i] = d
        prop = [i]
        while prop:
            j = prop.pop()
            bit = cand[j]
            for p in peers[j]:
                if cand[p] & bit:
                    cand[p] &= ~bit
                    if cand[p] == 0:
                        return False
                    if bin(cand[p]).count("1") == 1 and g[p] == 0:
                        g[p] = cand[p].bit_length()
                        prop.append(p)
        return True

    def backtrack() -> bool:
        best, best_n = -1, n + 1
        for i in range(geom.C):
            if g[i] == 0:
                cnt = bin(cand[i]).count("1")
                if cnt < best_n:
                    best_n, best = cnt, i
                    if cnt == 1:
                        break
        if best == -1:
            return True
        digits = [d for d in range(1, n + 1) if cand[best] & (1 << (d - 1))]
        rng.shuffle(digits)
        for d in digits:
            save_cand, save_g = cand[:], g[:]
            if fill(best, d) and backtrack():
                return True
            cand[:], g[:] = save_cand, save_g
        return False

    return g if backtrack() else None


def solvable_by_singles(grid: list[int], geom: Geom, units: list[list[int]], peers: list[list[int]]) -> bool:
    """True if naked + hidden singles alone complete the grid (i.e. no guessing)."""
    n = geom.n
    full = (1 << n) - 1
    cand = [full] * geom.C
    g = grid[:]
    for i in range(geom.C):
        if g[i]:
            cand[i] = 1 << (g[i] - 1)
            for p in peers[i]:
                cand[p] &= ~cand[i]

    progress = True
    while progress:
        progress = False
        # naked singles
        for i in range(geom.C):
            if g[i] == 0 and bin(cand[i]).count("1") == 1:
                g[i] = cand[i].bit_length()
                for p in peers[i]:
                    if cand[p] & cand[i]:
                        cand[p] &= ~cand[i]
                        if cand[p] == 0 and g[p] == 0:
                            return False
                progress = True
        # hidden singles
        for unit in units:
            for d in range(n):
                bit = 1 << d
                spots = [i for i in unit if g[i] == 0 and (cand[i] & bit)]
                if len(spots) == 1:
                    i = spots[0]
                    g[i] = d + 1
                    cand[i] = bit
                    for p in peers[i]:
                        cand[p] &= ~bit
                    progress = True
    return all(g[i] != 0 for i in range(geom.C))


# ----------------------------------------------------------------------------
# Puzzle generation
# ----------------------------------------------------------------------------


@dataclass
class Puzzle:
    clues: list[int]
    solution: list[int]
    n_clues: int
    requires_guessing: bool


def make_puzzle(geom: Geom, peers: list[list[int]], units: list[list[int]], rng: random.Random,
                target_clues: int) -> Puzzle:
    """Generate one unique-solution puzzle by digging holes to ~target_clues."""
    solution = solve_one([0] * geom.C, geom, peers, rng)
    assert solution is not None
    clues = solution[:]
    order = list(range(geom.C))
    rng.shuffle(order)
    n_clues = geom.C
    for i in order:
        if n_clues <= target_clues:
            break
        saved = clues[i]
        clues[i] = 0
        # Removing must preserve uniqueness.
        if count_solutions(clues, geom, peers, cap=2) != 1:
            clues[i] = saved  # restore
        else:
            n_clues -= 1
    rg = not solvable_by_singles(clues, geom, units, peers)
    return Puzzle(clues=clues, solution=solution, n_clues=n_clues, requires_guessing=rg)


# ----------------------------------------------------------------------------
# Serialization
# ----------------------------------------------------------------------------


def to_str(tokens: list[int]) -> str:
    return " ".join(str(t) for t in tokens)


def write_tsv(puzzles: list[Puzzle], path: Path, start_id: int = 0) -> None:
    with path.open("w") as f:
        f.write("puzzle_id\tclues\tsolution\tn_clues\tn_solutions\n")
        for i, p in enumerate(puzzles):
            f.write(f"{start_id + i}\t{to_str(p.clues)}\t{to_str(p.solution)}\t{p.n_clues}\t1\n")


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=4, help="board size N (4, 6, 9, ...)")
    ap.add_argument("--box-rows", type=int, default=None)
    ap.add_argument("--box-cols", type=int, default=None)
    ap.add_argument("--num-puzzles", type=int, default=1200, help="unique puzzles to generate")
    ap.add_argument("--target-clues", type=int, default=None,
                    help="dig holes down to ~this many clues (default: aggressive minimal)")
    ap.add_argument("--require-guessing", action="store_true",
                    help="keep only puzzles a singles-only solver cannot finish")
    ap.add_argument("--split", type=float, default=0.85, help="train fraction on unique puzzles")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    br, bc = (args.box_rows, args.box_cols) if args.box_rows and args.box_cols else factor_box(args.n)
    if br * bc != args.n:
        raise SystemExit(f"box {br}x{bc} != n {args.n}")
    geom = Geom(args.n, br, bc)
    peers = geom.peers()
    units = _units(geom)
    target = args.target_clues if args.target_clues is not None else max(args.n, geom.C // 4)

    print(f"[gen] N={args.n} box={br}x{bc} target_clues~{target} num={args.num_puzzles} seed={args.seed}",
          file=sys.stderr)
    rng = random.Random(args.seed)
    seen: set[tuple[int, ...]] = set()
    puzzles: list[Puzzle] = []
    attempts = 0
    max_attempts = args.num_puzzles * 50
    while len(puzzles) < args.num_puzzles and attempts < max_attempts:
        attempts += 1
        p = make_puzzle(geom, peers, units, rng, target)
        if args.require_guessing and not p.requires_guessing:
            continue
        key = tuple(p.clues)
        if key in seen:
            continue
        seen.add(key)
        puzzles.append(p)
        if len(puzzles) % 200 == 0:
            print(f"      {len(puzzles)}/{args.num_puzzles} (attempts={attempts})", file=sys.stderr)
    if len(puzzles) < args.num_puzzles:
        print(f"[warn] only generated {len(puzzles)} (attempts exhausted)", file=sys.stderr)

    rng.shuffle(puzzles)
    cut = int(len(puzzles) * args.split)
    train, test = puzzles[:cut], puzzles[cut:]
    hard = [p for p in test if p.requires_guessing]
    print(f"[split] train={len(train)} test={len(test)} test_hard={len(hard)}", file=sys.stderr)

    args.out.mkdir(parents=True, exist_ok=True)
    write_tsv(train, args.out / "train.tsv")
    write_tsv(test, args.out / "test.tsv")
    write_tsv(hard, args.out / "slice_hard.tsv")

    n_clues = [p.n_clues for p in puzzles]
    summary = {
        "n": args.n, "box_rows": br, "box_cols": bc,
        "num_generated": len(puzzles), "target_clues": target,
        "require_guessing": args.require_guessing,
        "train_size": len(train), "test_size": len(test), "test_hard_size": len(hard),
        "frac_requires_guessing": sum(p.requires_guessing for p in puzzles) / max(len(puzzles), 1),
        "clues": {"min": min(n_clues), "max": max(n_clues), "mean": sum(n_clues) / len(n_clues)},
        "seed": args.seed, "split": args.split,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    manifest = args.out / "MANIFEST.md5"
    with manifest.open("w") as f:
        for name in sorted(["train.tsv", "test.tsv", "slice_hard.tsv", "summary.json"]):
            f.write(f"{md5_file(args.out / name)}  {name}\n")
    print(f"[done] outputs in {args.out}/", file=sys.stderr)
    print(manifest.read_text(), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
