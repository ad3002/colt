#!/usr/bin/env python3
"""Convert a CoLT/LTD Sudoku TSV split into the GRAM (ad3002/gram) TSV format.

GRAM's N-Queens loader expects:
    puzzle_id  input_tokens  target_tokens  n_solutions  all_completions_compact

Token encoding here: 0 = pad (unused), 1 = blank, 1+d = digit d (1-indexed), so
vocab_size = N + 2. Our Sudoku puzzles are unique-solution, so n_solutions = 1
and all_completions = the single solution — GRAM's accuracy_valid_solution then
equals exact-match, and any other emitted grid is a *wrong answer* (GRAM has no
abstention), which is exactly the soundness contrast the benchmark measures.

Usage:
    python scripts/convert_sudoku_to_gram.py --in data/sudoku6 --out data/gram_sudoku6
"""

from __future__ import annotations

import argparse
from pathlib import Path


def convert_split(src: Path, dst: Path) -> int:
    rows = 0
    with src.open() as f, dst.open("w") as g:
        header = f.readline().rstrip("\n").split("\t")
        assert header == ["puzzle_id", "clues", "solution", "n_clues", "n_solutions"], header
        g.write("puzzle_id\tinput_tokens\ttarget_tokens\tn_solutions\tall_completions_compact\n")
        for line in f:
            pid, clues, solution, _ncl, _nsol = line.rstrip("\n").split("\t")
            inp = " ".join("1" if t == "0" else str(int(t) + 1) for t in clues.split(" "))
            tgt = " ".join(str(int(t) + 1) for t in solution.split(" "))
            g.write(f"{pid}\t{inp}\t{tgt}\t1\t{tgt}\n")
            rows += 1
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", type=Path, required=True)
    ap.add_argument("--out", dest="dst", type=Path, required=True)
    args = ap.parse_args()
    args.dst.mkdir(parents=True, exist_ok=True)
    for name in ["train.tsv", "test.tsv"]:
        n = convert_split(args.src / name, args.dst / name)
        print(f"{name}: {n} rows")
    return 0


if __name__ == "__main__":
    main()
