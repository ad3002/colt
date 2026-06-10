#!/usr/bin/env python3
"""H1: is the model an iterative deducer or a one-shot amortized guesser?

For each test puzzle, run ONE forward pass on the clues-only lattice + threshold
elimination and measure how much of the final answer it already commits:

  * singleton_rate  — fraction of blank cells driven to a single candidate;
  * commit_correct  — of those, fraction equal to the true solution value;
  * cand_per_blank  — mean candidates remaining per blank cell after pass 1;
  * full_commit     — fraction of puzzles where pass 1 alone determines every
                      blank (the pure one-shot regime).

If singleton_rate ≈ 1 on solved puzzles, the "iterative deduction" loop is
cosmetic at test time: the candidate head amortizes the entire solution into
its first elimination pass, and search merely wraps a guesser (paper §anatomy).

Usage:
    python scripts/h1_commit_rate.py --checkpoint runs/colt6_seed42/final.pt \
        --split data/sudoku6/test.tsv --out results/h1_colt6.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from colt import lattice as L                                    # noqa: E402
from colt.eval import _build_model                               # noqa: E402
from colt.tasks.sudoku import SudokuDataset, context_for, geometry_for  # noqa: E402


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--theta-elim", type=float, default=0.1)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = _build_model(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    device = next(model.parameters()).device

    with args.split.open() as f:
        f.readline()
        n_tokens = len(f.readline().split("\t")[1].split(" "))
    geom = geometry_for(int(round(n_tokens ** 0.5)))
    ctx = context_for(geom).to(device)
    ds = SudokuDataset(args.split, geom)
    clues = ds.clues_tensor().to(device)
    sols = ds.solution_tensor().to(device)

    x = L.clues_to_lattice(clues, geom.n_cand)
    out = model(x, ctx)
    b, _, _ = out.final()
    x1 = L.eliminate(x, b, args.theta_elim)

    blank = clues < 0                                   # (N, C)
    counts = L.count_candidates(x1)                     # (N, C)
    singleton = (counts == 1) & blank
    decoded = x1.argmax(dim=-1)
    correct_single = singleton & (decoded == sols)

    n_blank = blank.sum().item()
    n_single = singleton.sum().item()
    result = {
        "checkpoint": str(args.checkpoint),
        "split": str(args.split),
        "n_puzzles": len(ds),
        "singleton_rate": round(n_single / max(n_blank, 1), 4),
        "commit_correct": round(correct_single.sum().item() / max(n_single, 1), 4),
        "cand_per_blank_after_pass1": round((counts[blank].float().mean()).item(), 3),
        "full_commit_frac": round(((singleton | ~blank).all(dim=1)).float().mean().item(), 4),
        "theta_elim": args.theta_elim,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
