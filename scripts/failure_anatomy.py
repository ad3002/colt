#!/usr/bin/env python3
"""Failure anatomy: WHY do the resistant puzzles resist every search arm?

Hypothesis (paper §Results): on failed puzzles the propagator's very first
forward pass already eliminates a true-solution value (an unsound elimination),
after which no search strategy can reach the solution — eliminations are
deterministic per state, so every chain inherits the same poisoned lattice.
If true, accuracy differences between search arms vanish (as observed) and the
binding constraint is propagator calibration, not search.

For each test puzzle: run ONE forward pass on the clues-only lattice, apply
threshold elimination, and check whether any solution value was eliminated.
Report the contingency between {first-pass poisoned?} × {solved by full search?}
using an existing eval JSON for the solved-set.

Usage:
    python scripts/failure_anatomy.py --checkpoint runs/colt6_seed42/final.pt \
        --split data/sudoku6_hard/test.tsv \
        --eval-json results/colt6hard_dfs_learned.json --out results/anatomy6hard.json
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
    ap.add_argument("--eval-json", type=Path, required=True,
                    help="eval result on the same split (for the solved mask: re-run with same seed)")
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
    N = len(ds)

    # Reconstruct the solved mask by re-running the SAME eval as the JSON.
    ev = json.loads(args.eval_json.read_text())
    from colt.solve import StepConfig, dfs_solve, restart_solve  # noqa: E402
    from colt.tasks.sudoku import satisfies_constraints          # noqa: E402
    solver = dfs_solve if ev["search"] == "dfs" else restart_solve
    cfgs = ckpt["config"]
    step_cfg = StepConfig(theta_elim=float(cfgs["solve"]["theta_elim"]),
                          theta_cls=float(ev["theta_cls"]),
                          tau_decide=float(cfgs["solve"]["tau_decide"]))
    gen = torch.Generator(device=device).manual_seed(int(ev["seed"]))
    solved = torch.zeros(N, dtype=torch.bool)
    slot = 4
    for s in range(0, N, slot):
        e = min(s + slot, N)
        x0 = L.clues_to_lattice(clues[s:e], geom.n_cand)
        r = solver(model, ctx, x0, n_chains=int(ev["n_chains"]), max_rounds=int(ev["max_rounds"]),
                   cfg=step_cfg, verify_fn=lambda g: satisfies_constraints(g, geom), generator=gen,
                   policy_mode=ev["policy"])
        solved[s:e] = r.solved.cpu()

    # First-pass poisoning probe.
    x = L.clues_to_lattice(clues, geom.n_cand)
    out = model(x, ctx)
    b, _, _ = out.final()
    x1 = L.eliminate(x, b, args.theta_elim)
    sol_alive = x1.gather(2, sols.unsqueeze(-1)).squeeze(-1) > 0.5   # (N, C)
    poisoned = ~sol_alive.all(dim=1)                                  # (N,)
    poisoned = poisoned.cpu()

    tab = {
        "solved_clean": int((solved & ~poisoned).sum()),
        "solved_poisoned": int((solved & poisoned).sum()),
        "failed_clean": int((~solved & ~poisoned).sum()),
        "failed_poisoned": int((~solved & poisoned).sum()),
    }
    n_failed = int((~solved).sum())
    n_poisoned = int(poisoned.sum())
    result = {
        "checkpoint": str(args.checkpoint),
        "split": str(args.split),
        "n": N,
        "n_solved": int(solved.sum()),
        "n_failed": n_failed,
        "n_first_pass_poisoned": n_poisoned,
        "contingency": tab,
        "frac_failed_that_are_poisoned": round(tab["failed_poisoned"] / max(n_failed, 1), 4),
        "frac_solved_that_are_poisoned": round(tab["solved_poisoned"] / max(int(solved.sum()), 1), 4),
        "theta_elim": args.theta_elim,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
