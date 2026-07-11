#!/usr/bin/env python3
"""E4: single-environment reconciliation of the anatomy/H2 probe discrepancy.

The paper's two first-pass poisoning probes (scripts/failure_anatomy.py and the
K=1 setting of scripts/h2_symmetry_frames.py) disagreed on 4/180 hard-slice
puzzles when run in different sessions/environments (42 vs 46 poisoned). This
script executes the pre-registered reconciliation (REVISION_EXPERIMENTS.md E4):
on one device, one torch build, one dtype, run BOTH probe code paths
back-to-back on the same checkpoint and

  1. assert bitwise-equal first-pass elimination masks,
  2. report the margin histogram |sigma(b) - theta_elim| over solution values,
  3. count puzzles whose poisoned status flips within +/-0.02 of theta,
  4. re-run the full solve (same settings as the eval JSON) and report the
     poisoned x solved contingency in this same environment,
  5. sweep theta_elim in {0.10, 0.05, 0.02} for the insensitivity check.

Usage:
    python scripts/reconcile_probes.py --checkpoint runs/colt6_seed42_cpu/final.pt \
        --split data/sudoku6_hard/test.tsv \
        --eval-json results/colt6hard_dfs_learned.json \
        --out results/reconcile_anatomy_h2.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from colt import lattice as L                                    # noqa: E402
from colt.eval import _build_model                               # noqa: E402
from colt.solve import StepConfig, dfs_solve, restart_solve      # noqa: E402
from colt.tasks.sudoku import (                                  # noqa: E402
    SudokuDataset, context_for, geometry_for, satisfies_constraints)


@torch.no_grad()
def mask_path_anatomy(model, ctx, clues, sols, n_cand, theta):
    """Verbatim code path of scripts/failure_anatomy.py (first-pass probe)."""
    x = L.clues_to_lattice(clues, n_cand)
    out = model(x, ctx)
    b, _, _ = out.final()
    x1 = L.eliminate(x, b, theta)
    sol_alive = x1.gather(2, sols.unsqueeze(-1)).squeeze(-1) > 0.5
    return x1, ~sol_alive.all(dim=1), b


@torch.no_grad()
def mask_path_h2(model, ctx, clues, sols, n_cand, theta):
    """Verbatim code path of scripts/h2_symmetry_frames.py poisoning_rate (K=1)."""
    x = L.clues_to_lattice(clues, n_cand)
    out = model(x, ctx)
    b = out.final()[0]
    x1 = L.eliminate(x, b, theta)
    sol_alive = x1.gather(2, sols.unsqueeze(-1)).squeeze(-1) > 0.5
    return x1, ~sol_alive.all(dim=1), b


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--eval-json", type=Path, required=True)
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
    theta = float(ckpt["config"]["solve"]["theta_elim"])

    # --- 1. both probe code paths, back to back, one environment -------------
    x1_a, poisoned_a, b_a = mask_path_anatomy(model, ctx, clues, sols, geom.n_cand, theta)
    x1_h, poisoned_h, b_h = mask_path_h2(model, ctx, clues, sols, geom.n_cand, theta)
    masks_equal = bool(torch.equal(x1_a, x1_h))
    logits_equal = bool(torch.equal(b_a, b_h))
    assert masks_equal, "probe code paths disagree on ONE device: real bug (E4 rule 2)"

    # --- 2. margin histogram over solution values ----------------------------
    p = torch.sigmoid(b_a)
    sol_p = p.gather(2, sols.unsqueeze(-1)).squeeze(-1)          # (N, C)
    margin = (sol_p - theta).abs()
    edges = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    hist = torch.histogram(margin.flatten().cpu().float(),
                           bins=torch.tensor(edges)).hist.long().tolist()
    min_margin_per_puzzle = margin.min(dim=1).values

    # --- 3. flip counts within +/-0.02 of theta -------------------------------
    def poisoned_at(th):
        x1 = L.eliminate(L.clues_to_lattice(clues, geom.n_cand), b_a, th)
        alive = x1.gather(2, sols.unsqueeze(-1)).squeeze(-1) > 0.5
        return ~alive.all(dim=1)

    flips_up = int((poisoned_at(theta + 0.02) != poisoned_a).sum())
    flips_dn = int((poisoned_at(theta - 0.02) != poisoned_a).sum())

    # --- 5. theta sweep (insensitivity check) ---------------------------------
    sweep = {f"{th:.2f}": int(poisoned_at(th).sum()) for th in (0.10, 0.05, 0.02)}

    # --- 4. full solve in this environment, contingency -----------------------
    ev = json.loads(args.eval_json.read_text())
    solver = dfs_solve if ev["search"] == "dfs" else restart_solve
    cfgs = ckpt["config"]
    step_cfg = StepConfig(theta_elim=theta, theta_cls=float(ev["theta_cls"]),
                          tau_decide=float(cfgs["solve"]["tau_decide"]))
    gen = torch.Generator(device=device).manual_seed(int(ev["seed"]))
    solved = torch.zeros(N, dtype=torch.bool)
    for s in range(0, N, 4):
        e = min(s + 4, N)
        x0 = L.clues_to_lattice(clues[s:e], geom.n_cand)
        r = solver(model, ctx, x0, n_chains=int(ev["n_chains"]), max_rounds=int(ev["max_rounds"]),
                   cfg=step_cfg, verify_fn=lambda g: satisfies_constraints(g, geom),
                   generator=gen, policy_mode=ev["policy"])
        solved[s:e] = r.solved.cpu()
    pois = poisoned_a.cpu()
    tab = {
        "solved_clean": int((solved & ~pois).sum()),
        "solved_poisoned": int((solved & pois).sum()),
        "failed_clean": int((~solved & ~pois).sum()),
        "failed_poisoned": int((~solved & pois).sum()),
    }

    result = {
        "protocol": "REVISION_EXPERIMENTS.md E4",
        "environment": {
            "torch": torch.__version__, "device": str(device),
            "dtype": "float32", "platform": platform.platform(),
            "threads": torch.get_num_threads(),
        },
        "checkpoint": str(args.checkpoint),
        "checkpoint_note": ("original runs/colt6_seed42/final.pt was lost with the pod; "
                            "this is a same-recipe, same-seed, same-data CPU retrain"),
        "split": str(args.split),
        "n": N,
        "probe_paths_bitwise_equal": masks_equal,
        "logits_bitwise_equal": logits_equal,
        "theta_elim": theta,
        "n_poisoned": int(pois.sum()),
        "poisoned_flips_at_theta_plus_0.02": flips_up,
        "poisoned_flips_at_theta_minus_0.02": flips_dn,
        "theta_sweep_n_poisoned": sweep,
        "solution_value_margin_hist": {"bin_edges": edges, "counts": hist},
        "min_solution_margin_p05_p50_p95": [
            round(float(min_margin_per_puzzle.quantile(q)), 4) for q in (0.05, 0.5, 0.95)],
        "n_solved": int(solved.sum()),
        "accuracy": round(float(solved.float().mean()), 4),
        "contingency": tab,
        "published_gpu_counts": {"anatomy_probe": 42, "h2_k1_probe": 46,
                                 "accuracy_eval": 0.7667, "accuracy_h2_k1": 0.7444},
        "eval_settings": {k: ev[k] for k in ("search", "policy", "n_chains", "max_rounds", "seed")},
    }
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
