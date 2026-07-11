"""CoLT evaluation — soundness + accuracy with pluggable search and policy.

The ablation interface: ``--search {restart,dfs} --policy {random,mrv,learned}``
gives the full 2×3 matrix from ONE checkpoint (search and policy are
inference-side). ``--search restart --policy random`` reproduces LDT inference
exactly and is the control arm.

Usage:
    python -m colt.eval --checkpoint runs/colt6/final.pt \
        --split data/sudoku6/test.tsv --search dfs --policy learned \
        --n-chains 32 --max-rounds 60 --output results/colt6_dfs_learned.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from . import lattice as L
from .model import ColtModel
from .solve import POLICY_MODES, StepConfig, dfs_solve, restart_solve
from .tasks.sudoku import SudokuDataset, context_for, geometry_for, satisfies_constraints


def _build_model(cfg: dict) -> ColtModel:
    m = cfg["model"]
    return ColtModel(
        d_model=int(m["d_model"]), n_heads=int(m["n_heads"]),
        n_layers=int(m["n_layers"]), n_iters=int(m["n_iters"]),
        ff_mult=int(m.get("ff_mult", 4)), v_max=int(m.get("v_max", 9)),
        use_rel_bias=bool(m.get("use_rel_bias", True)),
        use_coord_mlp=bool(m.get("use_coord_mlp", True)),
        pos_table_size=int(m.get("pos_table_size", 0)),
    )


def select_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def _percentiles(values: list[int]) -> dict:
    if not values:
        return {"p50": None, "p75": None, "p90": None, "p95": None, "max": None}
    t = torch.tensor(sorted(values), dtype=torch.float32)
    q = torch.tensor([0.50, 0.75, 0.90, 0.95])
    p = torch.quantile(t, q).tolist()
    return {"p50": p[0], "p75": p[1], "p90": p[2], "p95": p[3], "max": float(t.max())}


@torch.no_grad()
def evaluate(model, ctx, dataset: SudokuDataset, cfg: StepConfig, search: str, policy: str,
             n_chains: int, max_rounds: int, slot_batch: int,
             device: torch.device, generator: torch.Generator) -> dict:
    clues = dataset.clues_tensor()
    sols = dataset.solution_tensor()
    total = len(dataset)
    n_answered = 0
    n_correct = 0
    suppressed = 0
    restarts = 0
    backjumps = 0
    rounds_solved: list[int] = []

    solver = dfs_solve if search == "dfs" else restart_solve
    verify = lambda g: satisfies_constraints(g, ctx.geom)
    for start in range(0, total, slot_batch):
        end = min(start + slot_batch, total)
        x0 = L.clues_to_lattice(clues[start:end], ctx.n_cand).to(device)
        res = solver(model, ctx, x0, n_chains=n_chains, max_rounds=max_rounds,
                     cfg=cfg, verify_fn=verify, generator=generator, policy_mode=policy)
        suppressed += res.unsound_outputs
        restarts += res.restarts
        backjumps += res.backjumps
        solved = res.solved.cpu()
        n_answered += int(solved.sum().item())
        for j in range(end - start):
            if bool(solved[j]):
                rounds_solved.append(int(res.rounds_used[j].item()))
                if torch.equal(res.solution[j].cpu(), sols[start + j]):
                    n_correct += 1

    accuracy = n_answered / max(total, 1)
    answered_correct = n_correct / max(n_answered, 1) if n_answered else 1.0
    return {
        "n_examples": total,
        "n_answered": n_answered,
        "accuracy": accuracy,
        "abstain_rate": 1.0 - accuracy,
        "answered_match_known_solution": answered_correct,
        "soundness": 1.0 if answered_correct == 1.0 else answered_correct,
        "suppressed_unsound": suppressed,
        "restarts": restarts,
        "backjumps": backjumps,
        "search": search,
        "policy": policy,
        "n_chains": n_chains,
        "max_rounds": max_rounds,
        "rounds": _percentiles(rounds_solved),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--search", choices=["restart", "dfs"], default="dfs")
    ap.add_argument("--policy", choices=list(POLICY_MODES), default="learned")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-chains", type=int, default=32)
    ap.add_argument("--max-rounds", type=int, default=60)
    ap.add_argument("--slot-batch", type=int, default=4)
    ap.add_argument("--theta-cls", type=float, default=None)
    ap.add_argument("--theta-elim", type=float, default=None,
                    help="override the elimination threshold (poisoning/calibration sweeps)")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N puzzles (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    device = select_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = _build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    # board size from the split itself (the checkpoint is size-agnostic)
    with args.split.open() as f:
        f.readline()
        n_tokens = len(f.readline().split("\t")[1].split(" "))
    geom = geometry_for(int(round(n_tokens ** 0.5)))
    ctx = context_for(geom).to(device)
    ds = SudokuDataset(args.split, geom)
    if args.limit > 0:
        ds._items = ds._items[: args.limit]

    theta_cls = args.theta_cls if args.theta_cls is not None else float(cfg["eval"]["theta_cls"])
    theta_elim = args.theta_elim if args.theta_elim is not None else float(cfg["solve"]["theta_elim"])
    step_cfg = StepConfig(theta_elim=theta_elim,
                          theta_cls=theta_cls, tau_decide=float(cfg["solve"]["tau_decide"]))
    gen = torch.Generator(device=device).manual_seed(args.seed)

    print(f"[eval] {args.split.name}: {len(ds)} puzzles, search={args.search}, policy={args.policy}, "
          f"chains={args.n_chains}, rounds={args.max_rounds}", file=sys.stderr)
    result = evaluate(model, ctx, ds, step_cfg, args.search, args.policy,
                      args.n_chains, args.max_rounds, args.slot_batch, device, gen)
    result.update({"split": str(args.split), "checkpoint": str(args.checkpoint),
                   "seed": args.seed, "theta_cls": theta_cls, "theta_elim": theta_elim})
    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
