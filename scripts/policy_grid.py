#!/usr/bin/env python3
"""E11: the learned policy's fair trial, at a density where search works.

Trains the standard 3-coloring model at one density (default c=3.0, the
regime where DFS beats restart on accuracy) and evaluates the full
{restart, dfs} x {random, mrv, learned} grid on the held-out instances,
plus first-pass singleton and poisoning rates. Run once per training seed.

Usage:
    python scripts/policy_grid.py --density 3.0 --seed 42 \
        --out results/policy_grid_c30_seed42.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from colt import lattice as L                                    # noqa: E402
from colt.losses import compute_colt_loss                        # noqa: E402
from colt.model import ColtModel                                 # noqa: E402
from colt.solve import StepConfig, dfs_solve, restart_solve      # noqa: E402
from colt.tasks.sudoku import TaskContext                        # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "pt", Path(__file__).resolve().parent / "phase_transition.py")
pt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pt)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--density", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-instances", type=int, default=700)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--chains", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    device = torch.device("cpu")
    d = args.density

    rng = random.Random(args.seed + int(d * 10))
    torch.manual_seed(args.seed + int(d * 10))
    t0 = time.time()
    inst, gen_stats = pt.gen_instances(d, args.n_instances, rng)
    n_train = int(0.85 * len(inst))
    train_i, test_i = inst[:n_train], inst[n_train:]
    print(f"[seed={args.seed}] {len(inst)} sat instances ({gen_stats})", file=sys.stderr)

    rels = torch.stack([pt.rel_and_coords(p["edges"], device)[0] for p in train_i])
    feats = torch.stack([pt.rel_and_coords(p["edges"], device)[1] for p in train_i])
    K = pt.K_SOLUTIONS
    sols = torch.zeros(len(train_i), K, pt.N_VERTICES, dtype=torch.long)
    smask = torch.zeros(len(train_i), K, dtype=torch.bool)
    for i, p in enumerate(train_i):
        for k, s in enumerate(p["solutions"][:K]):
            sols[i, k] = torch.tensor(s); smask[i, k] = True

    model = ColtModel(d_model=64, n_heads=4, n_layers=4, n_iters=8)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1, betas=(0.9, 0.95))
    gen = torch.Generator().manual_seed(args.seed)
    for step in range(args.steps):
        reveal = 0.9 * max(0.0, 1.0 - step / (0.7 * args.steps))
        idx = torch.randint(0, len(train_i), (args.batch,), generator=gen)
        bs, bm = sols[idx], smask[idx]
        pick = torch.randint(0, K, (args.batch,), generator=gen)
        pick = torch.minimum(pick, bm.sum(1) - 1)
        target = bs[torch.arange(args.batch), pick]
        cl = torch.full((args.batch, pt.N_VERTICES), -1, dtype=torch.long)
        q = torch.rand(args.batch, 1, generator=gen) * reveal
        rm = torch.rand(args.batch, pt.N_VERTICES, generator=gen) < q
        cl = torch.where(rm, target, cl)
        x = L.clues_to_lattice(cl, pt.N_COLORS)
        ctx = TaskContext(name="3col", geom=pt._G(), rel_ids=rels[idx],
                          coord_feats=feats[idx], n_cand=pt.N_COLORS)
        out = model(x, ctx)
        y_hat, tib = L.abstraction_target(x, bs, bm)
        loss = compute_colt_loss(out, x, y_hat, tib)
        opt.zero_grad(set_to_none=True)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0:
            print(f"[seed={args.seed}] step {step} loss {loss.total.item():.4f}", file=sys.stderr)

    cfg = StepConfig(theta_elim=0.1, theta_cls=0.6, tau_decide=1.5)
    grid = {f"{s}_{p}": 0 for s in ("restart", "dfs") for p in ("random", "mrv", "learned")}
    h1_single = 0
    poisoned = 0
    sgen = torch.Generator().manual_seed(args.seed + 1)
    for p in test_i:
        rel, ft = pt.rel_and_coords(p["edges"], device)
        ctx = pt.instance_ctx(rel, ft)
        x0 = L.clues_to_lattice(torch.full((1, pt.N_VERTICES), -1, dtype=torch.long), pt.N_COLORS)
        verify = lambda g, e=p["edges"]: pt.proper(g.cpu(), e).to(g.device)
        with torch.no_grad():
            b = model(x0, ctx).final()[0]
            x1 = L.eliminate(x0, b, cfg.theta_elim)
            h1_single += int((L.count_candidates(x1)[0] == 1).sum().item())
            # from scratch, poisoning = any vertex left with zero candidates or
            # all sampled solutions killed at some vertex
            alive_any = torch.zeros(pt.N_VERTICES, dtype=torch.bool)
            for s in p["solutions"]:
                sv = torch.tensor(s).unsqueeze(0)
                alive_any |= (x1.gather(2, sv.unsqueeze(-1)).squeeze(-1) > 0.5)[0]
            poisoned += int(not alive_any.all())
        for sname, solver in (("restart", restart_solve), ("dfs", dfs_solve)):
            for pol in ("random", "mrv", "learned"):
                r = solver(model, ctx, x0, n_chains=args.chains, max_rounds=args.rounds,
                           cfg=cfg, verify_fn=verify, generator=sgen, policy_mode=pol)
                grid[f"{sname}_{pol}"] += int(r.solved.sum().item())

    n = len(test_i)
    res = {"density": d, "seed": args.seed, "n_test": n, "gen": gen_stats,
           "grid_solved": grid,
           "grid_acc": {k: round(v / n, 4) for k, v in grid.items()},
           "h1_singleton_rate": round(h1_single / (n * pt.N_VERTICES), 4),
           "sampled_solutions_all_dead_rate": round(poisoned / n, 4),
           "wall_s": round(time.time() - t0, 1)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
