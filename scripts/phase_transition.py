#!/usr/bin/env python3
"""Boundary probe: 3-coloring of random graphs across the phase transition.

Random graph 3-colorability has a sharp phase transition near average degree
c ~ 4.69; instances near the transition are the classical "really hard"
constraint problems (Cheeseman et al. 1991; Monasson et al. 1999). Sweeping
average degree therefore turns task hardness into a controlled dial with a
theoretically known hard point, which is exactly what is needed to map the
architecture's capability boundary rather than report isolated task scores.

Per density point: generate satisfiable instances (each with its OWN random
graph; the model receives per-sample relation matrices), sample up to K
solutions per instance for the alpha-target, train a fresh model at that
density, then evaluate per instance with both restart and DFS search plus the
H1 one-shot commit probe. If DFS pulls ahead of restart anywhere, it will be
near the transition, where solutions are few and long sound search chains are
genuinely required.

Training here is plain supervised steps over curriculum-revealed states (no
resident pool): per-sample graphs make pool recycling awkward, and the H1
result shows the propagator, not pool-induced search states, carries accuracy
at these scales. Documented as a probe-specific simplification.

Usage:
    python scripts/phase_transition.py --densities 3.0 4.0 4.5 4.9 5.4 \
        --out results/phase_transition.json
"""

from __future__ import annotations

import argparse
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
from colt.tasks.sudoku import REL_CLS_CELL, REL_CLS_CLS, TaskContext  # noqa: E402

N_VERTICES = 40
N_COLORS = 3
K_SOLUTIONS = 12
NODE_CAP = 200_000


# ----------------------------------------------------------------------------
# Exact 3-coloring with node budget
# ----------------------------------------------------------------------------


def sample_colorings(edges, k_want: int, rng: random.Random, node_cap: int = NODE_CAP):
    """Up to k_want distinct proper colorings via randomized MRV backtracking.
    Returns (list of colorings, decided) where decided=False means node budget
    hit before either finding a coloring or proving none exists."""
    adj = [set() for _ in range(N_VERTICES)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    col = [-1] * N_VERTICES
    out: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    nodes = 0

    def bt() -> bool:
        nonlocal nodes
        nodes += 1
        if nodes > node_cap:
            return True
        best, best_opts = -1, None
        for i in range(N_VERTICES):
            if col[i] >= 0:
                continue
            used = {col[j] for j in adj[i] if col[j] >= 0}
            opts = [c for c in range(N_COLORS) if c not in used]
            if best == -1 or len(opts) < len(best_opts):
                best, best_opts = i, opts
                if not opts:
                    return False
        if best == -1:
            t = tuple(col)
            if t not in seen:
                seen.add(t)
                out.append(t)
            return len(out) >= k_want
        opts = best_opts[:]
        rng.shuffle(opts)
        for c in opts:
            col[best] = c
            if bt():
                col[best] = -1
                return True
            col[best] = -1
        return False

    bt()
    return [list(s) for s in out], nodes <= node_cap


def gen_instances(avg_degree: float, n_instances: int, rng: random.Random):
    """Satisfiable instances at the given average degree. Reports gen stats."""
    m = int(round(avg_degree * N_VERTICES / 2))
    all_pairs = [(u, v) for u in range(N_VERTICES) for v in range(u + 1, N_VERTICES)]
    instances = []
    tried = unsat = undecided = 0
    while len(instances) < n_instances and tried < n_instances * 60:
        tried += 1
        edges = rng.sample(all_pairs, m)
        sols, decided = sample_colorings(edges, K_SOLUTIONS, rng)
        if not decided:
            undecided += 1
            continue
        if not sols:
            unsat += 1
            continue
        instances.append({"edges": edges, "solutions": sols})
    return instances, {"tried": tried, "unsat": unsat, "undecided": undecided}


# ----------------------------------------------------------------------------
# Tensors / contexts
# ----------------------------------------------------------------------------


def rel_and_coords(edges, device):
    S = N_VERTICES + 1
    rel = torch.zeros(S, S, dtype=torch.long)
    for u, v in edges:
        rel[u + 1, v + 1] = 1
        rel[v + 1, u + 1] = 1
    for i in range(1, S):
        rel[i, i] = 7
    rel[0, :] = REL_CLS_CELL
    rel[:, 0] = REL_CLS_CELL
    rel[0, 0] = REL_CLS_CLS
    deg = torch.zeros(N_VERTICES)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    feats = torch.stack([
        deg / max(deg.max().item(), 1.0),
        torch.arange(N_VERTICES) / N_VERTICES,
        torch.zeros(N_VERTICES),
        torch.zeros(N_VERTICES),
        torch.full((N_VERTICES,), 1.0 / N_VERTICES),
    ], dim=1)
    return rel.to(device), feats.to(device)


class _G:
    n_cand = N_COLORS
    n_cells = N_VERTICES


def instance_ctx(rel, feats):
    return TaskContext(name="3col", geom=_G(), rel_ids=rel, coord_feats=feats, n_cand=N_COLORS)


def proper(grids: torch.Tensor, edges) -> torch.Tensor:
    ok = (grids >= 0).all(dim=1)
    for u, v in edges:
        ok &= grids[:, u] != grids[:, v]
    return ok


# ----------------------------------------------------------------------------
# Per-density experiment
# ----------------------------------------------------------------------------


def run_density(d: float, args, device) -> dict:
    rng = random.Random(args.seed + int(d * 10))
    torch.manual_seed(args.seed + int(d * 10))
    t0 = time.time()
    inst, gen_stats = gen_instances(d, args.n_instances, rng)
    n_train = int(0.85 * len(inst))
    train_i, test_i = inst[:n_train], inst[n_train:]
    mean_sols = sum(len(p["solutions"]) for p in inst) / max(len(inst), 1)
    print(f"[d={d}] gen: {len(inst)} sat instances ({gen_stats}), "
          f"mean_sampled_solutions={mean_sols:.1f}, {time.time()-t0:.0f}s", file=sys.stderr)
    if len(test_i) < 10:
        return {"avg_degree": d, "error": "too few satisfiable instances", **gen_stats}

    rels = torch.stack([rel_and_coords(p["edges"], device)[0] for p in train_i])
    feats = torch.stack([rel_and_coords(p["edges"], device)[1] for p in train_i])
    sols = torch.zeros(len(train_i), K_SOLUTIONS, N_VERTICES, dtype=torch.long)
    smask = torch.zeros(len(train_i), K_SOLUTIONS, dtype=torch.bool)
    for i, p in enumerate(train_i):
        for k, s in enumerate(p["solutions"][:K_SOLUTIONS]):
            sols[i, k] = torch.tensor(s)
            smask[i, k] = True
    sols, smask = sols.to(device), smask.to(device)

    model = ColtModel(d_model=64, n_heads=4, n_layers=4, n_iters=8).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1, betas=(0.9, 0.95))
    gen = torch.Generator().manual_seed(args.seed)

    for step in range(args.steps):
        reveal = 0.9 * max(0.0, 1.0 - step / (0.7 * args.steps))
        idx = torch.randint(0, len(train_i), (args.batch,), generator=gen)
        bs = sols[idx]
        bm = smask[idx]
        # start from zero clues; reveal toward a random sampled solution
        pick = torch.randint(0, K_SOLUTIONS, (args.batch,), generator=gen).to(device)
        pick = torch.minimum(pick, bm.sum(1) - 1)
        target = bs[torch.arange(args.batch), pick]                      # (B, C)
        cl = torch.full((args.batch, N_VERTICES), -1, dtype=torch.long, device=device)
        q = (torch.rand(args.batch, 1, generator=gen) * reveal).to(device)
        rm = torch.rand(args.batch, N_VERTICES, generator=gen).to(device) < q
        cl = torch.where(rm, target, cl)
        x = L.clues_to_lattice(cl, N_COLORS)
        ctx = TaskContext(name="3col", geom=_G(), rel_ids=rels[idx], coord_feats=feats[idx],
                          n_cand=N_COLORS)
        out = model(x, ctx)
        y_hat, tib = L.abstraction_target(x, bs, bm)
        loss = compute_colt_loss(out, x, y_hat, tib)
        opt.zero_grad(set_to_none=True)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0:
            print(f"[d={d}] step {step} loss {loss.total.item():.4f}", file=sys.stderr)

    # Per-instance eval: restart vs dfs + H1 singleton probe.
    cfg = StepConfig(theta_elim=0.1, theta_cls=0.6, tau_decide=1.5)
    res = {"restart": 0, "dfs": 0}
    h1_single = h1_total = 0
    sgen = torch.Generator(device=device).manual_seed(args.seed + 1)
    for p in test_i:
        rel, ft = rel_and_coords(p["edges"], device)
        ctx = instance_ctx(rel, ft)
        x0 = L.clues_to_lattice(torch.full((1, N_VERTICES), -1, dtype=torch.long), N_COLORS).to(device)
        verify = lambda g, e=p["edges"]: proper(g.cpu(), e).to(g.device)
        with torch.no_grad():
            out = model(x0, ctx)
            b = out.final()[0]
            x1 = L.eliminate(x0, b, cfg.theta_elim)
            counts = L.count_candidates(x1)[0]
            h1_single += int((counts == 1).sum().item())
            h1_total += N_VERTICES
        for name, solver in [("restart", restart_solve), ("dfs", dfs_solve)]:
            r = solver(model, ctx, x0, n_chains=args.chains, max_rounds=args.rounds,
                       cfg=cfg, verify_fn=verify, generator=sgen,
                       policy_mode="learned")
            res[name] += int(r.solved.sum().item())

    n_test = len(test_i)
    return {
        "avg_degree": d,
        "n_train": len(train_i), "n_test": n_test,
        "gen": gen_stats,
        "mean_sampled_solutions": round(mean_sols, 2),
        "acc_restart": round(res["restart"] / n_test, 4),
        "acc_dfs": round(res["dfs"] / n_test, 4),
        "h1_singleton_rate": round(h1_single / h1_total, 4),
        "wall_s": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--densities", type=float, nargs="+", default=[3.0, 4.0, 4.5, 4.9, 5.4])
    ap.add_argument("--n-instances", type=int, default=700)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--chains", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("results/phase_transition.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    points = []
    for d in args.densities:
        points.append(run_density(d, args, device))
        print(json.dumps(points[-1]), file=sys.stderr)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"task": f"3-coloring G(n={N_VERTICES}, m=d*n/2), from scratch, K<={K_SOLUTIONS}",
             "transition_at": 4.69, "points": points, "seed": args.seed}, indent=2) + "\n")
    print(json.dumps(points, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
