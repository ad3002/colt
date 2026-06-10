#!/usr/bin/env python3
"""Graph-coloring CSP: a non-Sudoku task through the same CoLT interface.

Purpose (paper, external-validity section): show that nothing in CoLT is
Sudoku-specific. The constraint graph here is an irregular random graph rather
than a grid, the relation vocabulary collapses to {none, shares-an-edge}, and
puzzles have MULTIPLE valid solutions, exercising the K>1 path of the
abstraction operator for the first time.

Task: one fixed connected Erdos-Renyi graph G(n_vertices, p) that is 4-colorable;
a puzzle is a partial proper coloring carved from a random full coloring such
that the number of valid completions is in [1, max_solutions]. All completions
are enumerated exactly and stored, so training uses the exact alpha-target over
the full solution set Y. Evaluation counts a puzzle solved when a chain emits a
verified proper coloring (any valid completion counts; soundness via the exact
verifier as always).

Self-contained: builds the dataset, trains, evaluates, writes JSONs.

Usage:
    python scripts/coloring_experiment.py --out-dir runs/coloring --results results
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
from colt.solve import StepConfig, TrainPool, dfs_solve          # noqa: E402
from colt.tasks.sudoku import REL_CLS_CELL, REL_CLS_CLS, TaskContext  # noqa: E402

N_VERTICES = 20
N_COLORS = 4
EDGE_P = 0.25


# ----------------------------------------------------------------------------
# Graph + exact solver
# ----------------------------------------------------------------------------


def make_graph(rng: random.Random) -> list[tuple[int, int]]:
    """One connected G(N_VERTICES, EDGE_P) that admits a proper 4-coloring."""
    while True:
        edges = [(u, v) for u in range(N_VERTICES) for v in range(u + 1, N_VERTICES)
                 if rng.random() < EDGE_P]
        adj = [set() for _ in range(N_VERTICES)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        # connected?
        seen = {0}
        stack = [0]
        while stack:
            for w in adj[stack.pop()]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        if len(seen) != N_VERTICES:
            continue
        if color_completions(edges, [-1] * N_VERTICES, cap=1):
            return edges


def color_completions(edges: list[tuple[int, int]], clues: list[int], cap: int) -> list[list[int]]:
    """All proper colorings extending ``clues`` (-1 = blank), up to ``cap``."""
    adj = [set() for _ in range(N_VERTICES)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    col = clues[:]
    out: list[list[int]] = []

    def bt() -> bool:
        if len(out) >= cap:
            return True
        # most-constrained blank vertex
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
            out.append(col[:])
            return len(out) >= cap
        for c in best_opts:
            col[best] = c
            if bt():
                col[best] = -1
                return True
            col[best] = -1
        return False

    bt()
    return out


# ----------------------------------------------------------------------------
# CoLT plumbing for the coloring task
# ----------------------------------------------------------------------------


def coloring_context(edges: list[tuple[int, int]], device) -> TaskContext:
    S = N_VERTICES + 1
    rel = torch.zeros(S, S, dtype=torch.long)
    for u, v in edges:
        rel[u + 1, v + 1] = 1                       # shares-a-constraint bit
        rel[v + 1, u + 1] = 1
    for i in range(1, S):
        rel[i, i] = 7                               # self id, as in Sudoku
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

    class _G:                                       # minimal geometry shim for verify
        n_cand = N_COLORS
        n_cells = N_VERTICES

    ctx = TaskContext(name="coloring20", geom=_G(), rel_ids=rel, coord_feats=feats,
                      n_cand=N_COLORS)
    return ctx.to(device) if device.type == "cuda" else ctx


def proper_coloring(grids: torch.Tensor, edges: list[tuple[int, int]]) -> torch.Tensor:
    """(n, C) long, -1 = blank. True iff fully colored and no edge monochromatic."""
    ok = (grids >= 0).all(dim=1)
    for u, v in edges:
        ok &= grids[:, u] != grids[:, v]
    return ok


# ----------------------------------------------------------------------------
# Dataset build
# ----------------------------------------------------------------------------


def build_dataset(seed: int, n_puzzles: int, max_solutions: int = 16):
    rng = random.Random(seed)
    edges = make_graph(rng)
    puzzles = []
    seen = set()
    attempts = 0
    while len(puzzles) < n_puzzles and attempts < n_puzzles * 200:
        attempts += 1
        full = color_completions(edges, [-1] * N_VERTICES, cap=500)
        base = list(rng.choice(full))
        keep = rng.randint(int(0.30 * N_VERTICES), int(0.55 * N_VERTICES))
        idx = list(range(N_VERTICES))
        rng.shuffle(idx)
        clues = [-1] * N_VERTICES
        for i in idx[:keep]:
            clues[i] = base[i]
        key = tuple(clues)
        if key in seen:
            continue
        sols = color_completions(edges, clues, cap=max_solutions + 1)
        if not (1 <= len(sols) <= max_solutions):
            continue
        seen.add(key)
        puzzles.append({"clues": clues, "solutions": sols})
    rng.shuffle(puzzles)
    cut = int(0.85 * len(puzzles))
    return edges, puzzles[:cut], puzzles[cut:]


def to_tensors(puzzles, k_max: int):
    n = len(puzzles)
    clues = torch.tensor([p["clues"] for p in puzzles], dtype=torch.long)
    sols = torch.zeros(n, k_max, N_VERTICES, dtype=torch.long)
    mask = torch.zeros(n, k_max, dtype=torch.bool)
    for i, p in enumerate(puzzles):
        for k, s in enumerate(p["solutions"][:k_max]):
            sols[i, k] = torch.tensor(s)
            mask[i, k] = True
    return clues, sols, mask


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-puzzles", type=int, default=1200)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out-dir", type=Path, default=Path("runs/coloring"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[build] graph + {args.n_puzzles} puzzles ...", file=sys.stderr)
    edges, train_p, test_p = build_dataset(args.seed, args.n_puzzles)
    k_max = max(len(p["solutions"]) for p in train_p + test_p)
    tr_clues, tr_sols, tr_mask = to_tensors(train_p, k_max)
    te_clues, te_sols, te_mask = to_tensors(test_p, k_max)
    n_sols = [len(p["solutions"]) for p in train_p + test_p]
    print(f"[build] train={len(train_p)} test={len(test_p)} K_max={k_max} "
          f"mean_solutions={sum(n_sols)/len(n_sols):.2f}", file=sys.stderr)

    ctx = coloring_context(edges, device)
    model = ColtModel(d_model=64, n_heads=4, n_layers=4, n_iters=8).to(device)
    print(f"[model] {model.num_params():,} params (same code as Sudoku)", file=sys.stderr)

    sample_gen = torch.Generator().manual_seed(args.seed)
    solve_gen = torch.Generator(device=device).manual_seed(args.seed + 1)
    state = {"reveal": 0.9}

    def sampler(n: int):
        idx = torch.randint(0, len(train_p), (n,), generator=sample_gen)
        cl = tr_clues[idx].clone()
        sols = tr_sols[idx]
        mask = tr_mask[idx]
        reveal = state["reveal"]
        if reveal > 0.0:
            # reveal toward the FIRST stored solution per puzzle
            q = torch.rand(n, 1, generator=sample_gen) * reveal
            blank = cl < 0
            rm = blank & (torch.rand(n, N_VERTICES, generator=sample_gen) < q)
            cl = torch.where(rm, sols[:, 0, :], cl)
        return L.clues_to_lattice(cl, N_COLORS), sols, mask

    pool = TrainPool(sampler, batch_size=args.batch, tau_age=40, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1, betas=(0.9, 0.95))
    cfg = StepConfig(theta_elim=0.1, theta_cls=0.9, tau_decide=1.5)
    t0 = time.time()
    for step in range(args.steps):
        state["reveal"] = 0.9 * max(0.0, 1.0 - step / (0.7 * args.steps))
        x, sols, mask = pool.current()
        out = model(x, ctx)
        y_hat, tib = L.abstraction_target(x, sols, mask)
        loss = compute_colt_loss(out, x, y_hat, tib)
        opt.zero_grad(set_to_none=True)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        b, _c, p = out.final()
        pool.advance(b.detach(), out.mean_cand().detach(), p.detach(), cfg, solve_gen)
        if step % 300 == 0 or step == args.steps - 1:
            print(json.dumps({"step": step, "loss": round(loss.total.item(), 4),
                              "bce": round(loss.bce.item(), 4),
                              "wall_s": round(time.time() - t0, 1)}), file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "edges": edges}, args.out_dir / "final.pt")

    # Evaluation: any verified proper coloring counts; soundness by construction.
    eval_cfg = StepConfig(theta_elim=0.1, theta_cls=0.6, tau_decide=1.5)
    verify = lambda g: proper_coloring(g.cpu(), edges).to(g.device)
    solved = 0
    matched_known = 0
    suppressed = 0
    x0_all = L.clues_to_lattice(te_clues, N_COLORS).to(device)
    for s in range(0, len(test_p), 8):
        e = min(s + 8, len(test_p))
        r = dfs_solve(model, ctx, x0_all[s:e], n_chains=16, max_rounds=30,
                      cfg=eval_cfg, verify_fn=verify, generator=solve_gen,
                      policy_mode="learned")
        suppressed += r.unsound_outputs
        for j in range(e - s):
            if bool(r.solved[j]):
                solved += 1
                sol = r.solution[j].cpu()
                ks = te_mask[s + j].sum().item()
                if any(torch.equal(sol, te_sols[s + j, k]) for k in range(int(ks))):
                    matched_known += 1
    result = {
        "task": "graph-coloring G(20, 0.25), 4 colors, multi-solution (K<=16)",
        "n_test": len(test_p),
        "accuracy": round(solved / len(test_p), 4),
        "answered_in_enumerated_solution_set": round(matched_known / max(solved, 1), 4),
        "suppressed_unsound": suppressed,
        "k_max": k_max,
        "steps": args.steps,
        "seed": args.seed,
    }
    print(json.dumps(result, indent=2))
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "coloring20.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
