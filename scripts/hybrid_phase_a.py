#!/usr/bin/env python3
"""Phase A of the hybrid program: neural-guided EXACT search on random 3-COL.

Division of labor (the constructive conclusion of paper 1): the solver is a
classical exact engine (arc-consistency propagation + chronological DFS with
value exclusion), correct by construction; the network supplies only the one
thing classical solvers delegate to hand-coded heuristics: WHICH (vertex,
color) to commit next. With the heuristic ablated to random or MRV the system
IS the classical baseline, so learning starts from parity and can only be
judged by node counts.

Supervision is exact, not sampled: for a training state s and each alive pin
(v, c), the label is whether pin+propagate leaves the instance extendable,
decided by the exact solver. The network (the unchanged CoLT architecture;
the candidate head becomes the pin scorer) is trained with masked BCE on
these labels. No RL, no search-time learning.

Metrics: solve rate and decisions-to-solution (median / p90) under a node
budget, at the training size n=40 (densities 4.2 and 4.5, the dead zone of
paper 1's boundary curve) and zero-shot at n=100 and n=200. Pre-registered
success bar: the learned heuristic beats MRV by >= 2x median decisions at the
transfer sizes.

Usage:
    python scripts/hybrid_phase_a.py --out results/hybrid_phase_a.json
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

from colt.model import ColtModel                                 # noqa: E402
from colt.tasks.sudoku import REL_CLS_CELL, REL_CLS_CLS, TaskContext  # noqa: E402

N_COLORS = 3


# ----------------------------------------------------------------------------
# Exact engine: AC propagation + DFS with pluggable (vertex, color) heuristic
# ----------------------------------------------------------------------------


def propagate(masks: list[int], adj: list[list[int]], queue: list[int]) -> bool:
    """Arc-consistency cascade from `queue` of newly-singleton vertices.
    masks[v] = bitmask of alive colors. Returns False on empty-domain conflict."""
    while queue:
        v = queue.pop()
        m = masks[v]
        for u in adj[v]:
            if masks[u] & m:
                was_single = masks[u].bit_count() == 1
                masks[u] &= ~m
                if masks[u] == 0:
                    return False
                if not was_single and masks[u].bit_count() == 1:
                    queue.append(u)
    return True


def initial_state(n: int) -> list[int]:
    return [(1 << N_COLORS) - 1] * n


def solve_exact(adj, masks, node_cap: int) -> tuple[bool, int]:
    """Plain exact MRV solver used for labeling. Returns (extendable, nodes)."""
    n = len(masks)
    nodes = 0

    def bt(masks) -> int:  # 1 sat, 0 unsat, -1 budget
        nonlocal nodes
        nodes += 1
        if nodes > node_cap:
            return -1
        best, bc = -1, 4
        for v in range(n):
            c = masks[v].bit_count()
            if c > 1 and c < bc:
                bc, best = c, v
        if best == -1:
            return 1
        for c in range(N_COLORS):
            bit = 1 << c
            if not masks[best] & bit:
                continue
            m2 = masks[:]
            m2[best] = bit
            if propagate(m2, adj, [best]) and bt(m2) == 1:
                return 1
        return 0

    r = bt(masks[:])
    return r == 1, nodes


def guided_search(adj, heuristic, node_cap: int, rng: random.Random):
    """DFS with exact propagation; `heuristic(masks) -> (v, c)` picks the branch.
    Returns (solved, decisions)."""
    n = len(adj)
    decisions = 0

    def bt(masks) -> int:
        nonlocal decisions
        if all(m.bit_count() == 1 for m in masks):
            return 1
        if decisions > node_cap:
            return -1
        v, c_first = heuristic(masks)
        order = [c_first] + [c for c in range(N_COLORS) if c != c_first and masks[v] & (1 << c)]
        for c in order:
            bit = 1 << c
            if not masks[v] & bit:
                continue
            decisions += 1
            m2 = masks[:]
            m2[v] = bit
            if propagate(m2, adj, [v]):
                r = bt(m2)
                if r != 0:
                    return r
        return 0

    masks = initial_state(n)
    if not propagate(masks, adj, [v for v in range(n) if masks[v].bit_count() == 1]):
        return False, 0
    return bt(masks) == 1, decisions


# --- heuristics ---------------------------------------------------------------


def h_random(rng):
    def h(masks):
        opts = [v for v, m in enumerate(masks) if m.bit_count() > 1]
        v = rng.choice(opts)
        cols = [c for c in range(N_COLORS) if masks[v] & (1 << c)]
        return v, rng.choice(cols)
    return h


def h_mrv(rng, adj=None, deg_tiebreak=False):
    def h(masks):
        opts = [(masks[v].bit_count(),
                 -(len(adj[v]) if deg_tiebreak else 0),
                 rng.random(), v)
                for v, m in enumerate(masks) if m.bit_count() > 1]
        v = min(opts)[3]
        cols = [c for c in range(N_COLORS) if masks[v] & (1 << c)]
        return v, rng.choice(cols)
    return h


def _pin_scores(model, ctx, device, masks):
    n = len(masks)
    x = torch.zeros(1, n, N_COLORS, device=device)
    for v, m in enumerate(masks):
        for c in range(N_COLORS):
            if m & (1 << c):
                x[0, v, c] = 1.0
    out = model(x, ctx)
    b = out.final()[0][0]                                  # (n, V) pin logits
    alive = x[0] > 0.5
    multi = torch.tensor([m.bit_count() > 1 for m in masks], device=device)
    return b, alive, multi


def h_learned(model, ctx, device):
    """v1 (ablation arm): succeed-first everywhere. Picks the globally safest
    pin. Theoretically wrong for variable choice (anti fail-first): prefers
    unconstrained vertices and grows deep useless subtrees."""
    @torch.no_grad()
    def h(masks):
        b, alive, multi = _pin_scores(model, ctx, device, masks)
        score = b.masked_fill(~alive, float("-inf"))
        score = score.masked_fill(~multi.unsqueeze(-1), float("-inf"))
        flat = int(score.flatten().argmax().item())
        return flat // N_COLORS, flat % N_COLORS
    return h


def h_learned_ff(model, ctx, device):
    """v2: fail-first vertex, succeed-first value (Haralick-Elliott readout of
    the same scorer). Vertex = argmin of the summed pin-survival mass, a
    learned refinement of MRV's syntactic domain count (it estimates how many
    values are SOLVER-extendable, not merely propagation-alive); value =
    argmax survival at that vertex."""
    @torch.no_grad()
    def h(masks):
        b, alive, multi = _pin_scores(model, ctx, device, masks)
        p = torch.sigmoid(b).masked_fill(~alive, 0.0)
        dom = p.sum(dim=-1).masked_fill(~multi, float("inf"))   # learned domain size
        v = int(dom.argmin().item())
        c = int(p[v].argmax().item())
        return v, c
    return h


# ----------------------------------------------------------------------------
# Instances, contexts, training data with exact pin labels
# ----------------------------------------------------------------------------


def gen_graph(n, avg_degree, rng):
    m = int(round(avg_degree * n / 2))
    pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    while True:
        edges = rng.sample(pairs, m)
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        masks = initial_state(n)
        ok, _ = solve_exact(adj, masks, 100_000)
        if ok:
            return edges, adj


def ctx_for(edges, n, device):
    S = n + 1
    rel = torch.zeros(S, S, dtype=torch.long)
    for u, v in edges:
        rel[u + 1, v + 1] = 1
        rel[v + 1, u + 1] = 1
    for i in range(1, S):
        rel[i, i] = 7
    rel[0, :] = REL_CLS_CELL
    rel[:, 0] = REL_CLS_CELL
    rel[0, 0] = REL_CLS_CLS
    deg = torch.zeros(n)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1

    class _G:
        n_cand = N_COLORS
        n_cells = n
    feats = torch.stack([deg / max(deg.max().item(), 1.0),
                         torch.arange(n) / n,
                         torch.zeros(n), torch.zeros(n),
                         torch.full((n,), 1.0 / n)], dim=1)
    return TaskContext("3col", _G(), rel.to(device), feats.to(device), N_COLORS)


def labeled_states(adj, n, rng, n_states, label_cap):
    """Extendable states at random depth (pins from a random solution, then
    propagate) with exact extendability labels for every alive pin."""
    masks0 = initial_state(n)
    # one reference solution for depth sampling
    sol = None
    def find(masks):
        nonlocal sol
        if all(m.bit_count() == 1 for m in masks):
            sol = [m.bit_length() - 1 for m in masks]
            return True
        best = min((m.bit_count(), v) for v, m in enumerate(masks) if m.bit_count() > 1)[1]
        cols = [c for c in range(N_COLORS) if masks[best] & (1 << c)]
        rng.shuffle(cols)
        for c in cols:
            m2 = masks[:]
            m2[best] = 1 << c
            if propagate(m2, adj, [best]) and find(m2):
                return True
        return False
    find(masks0[:])

    out = []
    for _ in range(n_states):
        depth = rng.randint(0, n - 2)
        order = list(range(n))
        rng.shuffle(order)
        masks = initial_state(n)
        ok = True
        for v in order[:depth]:
            if masks[v].bit_count() == 1:
                continue
            masks[v] = 1 << sol[v]
            if not propagate(masks, adj, [v]):
                ok = False
                break
        if not ok:
            continue
        labels = {}
        for v in range(n):
            if masks[v].bit_count() <= 1:
                continue
            for c in range(N_COLORS):
                if not masks[v] & (1 << c):
                    continue
                m2 = masks[:]
                m2[v] = 1 << c
                if not propagate(m2, adj, [v]):
                    labels[(v, c)] = 0.0
                    continue
                ext, nodes = solve_exact(adj, m2, label_cap)
                if nodes <= label_cap:
                    labels[(v, c)] = 1.0 if ext else 0.0
        if labels:
            out.append((masks, labels))
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-train-graphs", type=int, default=60)
    ap.add_argument("--states-per-graph", type=int, default=40)
    ap.add_argument("--train-density", type=float, default=4.5)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--label-cap", type=int, default=20000)
    ap.add_argument("--node-budget", type=int, default=2000)
    ap.add_argument("--smoke", action="store_true", help="tiny eval grid for pipeline validation")
    ap.add_argument("--out", type=Path, default=Path("results/hybrid_phase_a.json"))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    N_TRAIN = 40

    print("[data] graphs + exact pin labels ...", file=sys.stderr)
    t0 = time.time()
    graphs, states = [], []
    for gi in range(args.n_train_graphs):
        edges, adj = gen_graph(N_TRAIN, args.train_density, rng)
        st = labeled_states(adj, N_TRAIN, rng, args.states_per_graph, args.label_cap)
        graphs.append((edges, adj))
        states.extend([(gi, m, lab) for m, lab in st])
        if (gi + 1) % 20 == 0:
            print(f"  {gi+1}/{args.n_train_graphs} graphs, {len(states)} states, "
                  f"{time.time()-t0:.0f}s", file=sys.stderr)
    n_pins = sum(len(lab) for _, _, lab in states)
    pos = sum(v for _, _, lab in states for v in lab.values())
    print(f"[data] {len(states)} states, {n_pins} labeled pins "
          f"({100*pos/max(n_pins,1):.1f}% extendable), {time.time()-t0:.0f}s", file=sys.stderr)

    ctxs = [ctx_for(e, N_TRAIN, device) for e, _ in graphs]
    rels = torch.stack([c.rel_ids for c in ctxs])
    feats = torch.stack([c.coord_feats for c in ctxs])

    # tensorize states
    X = torch.zeros(len(states), N_TRAIN, N_COLORS)
    LBL = torch.zeros(len(states), N_TRAIN, N_COLORS)
    MSK = torch.zeros(len(states), N_TRAIN, N_COLORS)
    GI = torch.zeros(len(states), dtype=torch.long)
    for i, (gi, masks, lab) in enumerate(states):
        GI[i] = gi
        for v, m in enumerate(masks):
            for c in range(N_COLORS):
                if m & (1 << c):
                    X[i, v, c] = 1.0
        for (v, c), y in lab.items():
            LBL[i, v, c] = y
            MSK[i, v, c] = 1.0
    X, LBL, MSK = X.to(device), LBL.to(device), MSK.to(device)

    print("[train] pin-scorer (masked BCE on exact labels)", file=sys.stderr)
    model = ColtModel(d_model=64, n_heads=4, n_layers=4, n_iters=8).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1, betas=(0.9, 0.95))
    gen = torch.Generator().manual_seed(args.seed)
    for step in range(args.steps):
        idx = torch.randint(0, len(states), (args.batch,), generator=gen)
        ctx = TaskContext("3col", None, rels[GI[idx]], feats[GI[idx]], N_COLORS)
        out = model(X[idx], ctx)
        b = out.cand_logits.mean(dim=1)                 # (B, C, V) over iterations
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            b, LBL[idx], weight=MSK[idx], reduction="sum") / MSK[idx].sum().clamp_min(1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0:
            print(f"  step {step} loss {loss.item():.4f}", file=sys.stderr)
    model.eval()

    # ---------------- evaluation ----------------
    out_doc = {"train": {"n_graphs": args.n_train_graphs, "n_states": len(states),
                         "n_pins": n_pins, "pct_extendable": round(100*pos/max(n_pins,1), 1),
                         "density": args.train_density},
               "node_budget": args.node_budget, "evals": []}

    def eval_setting(n, density, n_inst, budget):
        arms = {"random": [], "mrv": [], "mrv_deg": [], "learned": [], "learned_ff": []}
        solved = {k: 0 for k in arms}
        for i in range(n_inst):
            erng = random.Random(args.seed + 1000 + i + int(density * 100) + n)
            edges, adj = gen_graph(n, density, erng)
            ctx = ctx_for(edges, n, device)
            for name in arms:
                hrng = random.Random(args.seed + i)
                h = (h_random(hrng) if name == "random" else
                     h_mrv(hrng, adj) if name == "mrv" else
                     h_mrv(hrng, adj, deg_tiebreak=True) if name == "mrv_deg" else
                     h_learned(model, ctx, device) if name == "learned" else
                     h_learned_ff(model, ctx, device))
                ok, dec = guided_search(adj, h, budget, hrng)
                if ok:
                    solved[name] += 1
                    arms[name].append(dec)
        def stats(v):
            if not v:
                return {"median": None, "p90": None}
            s = sorted(v)
            return {"median": s[len(s)//2], "p90": s[int(0.9*(len(s)-1))]}
        return {"n": n, "density": density, "n_instances": n_inst, "budget": budget,
                **{f"{k}_solve_rate": round(solved[k]/n_inst, 3) for k in arms},
                **{f"{k}_decisions": stats(v) for k, v in arms.items()}}

    grid = ([(40, 4.5, 6, 200)] if args.smoke else
            [(40, 4.2, 100, args.node_budget),
             (40, 4.5, 100, args.node_budget),
             (100, 4.5, 40, args.node_budget * 3),
             (200, 4.5, 12, args.node_budget * 4)])
    for (n, dens, ninst, budget) in grid:
        print(f"[eval] n={n} d={dens}", file=sys.stderr)
        r = eval_setting(n, dens, ninst, budget)
        out_doc["evals"].append(r)
        print(json.dumps(r), file=sys.stderr)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out_doc, indent=2) + "\n")

    out_doc["wall_s"] = round(time.time() - t0, 1)
    args.out.write_text(json.dumps(out_doc, indent=2) + "\n")
    print(json.dumps(out_doc["evals"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
