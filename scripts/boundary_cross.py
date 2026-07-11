#!/usr/bin/env python3
"""Can the d=4.0 boundary be crossed, and by which axis?

The phase-transition sweep (scripts/phase_transition.py) found a capability
cliff between average degree 3.0 (dfs 36.2%) and 4.0 (0%), far below the
classical 3-COL hardness transition at c~4.69: instances trivial for a
backtracking solver are unlearnable at the swept scale. This script holds the
task fixed at the first dead point (d=4.0, same seeded instance set) and moves
one resource axis at a time:

  A  baseline      d_model 64, 4k steps, K=12 targets, eval 16 chains x 40 rounds
  B  infer-budget  arm-A checkpoint, eval 32 x 240 (12x search compute)
  C  train-scale   d_model 128, 12k steps, batch 128 (~10x train compute)
  D  supervision   K=48 sampled solutions (4x less target noise)
  E  no-learning   stub propagator (keeps everything, never flags conflict):
                   what pure verifier-gated random search buys at both budgets
  F  diagnostic    elimination precision/recall of A and C propagators on
                   mixed-depth states (is ANY sound signal being learned?)

Interpretation rule, fixed in advance: if B, C, or D lifts accuracy well off
zero, the boundary is resource-soft along that axis and we report the
exchange rate. If all stay at zero while F shows no usable elimination signal
at shallow depth, the boundary is paradigm-level for from-scratch CSPs: with
no clues there are no forced inferences for a deduction-trained propagator to
amortize, and progress requires a different objective (value-guided guessing),
not more of the same resources.

Usage:
    python scripts/boundary_cross.py --out results/boundary_cross.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from colt import lattice as L                                    # noqa: E402
from colt.losses import compute_colt_loss                        # noqa: E402
from colt.model import ColtModel, ColtOutput                     # noqa: E402
from colt.solve import StepConfig, dfs_solve, restart_solve      # noqa: E402
from colt.tasks.sudoku import REL_CLS_CELL, REL_CLS_CLS, TaskContext  # noqa: E402

N_VERTICES = 40
N_COLORS = 3
NODE_CAP = 200_000
DENSITY = 4.0


# --- generation (as in phase_transition.py) ---------------------------------


def sample_colorings(edges, k_want, rng, node_cap=NODE_CAP):
    adj = [set() for _ in range(N_VERTICES)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    col = [-1] * N_VERTICES
    out, seen = [], set()
    nodes = 0

    def bt():
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


def gen_instances(avg_degree, n_instances, k_sols, rng):
    m = int(round(avg_degree * N_VERTICES / 2))
    pairs = [(u, v) for u in range(N_VERTICES) for v in range(u + 1, N_VERTICES)]
    inst, tried = [], 0
    while len(inst) < n_instances and tried < n_instances * 60:
        tried += 1
        edges = rng.sample(pairs, m)
        sols, decided = sample_colorings(edges, k_sols, rng)
        if decided and sols:
            inst.append({"edges": edges, "solutions": sols})
    return inst


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
    feats = torch.stack([deg / max(deg.max().item(), 1.0),
                         torch.arange(N_VERTICES) / N_VERTICES,
                         torch.zeros(N_VERTICES), torch.zeros(N_VERTICES),
                         torch.full((N_VERTICES,), 1.0 / N_VERTICES)], dim=1)
    return rel.to(device), feats.to(device)


class _G:
    n_cand = N_COLORS
    n_cells = N_VERTICES


def proper(grids, edges):
    ok = (grids >= 0).all(dim=1)
    for u, v in edges:
        ok &= grids[:, u] != grids[:, v]
    return ok


# --- training ----------------------------------------------------------------


def train_model(train_i, k_max, d_model, steps, batch, seed, device):
    rels = torch.stack([rel_and_coords(p["edges"], device)[0] for p in train_i])
    feats = torch.stack([rel_and_coords(p["edges"], device)[1] for p in train_i])
    sols = torch.zeros(len(train_i), k_max, N_VERTICES, dtype=torch.long)
    smask = torch.zeros(len(train_i), k_max, dtype=torch.bool)
    for i, p in enumerate(train_i):
        for k, s in enumerate(p["solutions"][:k_max]):
            sols[i, k] = torch.tensor(s)
            smask[i, k] = True
    sols, smask = sols.to(device), smask.to(device)

    model = ColtModel(d_model=d_model, n_heads=4, n_layers=4, n_iters=8).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1, betas=(0.9, 0.95))
    gen = torch.Generator().manual_seed(seed)
    last = 0.0
    for step in range(steps):
        reveal = 0.9 * max(0.0, 1.0 - step / (0.7 * steps))
        idx = torch.randint(0, len(train_i), (batch,), generator=gen)
        bs, bm = sols[idx], smask[idx]
        pick = torch.randint(0, k_max, (batch,), generator=gen).to(device)
        pick = torch.minimum(pick, bm.sum(1) - 1)
        target = bs[torch.arange(batch), pick]
        cl = torch.full((batch, N_VERTICES), -1, dtype=torch.long, device=device)
        q = (torch.rand(batch, 1, generator=gen) * reveal).to(device)
        rm = torch.rand(batch, N_VERTICES, generator=gen).to(device) < q
        cl = torch.where(rm, target, cl)
        x = L.clues_to_lattice(cl, N_COLORS)
        ctx = TaskContext("3col", _G(), rels[idx], feats[idx], N_COLORS)
        out = model(x, ctx)
        y_hat, tib = L.abstraction_target(x, bs, bm)
        loss = compute_colt_loss(out, x, y_hat, tib)
        opt.zero_grad(set_to_none=True)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last = loss.total.item()
        if step % 2000 == 0:
            print(f"  step {step} loss {last:.4f}", file=sys.stderr)
    return model, last


class StubPropagator(nn.Module):
    """No learning: keeps every candidate, never flags conflict, flat policy.
    Search degenerates to verifier-gated random assignment with backjumping."""

    def forward(self, x, ctx, n_iters=None):
        B, C, V = x.shape
        return ColtOutput(torch.full((B, 1, C, V), 10.0, device=x.device),
                          torch.full((B, 1), -10.0, device=x.device),
                          torch.zeros(B, 1, C, device=x.device))

    def eval(self):
        return self


# --- evaluation ----------------------------------------------------------------


@torch.no_grad()
def evaluate(model, test_i, chains, rounds, seed, device, theta_cls: float = 0.6):
    cfg = StepConfig(theta_elim=0.1, theta_cls=theta_cls, tau_decide=1.5)
    gen = torch.Generator(device=device).manual_seed(seed)
    res = {"restart": 0, "dfs": 0}
    for p in test_i:
        rel, ft = rel_and_coords(p["edges"], device)
        ctx = TaskContext("3col", _G(), rel, ft, N_COLORS)
        x0 = L.clues_to_lattice(torch.full((1, N_VERTICES), -1, dtype=torch.long), N_COLORS).to(device)
        verify = lambda g, e=p["edges"]: proper(g.cpu(), e).to(g.device)
        for name, solver in [("restart", restart_solve), ("dfs", dfs_solve)]:
            r = solver(model, ctx, x0, n_chains=chains, max_rounds=rounds, cfg=cfg,
                       verify_fn=verify, generator=gen, policy_mode="learned")
            res[name] += int(r.solved.sum().item())
    n = len(test_i)
    return {k: round(v / n, 4) for k, v in res.items()}


@torch.no_grad()
def elim_profile(model, test_i, k_max, seed, device, n_states=256):
    """Elimination precision/recall on mixed-depth states (vs sampled-solution union)."""
    gen = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    tp = fp = fn = 0
    shallow_kills = 0
    for _ in range(n_states):
        p = rng.choice(test_i)
        rel, ft = rel_and_coords(p["edges"], device)
        ctx = TaskContext("3col", _G(), rel, ft, N_COLORS)
        target = torch.tensor(rng.choice(p["solutions"]), device=device)
        q = rng.random() * 0.9
        cl = torch.full((N_VERTICES,), -1, dtype=torch.long, device=device)
        rm = torch.rand(N_VERTICES, generator=gen).to(device) < q
        cl = torch.where(rm, target, cl)
        x = L.clues_to_lattice(cl.unsqueeze(0), N_COLORS)
        out = model(x, ctx)
        b = out.final()[0]
        x1 = L.eliminate(x, b, 0.1)
        # ground truth: union of stored solutions consistent with the state
        sols = torch.tensor(p["solutions"], device=device)             # (K, C)
        alive_at = x[0].unsqueeze(0).expand(sols.shape[0], -1, -1).gather(
            2, sols.unsqueeze(-1)).squeeze(-1)                          # (K, C)
        cons = (alive_at > 0.5).all(dim=1)
        if not cons.any():
            continue
        union = torch.zeros(N_VERTICES, N_COLORS, device=device)
        for k in range(sols.shape[0]):
            if cons[k]:
                union[torch.arange(N_VERTICES), sols[k]] = 1.0
        alive0 = x[0] > 0.5
        should_kill = alive0 & (union < 0.5)        # not in any surviving sampled solution
        did_kill = alive0 & (x1[0] < 0.5)
        tp += int((did_kill & should_kill).sum())
        fp += int((did_kill & ~should_kill).sum())
        fn += int((~did_kill & should_kill).sum())
        if q < 0.2:
            shallow_kills += int(did_kill.sum())
    return {"elim_precision": round(tp / max(tp + fp, 1), 4),
            "elim_recall": round(tp / max(tp + fn, 1), 4),
            "shallow_state_kills": shallow_kills}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-instances", type=int, default=700)
    ap.add_argument("--out", type=Path, default=Path("results/boundary_cross.json"))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rng = random.Random(args.seed + 40)
    print("[gen] d=4.0 instances (K=48 sampled targets, reused for all arms)", file=sys.stderr)
    inst = gen_instances(DENSITY, args.n_instances, 48, rng)
    n_train = int(0.85 * len(inst))
    train_i, test_i = inst[:n_train], inst[n_train:]
    print(f"[gen] {len(train_i)} train / {len(test_i)} test", file=sys.stderr)

    out = {"density": DENSITY, "n_train": len(train_i), "n_test": len(test_i), "arms": {}}

    def record(name, val):
        out["arms"][name] = val
        print(json.dumps({name: val}), file=sys.stderr)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2) + "\n")

    t0 = time.time()
    print("[A] baseline d64/4k/K12", file=sys.stderr)
    model_a, loss_a = train_model(train_i, 12, 64, 4000, 64, args.seed, device)
    record("A_baseline_16x40", {**evaluate(model_a, test_i, 16, 40, args.seed, device),
                                "final_loss": round(loss_a, 4)})
    print("[B] same checkpoint, budget 32x240", file=sys.stderr)
    record("B_budget_32x240", evaluate(model_a, test_i, 32, 240, args.seed, device))
    print("[G] same checkpoint, CLS-conflict disabled (suspect: K-sampled bottom labels", file=sys.stderr)
    print("    miscalibrate the value head pessimistic, killing chains early)", file=sys.stderr)
    record("G_noCLS_16x40", evaluate(model_a, test_i, 16, 40, args.seed, device, theta_cls=1.01))
    record("G_noCLS_32x240", evaluate(model_a, test_i, 32, 240, args.seed, device, theta_cls=1.01))
    record("F_diag_A", elim_profile(model_a, test_i, 12, args.seed, device))

    print("[C] scale d128/12k/batch128", file=sys.stderr)
    model_c, loss_c = train_model(train_i, 12, 128, 12000, 128, args.seed, device)
    record("C_scale_16x40", {**evaluate(model_c, test_i, 16, 40, args.seed, device),
                             "final_loss": round(loss_c, 4)})
    record("C_scale_32x240", evaluate(model_c, test_i, 32, 240, args.seed, device))
    record("F_diag_C", elim_profile(model_c, test_i, 12, args.seed, device))

    print("[D] supervision K=48", file=sys.stderr)
    model_d, loss_d = train_model(train_i, 48, 64, 4000, 64, args.seed, device)
    record("D_K48_16x40", {**evaluate(model_d, test_i, 16, 40, args.seed, device),
                           "final_loss": round(loss_d, 4)})
    record("D_K48_32x240", evaluate(model_d, test_i, 32, 240, args.seed, device))

    print("[E] stub (no learning)", file=sys.stderr)
    stub = StubPropagator()
    record("E_stub_16x40", evaluate(stub, test_i, 16, 40, args.seed, device))
    record("E_stub_32x240", evaluate(stub, test_i, 32, 240, args.seed, device))

    out["wall_s"] = round(time.time() - t0, 1)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
