#!/usr/bin/env python3
"""E10: predicted-null and frame-type factorial for the H2 symmetry cure.

Frame types on the canonical checkpoint, hard slice:
  identity  — the same deterministic pass repeated K times (predicted null),
  digit     — value-permutation frames (the published H2 mechanism),
  geometry  — cell-permutation frames from the Sudoku geometry group
              (row perms within bands x band perms x column perms within
              stacks x stack perms; no transpose at 6x6),
  both      — digit and geometry composed.

For each type: first-pass poisoning under K-frame union (K in {4, 8}) and,
for the decisive cells (geometry K=8, both K=8), full solve
accuracy at the frozen budget. Digit solve numbers come from
results/h2_colt6cpu_hard_union.json (same checkpoint, same environment).

Usage:
    python scripts/h2_factorial.py --checkpoint runs/colt6_seed42_cpu/final.pt \
        --split data/sudoku6_hard/test.tsv --out results/h2_factorial_cpu.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from colt import lattice as L                                    # noqa: E402
from colt.eval import _build_model                               # noqa: E402
from colt.model import ColtOutput                                # noqa: E402
from colt.solve import StepConfig, dfs_solve                     # noqa: E402
from colt.tasks.sudoku import (                                  # noqa: E402
    SudokuDataset, context_for, geometry_for, satisfies_constraints)


def geometry_perm(n, br, bc, rng) -> torch.Tensor:
    """Random cell permutation from the geometry group. Returns pi with
    new_cell[i] = old_cell[pi[i]]."""
    n_bands, n_stacks = n // br, n // bc
    bands = torch.randperm(n_bands, generator=rng)
    stacks = torch.randperm(n_stacks, generator=rng)
    row_map = torch.empty(n, dtype=torch.long)
    for b in range(n_bands):
        rows = torch.randperm(br, generator=rng)
        for j in range(br):
            row_map[b * br + j] = bands[b] * br + rows[j]
    col_map = torch.empty(n, dtype=torch.long)
    for s in range(n_stacks):
        cols = torch.randperm(bc, generator=rng)
        for j in range(bc):
            col_map[s * bc + j] = stacks[s] * bc + cols[j]
    pi = torch.empty(n * n, dtype=torch.long)
    for r in range(n):
        for c in range(n):
            pi[r * n + c] = row_map[r] * n + col_map[c]
    return pi


class Frames(torch.nn.Module):
    """K-frame union/mean wrapper over digit and/or geometry transforms."""

    def __init__(self, model, K, kind, agg, geom, gen):
        super().__init__()
        self.model, self.K, self.kind, self.agg, self.geom, self.gen = \
            model, K, kind, agg, geom, gen

    def eval(self):
        self.model.eval(); return self

    @torch.no_grad()
    def forward(self, x, ctx, n_iters=None) -> ColtOutput:
        B, C, V = x.shape
        n, br, bc = self.geom.n, self.geom.box_rows, self.geom.box_cols
        cand, cls, pol = [], [], []
        for k in range(self.K):
            xk = x
            vperm = torch.arange(V)
            cperm = torch.arange(C)
            if k > 0 and self.kind in ("digit", "both"):
                vperm = torch.randperm(V, generator=self.gen)
            if k > 0 and self.kind in ("geometry", "both"):
                cperm = geometry_perm(n, br, bc, self.gen)
            xk = x[:, cperm][:, :, vperm]
            out = self.model(xk, ctx)
            b, c, p = out.final()
            vinv = torch.empty_like(vperm); vinv[vperm] = torch.arange(V)
            cinv = torch.empty_like(cperm); cinv[cperm] = torch.arange(C)
            cand.append(b[:, cinv][:, :, vinv])
            cls.append(c); pol.append(p[:, cinv])
        cand = torch.stack(cand)
        cand = cand.amax(0) if self.agg == "union" else cand.mean(0)
        return ColtOutput(cand_logits=cand.unsqueeze(1),
                          cls_logits=torch.stack(cls).mean(0).unsqueeze(1),
                          policy_logits=torch.stack(pol).mean(0).unsqueeze(1))


@torch.no_grad()
def poisoning(model, ctx, clues, sols, n_cand, theta):
    x = L.clues_to_lattice(clues, n_cand)
    b = model(x, ctx).final()[0]
    x1 = L.eliminate(x, b, theta)
    alive = x1.gather(2, sols.unsqueeze(-1)).squeeze(-1) > 0.5
    return int((~alive.all(dim=1)).sum())


@torch.no_grad()
def solve_acc(model, ctx, n_cand, n_items, clues, cfg, verify, seed):
    gen = torch.Generator().manual_seed(seed)
    solved = 0
    for s in range(0, n_items, 4):
        e = min(s + 4, n_items)
        x0 = L.clues_to_lattice(clues[s:e], n_cand)
        r = dfs_solve(model, ctx, x0, n_chains=32, max_rounds=60, cfg=cfg,
                      verify_fn=verify, generator=gen, policy_mode="learned")
        solved += int(r.solved.sum())
    return solved


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base = _build_model(ckpt["config"]); base.load_state_dict(ckpt["model"]); base.eval()
    with args.split.open() as f:
        f.readline(); n_tokens = len(f.readline().split("\t")[1].split(" "))
    geom = geometry_for(int(round(n_tokens ** 0.5)))
    ctx = context_for(geom)
    ds = SudokuDataset(args.split, geom); ds.geom = geom
    clues, sols = ds.clues_tensor(), ds.solution_tensor()
    theta = float(ckpt["config"]["solve"]["theta_elim"])
    cfg = StepConfig(theta_elim=theta,
                     theta_cls=float(ckpt["config"]["eval"]["theta_cls"]),
                     tau_decide=float(ckpt["config"]["solve"]["tau_decide"]))
    verify = lambda g: satisfies_constraints(g, geom)

    res = {"checkpoint": str(args.checkpoint), "split": str(args.split),
           "n": len(ds), "cells": {}}
    for kind, K in itertools.product(["identity", "digit", "geometry", "both"], [4, 8]):
        gen = torch.Generator().manual_seed(args.seed)
        w = Frames(base, K, kind, "union", geom, gen)
        pois = poisoning(w, ctx, clues, sols, geom.n_cand, theta)
        cell = {"poisoned": pois}
        if K == 8 and kind in ("geometry", "both"):
            gen2 = torch.Generator().manual_seed(args.seed)
            w2 = Frames(base, K, kind, "union", geom, gen2)
            cell["solved"] = solve_acc(w2, ctx, geom.n_cand, len(ds), clues, cfg,
                                       verify, args.seed)
        res["cells"][f"{kind}_K{K}_union"] = cell
        print(f"{kind} K={K} union -> {cell}", file=sys.stderr)
    # geometry mean K=8 (secondary)
    gen = torch.Generator().manual_seed(args.seed)
    res["cells"]["geometry_K8_mean"] = {
        "poisoned": poisoning(Frames(base, 8, "geometry", "mean", geom, gen),
                              ctx, clues, sols, geom.n_cand, theta)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
