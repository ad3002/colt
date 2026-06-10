#!/usr/bin/env python3
"""H2: does per-step symmetry ensembling de-poison the propagator at test time?

Wraps a trained model so that every forward pass runs K digit-permutation
frames (permute the value axis of the lattice, forward, invert the permutation
on the logits) and aggregates:

  * ``mean``  — average candidate logits across frames (ensemble);
  * ``union`` — elementwise max (keep a candidate if ANY frame keeps it):
                maximally conservative elimination, the direct antidote to
                value-specific poisoning.

For each (frames, aggregation) setting we report first-pass poisoning on the
split and full DFS solve accuracy at the frozen budget. Predictions
(paper §anatomy): an augmentation-trained model (permutation-equivariant by
training) gains little because it is already calibrated; an UNaugmented model
should see poisoning drop and accuracy rise with K — test-time decorrelation
without retraining. Either way the frames *localize* poison (frame
disagreement), the prerequisite for elimination-revival search (CoLT-2).

Usage:
    python scripts/h2_symmetry_frames.py --checkpoint runs/colt6_seed42/final.pt \
        --split data/sudoku6_hard/test.tsv --frames 1 4 8 --agg union \
        --out results/h2_colt6hard.json
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
from colt.model import ColtOutput                                # noqa: E402
from colt.solve import StepConfig, dfs_solve                     # noqa: E402
from colt.tasks.sudoku import SudokuDataset, context_for, geometry_for, satisfies_constraints  # noqa: E402


class SymmetryFrames(torch.nn.Module):
    """K digit-permutation frames per forward; aggregate inverse-mapped logits."""

    def __init__(self, model, n_frames: int, agg: str, generator: torch.Generator):
        super().__init__()
        self.model = model
        self.K = n_frames
        self.agg = agg
        self.gen = generator

    def eval(self):
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor, ctx, n_iters=None) -> ColtOutput:
        B, C, V = x.shape
        cand_frames, cls_frames, pol_frames = [], [], []
        for k in range(self.K):
            if k == 0:
                perm = torch.arange(V, device=x.device)       # identity frame first
            else:
                perm = torch.randperm(V, generator=self.gen).to(x.device)
            inv = torch.empty_like(perm)
            inv[perm] = torch.arange(V, device=x.device)
            out = self.model(x[:, :, perm], ctx)
            b, c, p = out.final()
            cand_frames.append(b[:, :, inv])                   # back to original labels
            cls_frames.append(c)
            pol_frames.append(p)
        cand = torch.stack(cand_frames)                        # (K, B, C, V)
        cand = cand.amax(dim=0) if self.agg == "union" else cand.mean(dim=0)
        cls = torch.stack(cls_frames).mean(dim=0)
        pol = torch.stack(pol_frames).mean(dim=0)
        return ColtOutput(cand_logits=cand.unsqueeze(1), cls_logits=cls.unsqueeze(1),
                          policy_logits=pol.unsqueeze(1))


@torch.no_grad()
def poisoning_rate(model, ctx, clues, sols, n_cand, theta_elim) -> float:
    x = L.clues_to_lattice(clues, n_cand)
    out = model(x, ctx)
    b = out.final()[0]
    x1 = L.eliminate(x, b, theta_elim)
    sol_alive = x1.gather(2, sols.unsqueeze(-1)).squeeze(-1) > 0.5
    return float((~sol_alive.all(dim=1)).float().mean().item())


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--frames", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--agg", choices=["mean", "union"], default="union")
    ap.add_argument("--n-chains", type=int, default=32)
    ap.add_argument("--max-rounds", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base = _build_model(ckpt["config"])
    base.load_state_dict(ckpt["model"])
    base.eval()
    if torch.cuda.is_available():
        base = base.cuda()
    device = next(base.parameters()).device

    with args.split.open() as f:
        f.readline()
        n_tokens = len(f.readline().split("\t")[1].split(" "))
    geom = geometry_for(int(round(n_tokens ** 0.5)))
    ctx = context_for(geom).to(device)
    ds = SudokuDataset(args.split, geom)
    clues = ds.clues_tensor().to(device)
    sols = ds.solution_tensor().to(device)
    cfgs = ckpt["config"]
    step_cfg = StepConfig(theta_elim=float(cfgs["solve"]["theta_elim"]),
                          theta_cls=float(cfgs["eval"]["theta_cls"]),
                          tau_decide=float(cfgs["solve"]["tau_decide"]))
    verify = lambda g: satisfies_constraints(g, geom)

    settings = []
    for K in args.frames:
        gen = torch.Generator().manual_seed(args.seed)
        wrapped = SymmetryFrames(base, K, args.agg, gen) if K > 1 else base
        pois = poisoning_rate(wrapped, ctx, clues, sols, geom.n_cand, step_cfg.theta_elim)
        sgen = torch.Generator(device=device).manual_seed(args.seed)
        solved = 0
        slot = 4
        for s in range(0, len(ds), slot):
            e = min(s + slot, len(ds))
            x0 = L.clues_to_lattice(clues[s:e], geom.n_cand)
            r = dfs_solve(wrapped, ctx, x0, n_chains=args.n_chains, max_rounds=args.max_rounds,
                          cfg=step_cfg, verify_fn=verify, generator=sgen, policy_mode="learned")
            solved += int(r.solved.sum().item())
        acc = solved / len(ds)
        settings.append({"frames": K, "agg": args.agg if K > 1 else "single",
                         "poisoning_rate": round(pois, 4), "accuracy": round(acc, 4)})
        print(json.dumps(settings[-1]))

    result = {"checkpoint": str(args.checkpoint), "split": str(args.split),
              "n_puzzles": len(ds), "settings": settings, "seed": args.seed}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
