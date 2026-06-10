#!/usr/bin/env python3
"""Zero-shot 9×9 propagation-transfer probe (Phase 3, BENCHMARKS.md).

Takes a checkpoint trained WITHOUT any 9×9 data and measures, on 9×9 states it
has never seen, whether the learned propagator transfers:

  * elimination precision — of the candidates one forward pass would eliminate
    (σ(b) < θ_elim among alive), the fraction that are correct to eliminate
    (i.e. not the solution value at that cell);
  * elimination recall — of the candidates that should be eliminated
    (alive ∧ not the solution value), the fraction the pass eliminates;
  * conflict-head AUC — ranking quality of σ(c) separating ⊥ states (true
    solution-value removed somewhere) from consistent states;
  * policy survival margin — mean p* of the policy's argmax cell minus the
    mean p* of a random multi-candidate cell (does the learned ordering still
    point at safer branches off-distribution?).

States are sampled at mixed deduction depths via the same reveal scheme used by
the training curriculum (but from *clues + solution* of the held-out 9×9 set;
no training happens here). Probes run a single forward pass per state.

Usage:
    python scripts/transfer_probe.py --checkpoint runs/colt_multi/final.pt \
        --split data/sudoku9_small/test.tsv --out results/transfer9_multi.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from colt import lattice as L                                   # noqa: E402
from colt.eval import _build_model                              # noqa: E402
from colt.losses import policy_target                           # noqa: E402
from colt.solve import select_branch_cells                      # noqa: E402
from colt.tasks.sudoku import SudokuDataset, context_for, geometry_for  # noqa: E402


def auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Mann–Whitney AUC of scores for labels ∈ {0,1}."""
    pos = scores[labels]
    neg = scores[~labels]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    greater = (pos.unsqueeze(1) > neg.unsqueeze(0)).float().mean()
    ties = (pos.unsqueeze(1) == neg.unsqueeze(0)).float().mean()
    return float(greater + 0.5 * ties)


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--n-states", type=int, default=512)
    ap.add_argument("--theta-elim", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = _build_model(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    geom = geometry_for(9)
    ctx = context_for(geom)
    ds = SudokuDataset(args.split, geom)
    clues_all = ds.clues_tensor()
    sols_all = ds.solution_tensor()
    C, V = geom.n_cells, geom.n_cand

    gen = torch.Generator().manual_seed(args.seed)
    n = args.n_states
    idx = torch.randint(0, len(ds), (n,), generator=gen)
    cl = clues_all[idx].clone()
    sol = sols_all[idx]
    # mixed-depth states: reveal blanks with per-state q ~ U(0, 0.9)
    q = torch.rand(n, 1, generator=gen) * 0.9
    blank = cl < 0
    reveal = blank & (torch.rand(n, C, generator=gen) < q)
    cl = torch.where(reveal, sol, cl)
    x = L.clues_to_lattice(cl, V)

    # corrupt half the states into ⊥ (remove the true value at one undecided cell)
    corrupt = torch.zeros(n, dtype=torch.bool)
    corrupt[: n // 2] = True
    corrupt = corrupt[torch.randperm(n, generator=gen)]
    for i in range(n):
        if corrupt[i]:
            multi = (L.count_candidates(x[i : i + 1])[0] >= 2).nonzero().squeeze(-1)
            if multi.numel() == 0:
                corrupt[i] = False
                continue
            cell = multi[torch.randint(0, multi.numel(), (1,), generator=gen)].item()
            x[i, cell, sol[i, cell]] = 0.0

    out = model(x, ctx)
    b, c, pol = out.final()

    # elimination precision / recall on consistent states only
    cons = ~corrupt
    alive = x[cons] > 0.5
    sol_oh = torch.zeros_like(x[cons]).scatter_(2, sol[cons].unsqueeze(-1), 1.0) > 0.5
    should_kill = alive & ~sol_oh
    would_kill = alive & (torch.sigmoid(b[cons]) < args.theta_elim)
    tp = (would_kill & should_kill).sum().item()
    fp = (would_kill & ~should_kill).sum().item()   # killed a true solution value
    fn = (~would_kill & should_kill).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    # conflict-head AUC over corrupt vs consistent
    head_auc = auc(torch.sigmoid(c), corrupt)

    # policy survival margin on consistent states
    y_hat, _tib = L.abstraction_target(x[cons], sol[cons].unsqueeze(1))
    p_star, mask = policy_target(x[cons], b[cons], y_hat, tau_decide=1.5)
    cells, branched = select_branch_cells(x[cons], pol[cons], "learned", gen, greedy=True)
    rnd_cells, _ = select_branch_cells(x[cons], None, "random", gen)
    rows = torch.arange(cons.sum())
    ok = branched & mask[rows, cells]
    pol_p = p_star[rows, cells][ok].mean().item() if ok.any() else float("nan")
    ok_r = branched & mask[rows, rnd_cells]
    rnd_p = p_star[rows, rnd_cells][ok_r].mean().item() if ok_r.any() else float("nan")

    result = {
        "checkpoint": str(args.checkpoint),
        "split": str(args.split),
        "n_states": n,
        "elimination_precision": round(precision, 4),
        "elimination_recall": round(recall, 4),
        "false_eliminations": fp,
        "conflict_auc": round(head_auc, 4),
        "policy_survival_argmax": round(pol_p, 4),
        "policy_survival_random": round(rnd_p, 4),
        "policy_margin": round(pol_p - rnd_p, 4),
        "theta_elim": args.theta_elim,
        "seed": args.seed,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
