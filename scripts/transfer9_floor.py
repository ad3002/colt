#!/usr/bin/env python3
"""E12a: rule-based floor for the zero-shot 9x9 elimination transfer.

Regenerates the same 512 mixed-depth states as scripts/transfer_probe.py
(same seed and construction), computes the eliminations reachable by plain
constraint propagation (peer elimination from decided cells, to fixpoint),
and reports the model's elimination precision/recall overall and restricted
to candidates NOT eliminated by propagation (incremental lift).

Usage:
    python scripts/transfer9_floor.py --checkpoint runs/colt_multi/final.pt \
        --split data/sudoku9_small/test.tsv --out results/transfer9_floor.json
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
from colt.tasks.sudoku import SudokuDataset, context_for, geometry_for  # noqa: E402


def propagate_fixpoint(x: torch.Tensor, peers) -> torch.Tensor:
    """Peer elimination from decided (singleton) cells, to fixpoint."""
    x = x.clone()
    B, C, V = x.shape
    changed = True
    while changed:
        changed = False
        counts = x.sum(dim=2)
        for b in range(B):
            for i in range(C):
                if counts[b, i] == 1:
                    v = int(x[b, i].argmax())
                    for p in peers[i]:
                        if x[b, p, v] > 0.5 and counts[b, p] > 1:
                            x[b, p, v] = 0.0
                            changed = True
            if changed:
                counts = x.sum(dim=2)
    return x


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--n-states", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--theta-elim", type=float, default=0.1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = _build_model(ckpt["config"]); model.load_state_dict(ckpt["model"]); model.eval()
    geom = geometry_for(9)
    ctx = context_for(geom)
    ds = SudokuDataset(args.split, geom)
    clues, sols = ds.clues_tensor(), ds.solution_tensor()

    # mixed-depth states, same construction as transfer_probe.py
    gen = torch.Generator().manual_seed(args.seed)
    n = args.n_states
    idx = torch.randint(0, len(ds), (n,), generator=gen)
    cl, sv = clues[idx], sols[idx]
    q = torch.rand(n, 1, generator=gen) * 0.9
    reveal = (torch.rand(n, geom.n_cells, generator=gen) < q) & (cl < 0)
    st = torch.where(reveal, sv, cl)
    x = L.clues_to_lattice(st, geom.n_cand)

    # peers for propagation
    peers = [[] for _ in range(geom.n_cells)]
    n9, br, bc = geom.n, geom.box_rows, geom.box_cols
    for i in range(geom.n_cells):
        r, c = divmod(i, n9)
        for j in range(geom.n_cells):
            if j == i: continue
            r2, c2 = divmod(j, n9)
            if r2 == r or c2 == c or (r2 // br == r // br and c2 // bc == c // bc):
                peers[i].append(j)

    xp = propagate_fixpoint(x, peers)
    prop_elim = (x > 0.5) & (xp < 0.5)                  # rule-based eliminations

    b = model(x, ctx).final()[0]
    model_elim = (x > 0.5) & (torch.sigmoid(b) < args.theta_elim)
    truth_elim = (x > 0.5) & (torch.nn.functional.one_hot(sv, geom.n_cand) < 0.5)

    def pr(mask):
        me, te = model_elim & mask, truth_elim & mask
        tp = int((me & te).sum()); fp = int((me & ~te).sum()); fn = int((te & ~me).sum())
        return {"tp": tp, "fp": fp, "fn": fn,
                "precision": round(tp / max(tp + fp, 1), 4),
                "recall": round(tp / max(tp + fn, 1), 4)}

    alive_undecided = (x > 0.5) & (x.sum(dim=2, keepdim=True) > 1)
    res = {
        "checkpoint": str(ckpt.get("config", {}).get("run_name", "")) or str(args.checkpoint),
        "n_states": n, "theta_elim": args.theta_elim,
        "propagation_floor": {
            "eliminations": int(prop_elim.sum()),
            "share_of_truth": round(int((prop_elim & truth_elim).sum()) /
                                    max(int(truth_elim.sum()), 1), 4)},
        "model_overall": pr(alive_undecided),
        "model_beyond_propagation": pr(alive_undecided & ~prop_elim),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
