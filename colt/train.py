"""CoLT training — pool-based parallel solve with policy-aware branching.

Supports multi-task training (Phase 3): pass several dataset directories and
one pool per task is kept resident; optimizer steps round-robin across tasks.
The model is size-agnostic (relational bias + v_max padding) so a single set of
weights trains on all of them.

Usage (single task):
    python -m colt.train --config configs/colt6.yaml \
        --dataset data/sudoku6 --output-dir runs/colt6 --device cpu

Multi-task:
    python -m colt.train --config configs/colt_multi.yaml \
        --dataset data/sudoku4 data/sudoku6 --output-dir runs/colt_multi
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from . import lattice as L
from .losses import compute_colt_loss
from .model import ColtModel
from .solve import StepConfig, TrainPool, restart_solve
from .tasks.sudoku import SudokuDataset, TaskContext, context_for, geometry_for, satisfies_constraints


def _deep_set(d: dict, dotted_key: str, raw_value: str) -> None:
    parts = dotted_key.split(".")
    node = d
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    if raw_value.lower() in {"true", "false"}:
        node[parts[-1]] = raw_value.lower() == "true"
        return
    for cast in (int, float):
        try:
            node[parts[-1]] = cast(raw_value)
            return
        except ValueError:
            pass
    node[parts[-1]] = raw_value


def load_config(path: Path, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(path.read_text())
    for kv in overrides:
        if "=" not in kv:
            raise ValueError(f"override must be key=value, got {kv!r}")
        k, v = kv.split("=", 1)
        _deep_set(cfg, k, v)
    return cfg


def build_model(cfg: dict) -> ColtModel:
    m = cfg["model"]
    return ColtModel(
        d_model=int(m["d_model"]),
        n_heads=int(m["n_heads"]),
        n_layers=int(m["n_layers"]),
        n_iters=int(m["n_iters"]),
        ff_mult=int(m.get("ff_mult", 4)),
        v_max=int(m.get("v_max", 9)),
        use_rel_bias=bool(m.get("use_rel_bias", True)),
        use_coord_mlp=bool(m.get("use_coord_mlp", True)),
        pos_table_size=int(m.get("pos_table_size", 0)),
    )


def step_config(cfg: dict, eval_mode: bool = False) -> StepConfig:
    s = cfg["solve"]
    theta_cls = float(cfg["eval"]["theta_cls"]) if eval_mode else float(s.get("theta_cls_train", 0.9))
    return StepConfig(theta_elim=float(s["theta_elim"]), theta_cls=theta_cls,
                      tau_decide=float(s["tau_decide"]))


def cosine_warmup(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def select_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


@dataclass
class TaskBundle:
    name: str
    ctx: TaskContext
    train_ds: SudokuDataset
    test_ds: SudokuDataset | None
    pool: TrainPool


def make_sampler(dataset: SudokuDataset, n_cand: int, sample_gen: torch.Generator, state: dict,
                 augment_digits: bool = False):
    """Curriculum sampler over deduction depth (same scheme as ad3002/LTD):
    fresh trajectories get blanks pinned to solution values with per-puzzle
    probability q ~ U(0, state['reveal']); the loop anneals reveal → 0 so the
    end of training matches the clues-only inference distribution.

    ``augment_digits`` applies a fresh random digit permutation to every sampled
    trajectory (clues and solution together) — the dataset-level augmentation of
    the LDT paper (its full recipe is digit permutations × dihedral symmetries,
    a ~2880× expansion). Sudoku constraints are invariant under value
    relabeling, so this is exact. Beyond data volume, it de-correlates the
    propagator's elimination errors across relabelings — the mechanism the
    failure-anatomy probe identifies as the accuracy ceiling (first-pass
    poisoning)."""
    clues_all = dataset.clues_tensor()
    sol_all = dataset.solution_tensor()
    C = clues_all.shape[1]

    def sampler(n: int):
        idx = torch.randint(0, len(dataset), (n,), generator=sample_gen)
        cl = clues_all[idx].clone()
        sol = sol_all[idx].clone()
        if augment_digits:
            perms = torch.stack([torch.randperm(n_cand, generator=sample_gen) for _ in range(n)])
            sol = perms.gather(1, sol)
            given = cl >= 0
            cl_safe = perms.gather(1, cl.clamp_min(0))
            cl = torch.where(given, cl_safe, cl)
        reveal = state["reveal"]
        if reveal > 0.0:
            q = torch.rand(n, 1, generator=sample_gen) * reveal
            blank = cl < 0
            reveal_mask = blank & (torch.rand(n, C, generator=sample_gen) < q)
            cl = torch.where(reveal_mask, sol, cl)
        x0 = L.clues_to_lattice(cl, n_cand)
        return x0, sol.unsqueeze(1), torch.ones(n, 1, dtype=torch.bool)

    return sampler


@torch.no_grad()
def quick_probe(model, bundle: TaskBundle, cfg: dict, device, gen, max_puzzles: int = 64) -> dict:
    if bundle.test_ds is None:
        return {}
    n = min(max_puzzles, len(bundle.test_ds))
    clues = bundle.test_ds.clues_tensor()[:n]
    x0 = L.clues_to_lattice(clues, bundle.ctx.n_cand).to(device)
    res = restart_solve(
        model, bundle.ctx, x0,
        n_chains=int(cfg["eval"].get("probe_chains", 16)),
        max_rounds=int(cfg["eval"].get("probe_rounds", 48)),
        cfg=step_config(cfg, eval_mode=True),
        verify_fn=lambda g: satisfies_constraints(g, bundle.ctx.geom),
        generator=gen,
        policy_mode="learned",
    )
    return {f"probe_{bundle.name}": round(res.solved.float().mean().item(), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, nargs="+", required=True,
                    help="one or more dirs with train.tsv (+ test.tsv)")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed
    seed = int(cfg["train"]["seed"])
    torch.manual_seed(seed)

    device = select_device(args.device)
    print(f"[device] {device}", file=sys.stderr)

    tr = cfg["train"]
    reveal_max = float(tr.get("curriculum_reveal_max", 0.0))
    anneal_frac = float(tr.get("curriculum_anneal_frac", 1.0))
    curr_state = {"reveal": reveal_max}
    sample_gen = torch.Generator().manual_seed(seed)
    solve_gen = torch.Generator(device=device).manual_seed(seed + 1)

    model = build_model(cfg).to(device)
    print(f"[model] {model.num_params():,} parameters (size-agnostic, v_max={model.v_max})",
          file=sys.stderr)

    bundles: list[TaskBundle] = []
    for ds_dir in args.dataset:
        # infer board size from the data: cells = n^2
        geom_n = None
        with (ds_dir / "train.tsv").open() as f:
            f.readline()
            n_tokens = len(f.readline().split("\t")[1].split(" "))
        geom_n = int(round(n_tokens ** 0.5))
        geom = geometry_for(geom_n)
        ctx = context_for(geom).to(device)
        train_ds = SudokuDataset(ds_dir / "train.tsv", geom)
        test_ds = SudokuDataset(ds_dir / "test.tsv", geom) if (ds_dir / "test.tsv").exists() else None
        sampler = make_sampler(train_ds, ctx.n_cand, sample_gen, curr_state,
                               augment_digits=bool(tr.get("augment_digits", False)))
        pool = TrainPool(sampler, batch_size=int(tr["batch_size"]),
                         tau_age=int(cfg["solve"]["tau_age"]), device=device)
        bundles.append(TaskBundle(ctx.name, ctx, train_ds, test_ds, pool))
        print(f"[task] {ctx.name}: train={len(train_ds)} test={len(test_ds) if test_ds else 0} "
              f"C={geom.n_cells} V={ctx.n_cand}", file=sys.stderr)

    opt = torch.optim.AdamW(model.parameters(), lr=float(tr["lr"]),
                            weight_decay=float(tr["weight_decay"]),
                            betas=tuple(float(b) for b in tr.get("betas", [0.9, 0.95])))
    steps = int(tr["steps"])
    warmup = int(tr.get("warmup_frac", 0.1) * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_warmup(s, steps, warmup))
    grad_clip = float(tr["grad_clip"])
    cfg_step = step_config(cfg, eval_mode=False)
    policy_mode = str(tr.get("pool_policy", "learned"))
    lw = cfg["loss"]
    log_every = int(cfg["logging"]["log_every"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"[train] steps={steps} batch={tr['batch_size']} tasks={[b.name for b in bundles]} "
          f"curriculum={reveal_max}/{anneal_frac} pool_policy={policy_mode}", file=sys.stderr)

    history: list[dict] = []
    t0 = time.time()
    for step in range(steps):
        if reveal_max > 0:
            frac = min(1.0, step / max(1, int(anneal_frac * steps)))
            curr_state["reveal"] = reveal_max * (1.0 - frac)
        bundle = bundles[step % len(bundles)]
        x, sols, mask = bundle.pool.current()
        out = model(x, bundle.ctx)
        y_hat, tib = L.abstraction_target(x, sols, mask)
        loss = compute_colt_loss(
            out, x, y_hat, tib,
            w_plus=float(lw["w_plus"]), w_minus=float(lw["w_minus"]),
            lambda_cls=float(lw["lambda_cls"]), lambda_ce=float(lw["lambda_ce"]),
            lambda_policy=float(lw.get("lambda_policy", 0.25)),
            tau_decide=cfg_step.tau_decide,
        )
        opt.zero_grad(set_to_none=True)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        sched.step()
        b_final, _c, pol_final = out.final()
        stats = bundle.pool.advance(b_final.detach(), out.mean_cand().detach(),
                                    pol_final.detach(), cfg_step, solve_gen,
                                    policy_mode=policy_mode)

        if step % log_every == 0 or step == steps - 1:
            d = loss.as_dict_for_logging()
            d.update({"step": step, "task": bundle.name, "lr": sched.get_last_lr()[0],
                      "wall_s": round(time.time() - t0, 1), "reveal": round(curr_state["reveal"], 3),
                      "n_solved": stats.n_solved, "n_conflict": stats.n_conflict,
                      "mean_cand": round(stats.mean_candidates, 3)})
            if args.eval_every and step > 0 and step % args.eval_every == 0:
                for b in bundles:
                    d.update(quick_probe(model, b, cfg, device, solve_gen))
            history.append(d)
            print(json.dumps(d), file=sys.stderr)

    ckpt = args.output_dir / "final.pt"
    torch.save({"model": model.state_dict(), "config": cfg, "seed": seed, "steps": steps}, ckpt)
    (args.output_dir / "history.jsonl").write_text("\n".join(json.dumps(h) for h in history) + "\n")
    print(f"[done] checkpoint → {ckpt}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
