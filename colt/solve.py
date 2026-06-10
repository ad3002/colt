"""CoLT search: training pool + two inference solvers (restart vs DFS).

Everything here runs inside LDT's *sound envelope*: a grid is only ever emitted
after passing the external constraint verifier, otherwise the model abstains.
Soundness therefore does not depend on any choice made in this file — which is
exactly what licenses aggressive learned search (DESIGN.md §2).

Components:

  Branch-cell policies (``select_branch_cells``)
      ``random``  — uniform over multi-candidate cells (LDT's choice; control).
      ``mrv``     — classical minimum-remaining-values heuristic (strong
                    hand-coded control the learned policy must beat).
      ``learned`` — sample/argmax over the policy head restricted to
                    multi-candidate cells.

  :class:`TrainPool`
      LDT's pool-based parallel solve, with branching driven by the configured
      policy so training states stay in-distribution for the search actually
      run at inference.

  :func:`restart_solve`
      LDT's multi-chain solver (conflict ⇒ restart from x₀) with pluggable
      cell policy. ``random`` reproduces LDT inference exactly (control arm).

  :func:`dfs_solve`
      The CoLT solver. Each chain keeps a decision stack; on conflict it
      *backjumps*: pops to the most recent decision with untried values,
      excludes the failed value, and resumes — chronological backtracking with
      value exclusion, i.e. depth-first search with the network as the
      propagator and the policy head as the variable-ordering heuristic. A
      per-puzzle **nogood ban set** of failed complete grids is shared across
      chains, so no wrong completion is ever re-explored — this directly
      converts LDT's observed pathology (tens of thousands of re-generated
      wrong grids on hard puzzles; see ad3002/LTD results) into search
      progress. Stack exhaustion ⇒ fresh restart (chains stay diverse through
      sampled value orderings).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from . import lattice as L
from .model import ColtModel
from .tasks.sudoku import TaskContext

VerifyFn = Callable[[torch.Tensor], torch.Tensor]
Sampler = Callable[[int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]

POLICY_MODES = ("random", "mrv", "learned")


@dataclass
class StepConfig:
    theta_elim: float = 0.1
    theta_cls: float = 0.6
    tau_decide: float = 1.5


# ----------------------------------------------------------------------------
# Branch-cell selection + value pinning
# ----------------------------------------------------------------------------


def select_branch_cells(
    x: torch.Tensor,
    policy_logits: torch.Tensor | None,
    mode: str,
    generator: torch.Generator | None = None,
    greedy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick one branch cell per batch element among multi-candidate cells.

    Returns (cell_idx (B,), branched (B,)); cell_idx is arbitrary where
    ``branched`` is False (caller must mask).
    """
    B, C, V = x.shape
    counts = L.count_candidates(x)
    multi = counts >= 2
    branched = multi.any(dim=1)

    if mode == "random":
        score = torch.rand(B, C, generator=generator, device=x.device)
    elif mode == "mrv":
        # fewest candidates first; random tie-break
        tie = torch.rand(B, C, generator=generator, device=x.device)
        score = -counts.float() + 0.01 * tie
    elif mode == "learned":
        assert policy_logits is not None, "learned policy requires policy_logits"
        if greedy:
            score = policy_logits
        else:
            g = -torch.log(-torch.log(
                torch.rand(B, C, generator=generator, device=x.device).clamp_min(1e-9)
            ))
            score = policy_logits + g  # Gumbel sampling ∝ softmax(policy)
    else:
        raise ValueError(f"unknown policy mode {mode!r}")

    score = score.masked_fill(~multi, float("-inf"))
    cell_idx = score.argmax(dim=1)
    return cell_idx, branched


def branch_value_probs(
    x_cell: torch.Tensor, logits_cell: torch.Tensor, tau_decide: float
) -> torch.Tensor:
    """(.., V) alive-masked softmax of the branch distribution at one cell."""
    alive = x_cell > 0.5
    masked = (logits_cell / tau_decide).masked_fill(~alive, float("-inf"))
    return torch.softmax(masked, dim=-1)


def pin_cells(
    x: torch.Tensor,
    cand_logits: torch.Tensor,
    cell_idx: torch.Tensor,
    branched: torch.Tensor,
    tau_decide: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Pin each branched element's chosen cell to a value sampled from the
    branch distribution. Non-branched elements are returned unchanged."""
    B, C, V = x.shape
    rows = torch.arange(B, device=x.device)
    x_cell = x[rows, cell_idx]                       # (B, V)
    logits_cell = cand_logits[rows, cell_idx]        # (B, V)
    probs = branch_value_probs(x_cell, logits_cell, tau_decide)
    safe = torch.where(branched.unsqueeze(-1), probs, torch.ones_like(probs) / V)
    chosen = torch.multinomial(safe, 1, generator=generator).squeeze(-1)
    pinned = F.one_hot(chosen, num_classes=V).to(x.dtype)
    x_new = x.clone()
    x_new[rows, cell_idx] = torch.where(branched.unsqueeze(-1), pinned, x_cell)
    return x_new


# ----------------------------------------------------------------------------
# Training pool (Algorithm 1, policy-aware)
# ----------------------------------------------------------------------------


@dataclass
class AdvanceStats:
    n_solved: int
    n_conflict: int
    n_reset: int
    mean_age: float
    mean_candidates: float


class TrainPool:
    def __init__(self, sampler: Sampler, batch_size: int, tau_age: int, device: torch.device):
        self.sampler = sampler
        self.B = batch_size
        self.tau_age = tau_age
        self.device = device
        x0, sols, sol_mask = sampler(batch_size)
        self.x = x0.to(device)
        self.sols = sols.to(device)
        self.sol_mask = sol_mask.to(device)
        self.age = torch.zeros(batch_size, dtype=torch.long, device=device)

    def current(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x, self.sols, self.sol_mask

    @torch.no_grad()
    def advance(
        self,
        cand_final_logits: torch.Tensor,
        cand_branch_logits: torch.Tensor,
        policy_logits: torch.Tensor,
        cfg: StepConfig,
        generator: torch.Generator | None = None,
        policy_mode: str = "learned",
    ) -> AdvanceStats:
        x1 = L.eliminate(self.x, cand_final_logits, cfg.theta_elim)
        _, no_solution = L.abstraction_target(x1, self.sols, self.sol_mask)
        conflict = no_solution
        solved = L.is_all_singleton(x1) & ~no_solution

        cells, branched = select_branch_cells(x1, policy_logits, policy_mode, generator)
        x2 = pin_cells(x1, cand_branch_logits, cells, branched, cfg.tau_decide, generator)

        self.age = self.age + 1
        reset = conflict | solved | (self.age >= self.tau_age)
        n_reset = int(reset.sum().item())
        if n_reset > 0:
            fx, fs, fm = self.sampler(n_reset)
            idx = reset.nonzero(as_tuple=False).squeeze(-1)
            x2[idx] = fx.to(self.device)
            self.sols[idx] = fs.to(self.device)
            self.sol_mask[idx] = fm.to(self.device)
            self.age[idx] = 0

        self.x = x2
        return AdvanceStats(
            n_solved=int(solved.sum().item()),
            n_conflict=int(conflict.sum().item()),
            n_reset=n_reset,
            mean_age=float(self.age.float().mean().item()),
            mean_candidates=float(L.count_candidates(self.x).float().mean().item()),
        )


# ----------------------------------------------------------------------------
# Inference result container
# ----------------------------------------------------------------------------


@dataclass
class InferenceResult:
    solved: torch.Tensor
    solution: torch.Tensor
    rounds_used: torch.Tensor
    unsound_outputs: int
    restarts: int = 0
    backjumps: int = 0


# ----------------------------------------------------------------------------
# Restart solver (LDT control with pluggable policy)
# ----------------------------------------------------------------------------


@torch.no_grad()
def restart_solve(
    model: ColtModel,
    ctx: TaskContext,
    x0: torch.Tensor,
    n_chains: int,
    max_rounds: int,
    cfg: StepConfig,
    verify_fn: VerifyFn | None = None,
    generator: torch.Generator | None = None,
    policy_mode: str = "random",
) -> InferenceResult:
    model.eval()
    device = x0.device
    M, C, V = x0.shape
    K = n_chains
    x = x0.repeat_interleave(K, dim=0).clone()
    x0_chain = x.clone()
    puzzle_of_chain = torch.arange(M, device=device).repeat_interleave(K)

    solved = torch.zeros(M, dtype=torch.bool, device=device)
    solution = torch.full((M, C), -1, dtype=torch.long, device=device)
    rounds_used = torch.full((M,), max_rounds, dtype=torch.long, device=device)
    unsound = 0
    restarts = 0

    for r in range(max_rounds):
        if bool(solved.all()):
            break
        out = model(x, ctx)
        b_final, c_final, pol_final = out.final()
        x = L.eliminate(x, b_final, cfg.theta_elim)
        conflict = L.detect_conflict(x, c_final, cfg.theta_cls)
        is_solved = L.is_solved(x)

        solved_chains = is_solved.nonzero(as_tuple=False).squeeze(-1)
        if solved_chains.numel() > 0:
            grids = L.decode_singletons(x[solved_chains])
            ok = verify_fn(grids) if verify_fn is not None else torch.ones(
                grids.shape[0], dtype=torch.bool, device=device
            )
            unsound += int((~ok).sum().item())
            good = solved_chains[ok]
            if good.numel() > 0:
                pz = puzzle_of_chain[good]
                good_grids = L.decode_singletons(x[good])
                newly = ~solved[pz]
                if bool(newly.any()):
                    sel = newly.nonzero(as_tuple=False).squeeze(-1)
                    solved[pz[sel]] = True
                    solution[pz[sel]] = good_grids[sel]
                    rounds_used[pz[sel]] = r + 1

        active = ~(conflict | is_solved)
        cells, branched = select_branch_cells(x, pol_final, policy_mode, generator)
        x = pin_cells(x, b_final, cells, branched, cfg.tau_decide, generator)
        dead = ~active
        restart = dead & ~solved[puzzle_of_chain]
        if bool(restart.any()):
            ridx = restart.nonzero(as_tuple=False).squeeze(-1)
            x[ridx] = x0_chain[ridx]
            restarts += int(ridx.numel())

    return InferenceResult(solved=solved, solution=solution, rounds_used=rounds_used,
                           unsound_outputs=unsound, restarts=restarts)


# ----------------------------------------------------------------------------
# DFS solver (CoLT): decision stacks + backjumping + nogood ban set
# ----------------------------------------------------------------------------


@dataclass
class _Frame:
    """One decision node: pre-pin state, the branched cell, its branch
    distribution at decision time, and the values already tried there."""

    x_before: torch.Tensor          # (C, V)
    cell: int
    probs: torch.Tensor             # (V,)
    tried: torch.Tensor = field(default=None)  # (V,) bool

    def __post_init__(self):
        if self.tried is None:
            self.tried = torch.zeros_like(self.probs, dtype=torch.bool)


def _pop_and_resume(
    stack: list[_Frame],
    x0_row: torch.Tensor,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, bool, int]:
    """Backjump: find the deepest decision with untried alive values, try the
    next value there. Returns (new_state, did_restart, n_pops)."""
    pops = 0
    while stack:
        f = stack[-1]
        alive = f.x_before[f.cell] > 0.5
        remaining = alive & ~f.tried
        if remaining.any():
            probs = f.probs.masked_fill(~remaining, 0.0)
            total = probs.sum()
            if total <= 0:
                probs = remaining.float()
                total = probs.sum()
            probs = probs / total
            d = int(torch.multinomial(probs, 1, generator=generator).item())
            f.tried[d] = True
            x_new = f.x_before.clone()
            x_new[f.cell] = 0.0
            x_new[f.cell, d] = 1.0
            return x_new, False, pops
        stack.pop()
        pops += 1
    return x0_row.clone(), True, pops


@torch.no_grad()
def dfs_solve(
    model: ColtModel,
    ctx: TaskContext,
    x0: torch.Tensor,
    n_chains: int,
    max_rounds: int,
    cfg: StepConfig,
    verify_fn: VerifyFn | None = None,
    generator: torch.Generator | None = None,
    policy_mode: str = "learned",
    use_ban_set: bool = True,
) -> InferenceResult:
    model.eval()
    device = x0.device
    M, C, V = x0.shape
    K = n_chains
    N = M * K
    x = x0.repeat_interleave(K, dim=0).clone()
    x0_chain = x.clone()
    puzzle_of_chain = [i // K for i in range(N)]

    stacks: list[list[_Frame]] = [[] for _ in range(N)]
    ban: list[set[bytes]] = [set() for _ in range(M)]

    solved = torch.zeros(M, dtype=torch.bool, device=device)
    solution = torch.full((M, C), -1, dtype=torch.long, device=device)
    rounds_used = torch.full((M,), max_rounds, dtype=torch.long, device=device)
    unsound = 0
    restarts = 0
    backjumps = 0

    for r in range(max_rounds):
        if bool(solved.all()):
            break
        out = model(x, ctx)
        b_final, c_final, pol_final = out.final()
        x = L.eliminate(x, b_final, cfg.theta_elim)
        conflict_t = L.detect_conflict(x, c_final, cfg.theta_cls)
        solved_t = L.is_solved(x)

        # Pre-pick branch cells for all chains in one batched call.
        cells, branched = select_branch_cells(x, pol_final, policy_mode, generator)

        for i in range(N):
            p = puzzle_of_chain[i]
            if bool(solved[p]):
                continue
            hit_conflict = bool(conflict_t[i])

            if bool(solved_t[i]):
                grid = L.decode_singletons(x[i].unsqueeze(0)).squeeze(0)
                key = bytes(grid.tolist()) if use_ban_set else None
                if key is not None and key in ban[p]:
                    hit_conflict = True       # known-bad completion: prune
                else:
                    ok = bool(verify_fn(grid.unsqueeze(0)).item()) if verify_fn is not None else True
                    if ok:
                        solved[p] = True
                        solution[p] = grid
                        rounds_used[p] = r + 1
                        continue
                    unsound += 1
                    if key is not None:
                        ban[p].add(key)
                    hit_conflict = True       # treat the wrong leaf as a conflict

            if hit_conflict:
                x_new, did_restart, pops = _pop_and_resume(stacks[i], x0_chain[i], generator)
                x[i] = x_new
                backjumps += pops
                if did_restart:
                    stacks[i].clear()
                    restarts += 1
                continue

            # Active chain: commit one decision (push a frame, pin the value).
            if bool(branched[i]):
                cell = int(cells[i].item())
                probs = branch_value_probs(x[i, cell], b_final[i, cell], cfg.tau_decide)
                frame = _Frame(x_before=x[i].clone(), cell=cell, probs=probs)
                d = int(torch.multinomial(probs, 1, generator=generator).item())
                frame.tried[d] = True
                stacks[i].append(frame)
                x[i, cell] = 0.0
                x[i, cell, d] = 1.0

    return InferenceResult(solved=solved, solution=solution, rounds_used=rounds_used,
                           unsound_outputs=unsound, restarts=restarts, backjumps=backjumps)
