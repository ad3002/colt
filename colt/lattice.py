"""The lattice: candidate-set state and the deduction operators.

This module is the formal core of the paper (arXiv:2605.08605v1, §2 "Sound
deduction on a lattice" and §4 Algorithms 1–2). Everything here is pure tensor
math on a batched candidate-set representation; the neural network lives
elsewhere (:mod:`ltd.model`).

State representation
--------------------
A problem with ``C`` cells and ``V`` candidate values per cell is represented
by a **multi-hot lattice state** ``x ∈ {0,1}^{B×C×V}``. ``x[b, i, v] = 1`` means
"value ``v`` is still viable at cell ``i``" in batch element ``b``. The set of
candidate sets forms a finite lattice under inclusion:

    meet  a ⊓ b  = elementwise min  (intersection of candidate sets)
    join  a ⊔ b  = elementwise max  (union)
    top   ⊤       = all-ones         (nothing decided yet)
    bottom ⊥      = any cell empty   (unsatisfiable)

Deduction *only ever removes* candidates — the state moves monotonically *down*
the lattice. That monotonicity is what makes the procedure *sound*: a value is
never resurrected, so as long as the network never eliminates a value that is
part of a real solution, every real solution stays reachable.

The abstraction / concretization pair (§2)
-------------------------------------------
    α(S)(i) = { s(i) : s ∈ S }                  abstraction  (concrete → lattice)
    γ(a)    = { s ∈ Sols : ∀i. s(i) ∈ a(i) }    concretization (lattice → concrete)
    ded_p(a) = α(γ(a) ∩ ‖p‖)                    most-precise sound deduction

The supervised training target is the most-precise sound deduction restricted
to the *known* solution set ``Y`` (Algorithm 2):

    ŷ = x ⊓ α({ y ∈ Y : y consistent with x })

For a unique-solution puzzle (Sudoku, ``K = 1``) this collapses to: "if the one
true solution is still alive in ``x``, the target is its one-hot at every cell;
otherwise the state is ⊥ (a wrong branch was taken) and the network should
raise the conflict flag." For multi-solution problems (Maze, ``K`` up to 512)
the target softens to the union over all surviving solutions and sharpens as
the state commits.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Encoding / decoding
# ----------------------------------------------------------------------------


def clues_to_lattice(clues: torch.Tensor, n_cand: int) -> torch.Tensor:
    """Initial lattice ``x₀`` from a puzzle. ``clues``: (B, C) long, value in
    ``[0, n_cand)`` for a given cell or ``-1`` for an undetermined cell.

    Given cells become a one-hot (pinned singleton); undetermined cells start as
    the full candidate set (⊤ at that cell).
    """
    B, C = clues.shape
    x = clues.new_zeros((B, C, n_cand), dtype=torch.float32)
    given = clues >= 0
    # full set for undetermined cells
    x[~given] = 1.0
    # one-hot for given cells
    if given.any():
        idx = clues.clamp_min(0).unsqueeze(-1)  # (B, C, 1)
        onehot = torch.zeros_like(x).scatter_(2, idx, 1.0)
        x = torch.where(given.unsqueeze(-1), onehot, x)
    return x


def solutions_to_onehot(sols: torch.Tensor, n_cand: int) -> torch.Tensor:
    """(B, K, C) long digits → (B, K, C, V) float one-hot."""
    return F.one_hot(sols.long(), num_classes=n_cand).float()


def count_candidates(x: torch.Tensor) -> torch.Tensor:
    """(B, C, V) → (B, C) number of alive candidates per cell."""
    return (x > 0.5).sum(dim=-1)


def is_bottom(x: torch.Tensor) -> torch.Tensor:
    """(B,) bool — True if any cell has an empty candidate set (⊥)."""
    return (count_candidates(x) == 0).any(dim=-1)


def is_all_singleton(x: torch.Tensor) -> torch.Tensor:
    """(B,) bool — True if every cell has exactly one alive candidate."""
    return (count_candidates(x) == 1).all(dim=-1)


def is_solved(x: torch.Tensor) -> torch.Tensor:
    """(B,) bool — fully determined and not ⊥.

    Note this is *lattice*-solved (all singletons). Whether the singleton
    assignment actually satisfies the problem constraints is a separate check
    (:func:`ltd.datasets.sudoku.satisfies_constraints`) used for soundness.
    """
    return is_all_singleton(x) & ~is_bottom(x)


def decode_singletons(x: torch.Tensor) -> torch.Tensor:
    """(B, C, V) → (B, C) long: the chosen value where a cell is a singleton,
    else ``-1``. Cells with 0 or ≥2 candidates decode to ``-1``."""
    counts = count_candidates(x)
    val = x.argmax(dim=-1)
    return torch.where(counts == 1, val, torch.full_like(val, -1))


# ----------------------------------------------------------------------------
# Lattice meet / join
# ----------------------------------------------------------------------------


def meet(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a ⊓ b — intersection of candidate sets (elementwise min)."""
    return torch.minimum(a, b)


def join(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a ⊔ b — union of candidate sets (elementwise max)."""
    return torch.maximum(a, b)


# ----------------------------------------------------------------------------
# Abstraction operator α and the supervised deduction target
# ----------------------------------------------------------------------------


def surviving_solutions(
    x: torch.Tensor, sols: torch.Tensor, sol_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Which known solutions are still consistent with the current lattice.

    A solution ``y`` is *consistent with* ``x`` iff its value at every cell is
    still alive: ``∀i. x[i, y(i)] = 1``.

    Args:
        x:        (B, C, V) lattice state.
        sols:     (B, K, C) long, candidate value per cell for each solution.
        sol_mask: (B, K) bool marking valid solution slots (padding-safe for
                  variable ``K``). Defaults to all-valid.

    Returns:
        (B, K) bool — True where solution ``k`` survives in ``x``.
    """
    B, K, C = sols.shape
    # gather x at each solution's per-cell value: (B, K, C)
    x_exp = x.unsqueeze(1).expand(B, K, C, x.shape[-1])
    alive_at_sol = x_exp.gather(3, sols.long().unsqueeze(-1)).squeeze(-1)  # (B,K,C)
    consistent = (alive_at_sol > 0.5).all(dim=2)  # (B, K)
    if sol_mask is not None:
        consistent = consistent & sol_mask
    return consistent


def abstraction_target(
    x: torch.Tensor,
    sols: torch.Tensor,
    sol_mask: torch.Tensor | None = None,
    n_cand: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ŷ = x ⊓ α({y ∈ Y : y consistent with x}) and the ⊥ indicator.

    Args:
        x:    (B, C, V) lattice state.
        sols: (B, K, C) long solution values.
        sol_mask: (B, K) bool valid-slot mask, or None.
        n_cand: V (inferred from x if None).

    Returns:
        y_hat:           (B, C, V) float multi-hot supervised target.
        target_is_bottom: (B,) bool — True when *no* known solution survives
                          (the branch is dead; the conflict head should fire).
    """
    V = n_cand if n_cand is not None else x.shape[-1]
    consistent = surviving_solutions(x, sols, sol_mask)  # (B, K)
    sols_oh = solutions_to_onehot(sols, V)               # (B, K, C, V)
    # α over surviving solutions: union (max) of their one-hots, masking dead ones.
    masked = sols_oh * consistent.float().unsqueeze(-1).unsqueeze(-1)
    alpha = masked.amax(dim=1)                            # (B, C, V) in {0,1}
    y_hat = meet(x, alpha)                                # x ⊓ α(S')
    target_is_bottom = ~consistent.any(dim=1)             # (B,)
    return y_hat, target_is_bottom


# ----------------------------------------------------------------------------
# Step-operator primitives (Algorithm 2)
# ----------------------------------------------------------------------------


def eliminate(x: torch.Tensor, cand_logits: torch.Tensor, theta_elim: float) -> torch.Tensor:
    """Threshold elimination (Algorithm 2, step 2).

    Zero out *alive* candidates whose predicted confidence ``σ(b) < θ_elim``.
    Already-dead candidates stay dead (we multiply by the current state), so the
    result is ``≤ x`` in the lattice order — deduction only ever removes.
    """
    keep = (torch.sigmoid(cand_logits) >= theta_elim).to(x.dtype)
    return x * keep


def detect_conflict(x: torch.Tensor, cls_logit: torch.Tensor, theta_cls: float) -> torch.Tensor:
    """(B,) bool conflict flag (Algorithm 2, step 3).

    Dual ⊥ representation (§3): the *implicit* signal is an empty candidate set
    at some cell; the *explicit* signal is the CLS conflict head firing above
    ``θ_CLS``. Either triggers a conflict.
    """
    return is_bottom(x) | (torch.sigmoid(cls_logit) > theta_cls)


def branch(
    x: torch.Tensor,
    cand_logits: torch.Tensor,
    tau_decide: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stochastic branching (Algorithm 2, step 4).

    For each batch element with at least one multi-candidate cell, pick such a
    cell uniformly at random, sample a value ``d* ~ softmax(b_{i*} / τ_decide)``
    restricted to that cell's alive candidates, and **pin** the cell to ``d*``
    (commit a guess). Batch elements already fully determined are left unchanged.

    Returns:
        x_new:    (B, C, V) state after pinning.
        branched: (B,) bool — True where a cell was actually pinned.
        cell_idx: (B,) long — the pinned cell index (``-1`` where not branched).
    """
    B, C, V = x.shape
    counts = count_candidates(x)                       # (B, C)
    multi = counts >= 2                                 # (B, C)
    branched = multi.any(dim=1)                         # (B,)

    # Uniformly pick a multi-candidate cell: random scores, -inf elsewhere, argmax.
    rand = torch.rand(B, C, generator=generator, device=x.device)
    rand = rand.masked_fill(~multi, float("-inf"))
    cell_idx = rand.argmax(dim=1)                        # (B,)

    # Gather that cell's logits and alive mask: (B, V)
    gathered_logits = cand_logits.gather(1, cell_idx.view(B, 1, 1).expand(B, 1, V)).squeeze(1)
    gathered_alive = x.gather(1, cell_idx.view(B, 1, 1).expand(B, 1, V)).squeeze(1) > 0.5
    masked_logits = (gathered_logits / tau_decide).masked_fill(~gathered_alive, float("-inf"))
    probs = torch.softmax(masked_logits, dim=-1)         # (B, V)
    # multinomial needs finite, non-negative rows; dead rows (no branch) get
    # a uniform fallback so the call is well-defined — masked out below.
    safe = torch.where(branched.unsqueeze(-1), probs, torch.ones_like(probs) / V)
    chosen = torch.multinomial(safe, num_samples=1, generator=generator).squeeze(-1)  # (B,)

    pinned = F.one_hot(chosen, num_classes=V).to(x.dtype)  # (B, V)
    x_new = x.clone()
    rows = torch.arange(B, device=x.device)
    # only overwrite for branched elements
    new_cells = torch.where(branched.unsqueeze(-1), pinned, x[rows, cell_idx])
    x_new[rows, cell_idx] = new_cells
    cell_idx = torch.where(branched, cell_idx, torch.full_like(cell_idx, -1))
    return x_new, branched, cell_idx
