"""CoLT unit tests: relations, model size-agnosticism, policy, losses, solvers."""

from __future__ import annotations

import torch
import torch.nn as nn

from colt import lattice as L
from colt.losses import compute_colt_loss, policy_bce, policy_target
from colt.model import ColtModel, ColtOutput
from colt.solve import (
    StepConfig,
    TrainPool,
    _Frame,
    _pop_and_resume,
    dfs_solve,
    pin_cells,
    restart_solve,
    select_branch_cells,
)
from colt.tasks.sudoku import (
    SudokuGeometry,
    context_for,
    coord_features,
    geometry_for,
    relation_ids,
    satisfies_constraints,
)

V4 = 4


def _ctx(n: int):
    return context_for(geometry_for(n))


def _model(**kw) -> ColtModel:
    args = dict(d_model=32, n_heads=4, n_layers=2, n_iters=3)
    args.update(kw)
    return ColtModel(**args)


# --- geometry / relations ----------------------------------------------------


def test_relation_ids_scheme():
    rel = relation_ids(geometry_for(4))
    assert rel.shape == (17, 17)
    assert rel[0, 0].item() == 9          # CLS↔CLS
    assert rel[0, 5].item() == 8          # CLS↔cell
    assert rel[3, 3].item() == 7          # self = row+col+box
    # cells (0,0) and (0,1): same row + same 2×2 box → 1 + 4 = 5
    assert rel[1, 2].item() == 5
    # cells (0,0) and (0,2): same row only → 1
    assert rel[1, 3].item() == 1
    # cells (0,0) and (2,0): same col only → 2
    assert rel[1, 1 + 8].item() == 2


def test_coord_features_normalized():
    f = coord_features(geometry_for(6))
    assert f.shape == (36, 5)
    assert (f >= 0).all() and (f <= 1).all()


# --- model -------------------------------------------------------------------


def test_one_model_two_sizes():
    m = _model()
    for n in (4, 6, 9):
        ctx = _ctx(n)
        x = torch.ones(2, ctx.geom.n_cells, ctx.n_cand)
        out = m(x, ctx)
        assert out.cand_logits.shape == (2, 3, ctx.geom.n_cells, ctx.n_cand)
        assert out.cls_logits.shape == (2, 3)
        assert out.policy_logits.shape == (2, 3, ctx.geom.n_cells)
        assert torch.isfinite(out.cand_logits).all()


def test_backward_connects_all_params():
    m = _model()
    ctx = _ctx(4)
    x = (torch.rand(2, 16, V4) > 0.3).float()
    out = m(x, ctx)
    (out.cand_logits.sum() + out.cls_logits.sum() + out.policy_logits.sum()).backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert missing == []


def test_v_max_guard():
    m = _model(v_max=6)
    ctx = _ctx(9)
    x = torch.ones(1, 81, 9)
    try:
        m(x, ctx)
        raise AssertionError("expected ValueError for V > v_max")
    except ValueError:
        pass


# --- policy target / loss ----------------------------------------------------


def test_policy_target_math():
    # one cell, two alive values {0,1}; solution value 0; equal logits → p* = 0.5
    x = torch.zeros(1, 1, V4)
    x[0, 0, 0] = x[0, 0, 1] = 1.0
    y_hat = torch.zeros(1, 1, V4)
    y_hat[0, 0, 0] = 1.0
    logits = torch.zeros(1, 1, V4)
    p, mask = policy_target(x, logits, y_hat, tau_decide=1.5)
    assert mask[0, 0].item() is True
    assert abs(p[0, 0].item() - 0.5) < 1e-5
    # skew the logits toward the true value → p* > 0.5
    logits[0, 0, 0] = 3.0
    p2, _ = policy_target(x, logits, y_hat, tau_decide=1.5)
    assert p2[0, 0].item() > 0.8


def test_policy_target_masks_bottom_and_singletons():
    x = torch.ones(1, 2, V4)
    x[0, 1] = torch.tensor([1.0, 0, 0, 0])      # singleton cell
    y_hat = torch.zeros(1, 2, V4)                # ⊥ target (nothing survives)
    _, mask = policy_target(x, torch.zeros(1, 2, V4), y_hat, 1.5)
    assert not mask.any()                        # ⊥ row fully masked


def test_full_loss_backprops():
    cand = torch.randn(2, 3, 4, V4, requires_grad=True)
    cls = torch.randn(2, 3, requires_grad=True)
    pol = torch.randn(2, 3, 4, requires_grad=True)
    out = ColtOutput(cand, cls, pol)
    x = torch.ones(2, 4, V4)
    y_hat = torch.zeros(2, 4, V4)
    y_hat[:, :, 0] = 1.0
    tib = torch.tensor([False, True])
    loss = compute_colt_loss(out, x, y_hat, tib)
    loss.total.backward()
    assert all(t.grad is not None for t in (cand, cls, pol))
    assert torch.isfinite(loss.total)


# --- branch policies ----------------------------------------------------------


def test_mrv_picks_min_count_cell():
    x = torch.ones(1, 4, V4)
    x[0, 2] = torch.tensor([1.0, 1, 0, 0])
    gen = torch.Generator().manual_seed(0)
    cells, br = select_branch_cells(x, None, "mrv", gen)
    assert br.item() and cells.item() == 2


def test_learned_greedy_picks_argmax_policy():
    x = torch.ones(1, 4, V4)
    pol = torch.tensor([[0.0, 5.0, -1.0, 0.0]])
    cells, _ = select_branch_cells(x, pol, "learned", greedy=True)
    assert cells.item() == 1


def test_learned_respects_multi_mask():
    x = torch.ones(1, 3, V4)
    x[0, 1] = torch.tensor([1.0, 0, 0, 0])      # singleton — not branchable
    pol = torch.tensor([[0.0, 99.0, 0.0]])       # policy loves the singleton
    cells, _ = select_branch_cells(x, pol, "learned", greedy=True)
    assert cells.item() != 1


def test_pin_cells_pins_alive_value():
    x = torch.ones(2, 4, V4)
    logits = torch.zeros(2, 4, V4)
    gen = torch.Generator().manual_seed(0)
    cells = torch.tensor([1, 3])
    br = torch.tensor([True, True])
    x2 = pin_cells(x, logits, cells, br, 1.5, gen)
    assert (L.count_candidates(x2)[torch.arange(2), cells] == 1).all()


# --- DFS frame mechanics -------------------------------------------------------


def test_pop_and_resume_tries_next_value_then_restarts():
    x_before = torch.ones(2, V4)                  # 2 cells, all alive
    probs = torch.tensor([0.5, 0.5, 0.0, 0.0])
    f = _Frame(x_before=x_before.clone(), cell=0, probs=probs.clone())
    f.tried[0] = True                              # value 0 already failed
    stack = [f]
    x0 = torch.full((2, V4), 0.25)
    gen = torch.Generator().manual_seed(0)
    x_new, restarted, pops = _pop_and_resume(stack, x0, gen)
    assert not restarted and pops == 0
    assert x_new[0].tolist() == [0.0, 1.0, 0.0, 0.0]   # pinned to value 1
    assert f.tried.tolist() == [True, True, False, False]
    # exhaust value 1 too (probs only cover {0,1}) → node dead → restart
    stack2 = [f]
    x_new2, restarted2, pops2 = _pop_and_resume(stack2, x0, gen)
    # remaining alive&untried = {2,3}: probs are 0 there → uniform fallback
    assert not restarted2
    assert x_new2[0].argmax().item() in (2, 3)
    # now exhaust everything
    f.tried[:] = True
    stack3 = [f]
    x_new3, restarted3, pops3 = _pop_and_resume(stack3, x0, gen)
    assert restarted3 and pops3 == 1
    assert torch.equal(x_new3, x0)


# --- solvers with a deterministic stub ----------------------------------------


class _KeepAliveStub(nn.Module):
    """Keeps alive candidates, never fires the conflict head, flat policy."""

    n_iters = 1

    def forward(self, x, ctx, n_iters=None):
        cand = ((x * 2.0) - 1.0) * 10.0
        B, C, _ = x.shape
        return ColtOutput(
            cand_logits=cand.unsqueeze(1),
            cls_logits=torch.full((B, 1), -10.0),
            policy_logits=torch.zeros(B, 1, C),
        )

    def eval(self):
        return self


def test_dfs_ban_set_suppresses_and_prunes():
    """A solved-but-wrong grid is suppressed, banned, and the chain backtracks."""
    ctx = _ctx(4)
    sol = torch.tensor([[0, 1, 2, 3] + [0] * 12])  # nonsense grid, fully determined
    x0 = L.clues_to_lattice(sol, V4)
    res = dfs_solve(_KeepAliveStub(), ctx, x0, n_chains=2, max_rounds=4,
                    cfg=StepConfig(),
                    verify_fn=lambda g: torch.zeros(g.shape[0], dtype=torch.bool),
                    generator=torch.Generator().manual_seed(0))
    assert res.solved.tolist() == [False]
    assert res.unsound_outputs >= 1
    # the SECOND chain hitting the same grid must be pruned via the ban set,
    # not re-counted: with 2 chains × 4 rounds and one unique wrong grid,
    # unsound counts it once per chain at most before banning kicks in.
    assert res.unsound_outputs <= 2


def test_restart_solver_accepts_verified():
    ctx = _ctx(4)
    valid = torch.tensor([[0, 1, 2, 3, 2, 3, 0, 1, 1, 0, 3, 2, 3, 2, 1, 0]])
    x0 = L.clues_to_lattice(valid, V4)
    res = restart_solve(_KeepAliveStub(), ctx, x0, n_chains=2, max_rounds=3,
                        cfg=StepConfig(),
                        verify_fn=lambda g: satisfies_constraints(g, ctx.geom),
                        generator=torch.Generator().manual_seed(0))
    assert res.solved.tolist() == [True]
    assert torch.equal(res.solution[0], valid[0])


def test_trainpool_policy_modes_run():
    gen = torch.Generator().manual_seed(0)
    sol = torch.tensor([[0, 3, 1, 2, 2, 1, 3, 0, 1, 2, 0, 3, 3, 0, 2, 1]])

    def sampler(n):
        return (torch.ones(n, 16, V4), sol.expand(n, -1).unsqueeze(1).contiguous(),
                torch.ones(n, 1, dtype=torch.bool))

    for mode in ("random", "mrv", "learned"):
        pool = TrainPool(sampler, 4, 20, torch.device("cpu"))
        b = torch.randn(4, 16, V4)
        p = torch.randn(4, 16)
        stats = pool.advance(b, b, p, StepConfig(), gen, policy_mode=mode)
        assert stats.mean_candidates > 0


# --- exact verifier (the soundness gate) -------------------------------------


def test_verifier_accepts_valid_and_rejects_each_violation_type():
    import torch

    geom = geometry_for(4)
    valid = torch.tensor(
        [[0, 1, 2, 3,
          2, 3, 0, 1,
          1, 0, 3, 2,
          3, 2, 1, 0]], dtype=torch.long)
    assert satisfies_constraints(valid, geom).tolist() == [True]

    row_dup = valid.clone(); row_dup[0, 1] = 0          # duplicate in row 0
    col_dup = valid.clone(); col_dup[0, 4] = 0          # duplicate in column 0
    box_dup = valid.clone(); box_dup[0, 5] = 0          # duplicate in top-left box
    blank = valid.clone(); blank[0, 7] = -1             # incomplete grid
    for bad in (row_dup, col_dup, box_dup, blank):
        assert satisfies_constraints(bad, geom).tolist() == [False]

    batch = torch.cat([valid, row_dup, col_dup, box_dup, blank])
    assert satisfies_constraints(batch, geom).tolist() == [True, False, False, False, False]


# --- E8 component-ablation switches ------------------------------------------


def test_ablation_switches_shapes_and_frozen_zero_rel_bias():
    geom = geometry_for(4)
    ctx = context_for(geom)
    x = L.clues_to_lattice(torch.full((2, 16), -1, dtype=torch.long), geom.n_cand)

    # Arm A parameterization: positional tables only, no rel bias, no coord MLP.
    m = ColtModel(d_model=32, n_heads=2, n_layers=2, n_iters=3,
                  use_rel_bias=False, use_coord_mlp=False, pos_table_size=16)
    out = m(x, ctx)
    assert out.cand_logits.shape == (2, 3, 16, geom.n_cand)

    rel = [p for n, p in m.core.named_parameters() if "rel_bias" in n]
    assert rel and all(not p.requires_grad for p in rel)
    assert all(p.abs().sum().item() == 0.0 for p in rel)

    # One optimizer step must leave the frozen zero bias untouched
    # (plain-attention equivalence is exact, not approximate).
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2, weight_decay=0.1)
    m(x, ctx).mean_cand().sum().backward()
    opt.step()
    assert all(p.abs().sum().item() == 0.0 for p in rel)

    # pos_table shorter than the board must fail loudly, not truncate.
    small = ColtModel(d_model=32, n_heads=2, n_layers=2, n_iters=1, pos_table_size=9)
    try:
        small(x, ctx)
        assert False, "expected ValueError for pos_table smaller than board"
    except ValueError:
        pass
