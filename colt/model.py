"""The CoLT network — a size-agnostic lattice reasoner with search heads.

Same recurrent skeleton as our LDT reimplementation (a shared stack of
[Attention + SwiGLU] layers unrolled L times with input reinjection, heads read
at every iteration), with three differences that implement DESIGN.md:

1. **Size-agnostic I/O.** The lattice multi-hot is padded from the task's V to
   a fixed ``v_max`` before the input projection, and the candidate head emits
   ``v_max`` logits sliced back to V. Padded value slots are permanently dead
   candidates — semantically consistent with the lattice. One weight shape
   serves every board size.

2. **Structure via the constraint graph, not positional tables.** Attention
   layers add a learned per-head relation bias (colt.blocks.RelationalMHSA)
   over the task's relation matrix; cell content gets an MLP over normalized
   coordinates. Nothing in the parameters depends on board size.

3. **A policy head.** Per cell, a scalar logit π_i scoring "how safe is it to
   branch here" — trained to predict the probability that sampling a value
   from the branch distribution at cell i keeps a known solution alive
   (colt.losses.policy_target). At inference this replaces LDT's
   uniform-random branch-cell choice. The conflict (CLS) head doubles as the
   value function V(x) ≈ P(state still consistent) = 1 − σ(c).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import RelationalStack, count_parameters
from .tasks.sudoku import TaskContext

V_MAX_DEFAULT = 9
COORD_FEATS = 5


@dataclass
class ColtOutput:
    """Per-iteration logits from one forward pass.

    cand_logits:   (B, L, C, V)  candidate logits (sliced to the task's V).
    cls_logits:    (B, L)        conflict-head logit.
    policy_logits: (B, L, C)     branch-cell policy logit.
    """

    cand_logits: torch.Tensor
    cls_logits: torch.Tensor
    policy_logits: torch.Tensor

    def final(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.cand_logits[:, -1], self.cls_logits[:, -1], self.policy_logits[:, -1]

    def mean_cand(self) -> torch.Tensor:
        """(B, C, V) iteration-averaged candidate logits (training branch, Alg. 2)."""
        return self.cand_logits.mean(dim=1)


class ColtModel(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        n_iters: int = 10,
        ff_mult: int = 4,
        v_max: int = V_MAX_DEFAULT,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_iters = n_iters
        self.v_max = v_max

        self.input_proj = nn.Linear(v_max, d_model)
        self.coord_mlp = nn.Sequential(
            nn.Linear(COORD_FEATS, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        self.core = RelationalStack(n_layers, d_model, n_heads, ff_mult * d_model)

        self.cand_head = nn.Linear(d_model, v_max)
        self.cls_head = nn.Linear(d_model, 1)
        self.policy_head = nn.Linear(d_model, 1)

    def _pad_v(self, x: torch.Tensor) -> torch.Tensor:
        V = x.shape[-1]
        if V == self.v_max:
            return x
        if V > self.v_max:
            raise ValueError(f"task V={V} exceeds model v_max={self.v_max}")
        return F.pad(x, (0, self.v_max - V))

    def encode(self, x: torch.Tensor, ctx: TaskContext) -> torch.Tensor:
        """Lattice state (B, C, V) → injected sequence (B, C+1, d).

        ``ctx.coord_feats`` may be (C, F) shared across the batch or (B, C, F)
        per-sample (per-instance constraint graphs, e.g. random-graph coloring).
        """
        B = x.shape[0]
        coord = self.coord_mlp(ctx.coord_feats)
        if coord.dim() == 2:
            coord = coord.unsqueeze(0)
        cell = self.input_proj(self._pad_v(x)) + coord
        cls = self.cls_token.expand(B, -1, -1)
        return torch.cat([cls, cell], dim=1)

    def forward(self, x: torch.Tensor, ctx: TaskContext, n_iters: int | None = None) -> ColtOutput:
        L = n_iters if n_iters is not None else self.n_iters
        V = x.shape[-1]
        inj = self.encode(x, ctx)
        h = torch.zeros_like(inj)
        cand_list, cls_list, pol_list = [], [], []
        for _ in range(L):
            h = self.core(h + inj, ctx.rel_ids)
            cls_h = h[:, 0, :]
            cell_h = h[:, 1:, :]
            cand_list.append(self.cand_head(cell_h)[..., :V])          # (B, C, V)
            cls_list.append(self.cls_head(cls_h).squeeze(-1))          # (B,)
            pol_list.append(self.policy_head(cell_h).squeeze(-1))      # (B, C)
        return ColtOutput(
            cand_logits=torch.stack(cand_list, dim=1),
            cls_logits=torch.stack(cls_list, dim=1),
            policy_logits=torch.stack(pol_list, dim=1),
        )

    def num_params(self) -> int:
        return count_parameters(self)
