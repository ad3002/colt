"""Transformer primitives for CoLT.

The key departure from LDT's blocks: position/structure information enters via
a **relational attention bias** over the task's constraint graph instead of
per-size learned positional tables. Every attention layer adds a learned
per-head scalar b[rel(i,j)] to the attention logits, where rel(i,j) encodes the
constraint relation between cells i and j (same row / column / box, bit-packed)
plus CLS rows. Cell content additionally gets a small MLP over *normalized*
coordinates. Both are size-agnostic: one set of weights serves any board size
or box shape — the mechanism behind CoLT's one-checkpoint-many-tasks claim
(DESIGN.md §4).

Relation id scheme (see colt.tasks.sudoku.relation_ids):
  bit0 = same row, bit1 = same column, bit2 = same box  → ids 0..7
  (id 7 = all three ⇔ i == j on a Sudoku grid, i.e. self)
  id 8 = CLS↔cell (either side is the CLS token), id 9 = CLS↔CLS.

Backbone block is the same pre-norm [Attention + SwiGLU] with RMSNorm used in
our LDT reimplementation (ad3002/LTD), which follows the HRM/TRM lineage.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_RELATIONS = 10


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, bias: bool = False):
        super().__init__()
        self.w_in = nn.Linear(d_model, 2 * d_hidden, bias=bias)
        self.w_out = nn.Linear(d_hidden, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, val = self.w_in(x).chunk(2, dim=-1)
        return self.w_out(F.silu(gate) * val)


class RelationalMHSA(nn.Module):
    """Bidirectional MHSA with an additive learned relation bias.

    ``rel_ids``: (S, S) long tensor of relation ids in [0, N_RELATIONS) for the
    full sequence (CLS + cells). The bias is a per-head scalar per relation id,
    added to the pre-softmax attention logits — the graph-attention-bias
    pattern (cf. Graphormer), here over the CSP constraint graph.
    """

    def __init__(self, d_model: int, n_heads: int, n_relations: int = N_RELATIONS):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.rel_bias = nn.Embedding(n_relations, n_heads)
        nn.init.zeros_(self.rel_bias.weight)  # start as plain attention

    def forward(self, x: torch.Tensor, rel_ids: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        if rel_ids.dim() == 2:                              # (S, S): one graph for the batch
            bias = self.rel_bias(rel_ids).permute(2, 0, 1).unsqueeze(0)   # (1, H, S, S)
        else:                                               # (B, S, S): per-sample graphs
            bias = self.rel_bias(rel_ids).permute(0, 3, 1, 2)             # (B, H, S, S)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        attn = attn.transpose(1, 2).contiguous().view(B, S, D)
        return self.out(attn)


class RelationalLayer(nn.Module):
    """One pre-norm [RelationalMHSA + SwiGLU] residual layer."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.norm_attn = RMSNorm(d_model)
        self.attn = RelationalMHSA(d_model, n_heads)
        self.norm_ffn = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor, rel_ids: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x), rel_ids)
        x = x + self.ffn(self.norm_ffn(x))
        return x


class RelationalStack(nn.Module):
    """Stack of ``n_layers`` relational layers + final RMSNorm; this is the unit
    the model re-applies L times (recurrent weight sharing)."""

    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [RelationalLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.final_norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, rel_ids: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, rel_ids)
        return self.final_norm(x)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
