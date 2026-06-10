"""CoLT training loss = LDT's Eq. 1 + a branch-policy term.

LDT components (identical to ad3002/LTD): asymmetric candidate BCE (w⁺ = 4.0,
w⁻ = 0.5 — false eliminations 8× costlier than false retentions), conflict-head
BCE against 1[ŷ = ⊥], singleton cross-entropy. All applied at every internal
iteration and averaged.

New: **policy BCE**. For each multi-candidate cell i, the supervised target is
the *branch-survival probability*

    p*_i = Σ_{d : ŷ_i[d] = 1}  softmax(b_i / τ_decide over alive candidates)[d]

— the probability that committing a value sampled from the actual branch
distribution at cell i keeps a known solution alive. The policy head learns
σ(π_i) ≈ p*_i, so at inference "branch at argmax σ(π)" means "branch where a
sampled commitment is most likely to survive". This subsumes the classical MRV
heuristic (fewer candidates → more mass per candidate) but can exploit
constraint structure MRV cannot see. Cells with < 2 candidates and ⊥-target
states are masked out (no meaningful branch exists there).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model import ColtOutput


@dataclass
class ColtLossBreakdown:
    total: torch.Tensor
    bce: torch.Tensor
    cls: torch.Tensor
    ce: torch.Tensor
    policy: torch.Tensor
    frac_bottom: float
    cls_acc: float

    def as_dict_for_logging(self) -> dict[str, float]:
        return {
            "loss_total": self.total.item(),
            "loss_bce": self.bce.item(),
            "loss_cls": self.cls.item(),
            "loss_ce": self.ce.item(),
            "loss_policy": self.policy.item(),
            "frac_bottom": self.frac_bottom,
            "cls_acc": self.cls_acc,
        }


def asymmetric_bce(cand_logits: torch.Tensor, y_hat: torch.Tensor, w_plus: float, w_minus: float) -> torch.Tensor:
    t = y_hat.unsqueeze(1).expand_as(cand_logits)
    weight = w_plus * t + w_minus * (1.0 - t)
    return F.binary_cross_entropy_with_logits(cand_logits, t, weight=weight, reduction="mean")


def conflict_bce(cls_logits: torch.Tensor, target_is_bottom: torch.Tensor) -> torch.Tensor:
    t = target_is_bottom.float().unsqueeze(1).expand_as(cls_logits)
    return F.binary_cross_entropy_with_logits(cls_logits, t, reduction="mean")


def singleton_ce(cand_logits: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    B, L, C, V = cand_logits.shape
    singleton = y_hat.sum(dim=-1) == 1.0
    target_cls = y_hat.argmax(dim=-1)
    mask = singleton.unsqueeze(1).expand(B, L, C).reshape(-1)
    if mask.sum() == 0:
        return cand_logits.new_zeros(())
    logits_flat = cand_logits.reshape(B * L * C, V)
    tgt_flat = target_cls.unsqueeze(1).expand(B, L, C).reshape(-1)
    ce = F.cross_entropy(logits_flat, tgt_flat, reduction="none")
    return (ce * mask).sum() / mask.sum()


def policy_target(
    x: torch.Tensor,
    branch_logits: torch.Tensor,
    y_hat: torch.Tensor,
    tau_decide: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Branch-survival probability per cell + validity mask.

    x:             (B, C, V) lattice state (defines alive candidates).
    branch_logits: (B, C, V) the logits branching actually samples from
                   (iteration-mean at training, per Algorithm 2).
    y_hat:         (B, C, V) abstraction target (x ⊓ α of surviving solutions).

    Returns (p_star, mask) of shapes (B, C): p*_i as defined in the module
    docstring, and mask = cells with ≥ 2 alive candidates in non-⊥ rows.
    """
    alive = x > 0.5
    masked = (branch_logits / tau_decide).masked_fill(~alive, float("-inf"))
    probs = torch.softmax(masked, dim=-1)                  # (B, C, V)
    probs = torch.nan_to_num(probs, nan=0.0)               # rows with no alive cand
    p_star = (probs * (y_hat > 0.5).float()).sum(dim=-1)   # (B, C)
    multi = alive.sum(dim=-1) >= 2                          # (B, C)
    consistent = (y_hat.sum(dim=(1, 2)) > 0).unsqueeze(1)   # (B, 1) — non-⊥ rows
    return p_star, multi & consistent


def policy_bce(
    policy_logits: torch.Tensor,
    p_star: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """BCE(σ(π), p*) over masked cells, every iteration. policy_logits: (B, L, C)."""
    B, L, C = policy_logits.shape
    t = p_star.unsqueeze(1).expand(B, L, C)
    m = mask.unsqueeze(1).expand(B, L, C).float()
    if m.sum() == 0:
        return policy_logits.new_zeros(())
    loss = F.binary_cross_entropy_with_logits(policy_logits, t.clamp(0.0, 1.0), reduction="none")
    return (loss * m).sum() / m.sum()


def compute_colt_loss(
    output: ColtOutput,
    x: torch.Tensor,
    y_hat: torch.Tensor,
    target_is_bottom: torch.Tensor,
    w_plus: float = 4.0,
    w_minus: float = 0.5,
    lambda_cls: float = 0.1,
    lambda_ce: float = 0.2,
    lambda_policy: float = 0.25,
    tau_decide: float = 1.5,
) -> ColtLossBreakdown:
    bce = asymmetric_bce(output.cand_logits, y_hat, w_plus, w_minus)
    cls = conflict_bce(output.cls_logits, target_is_bottom)
    ce = singleton_ce(output.cand_logits, y_hat)
    p_star, mask = policy_target(x, output.mean_cand().detach(), y_hat, tau_decide)
    pol = policy_bce(output.policy_logits, p_star, mask)
    total = bce + lambda_cls * cls + lambda_ce * ce + lambda_policy * pol

    with torch.no_grad():
        frac_bottom = target_is_bottom.float().mean().item()
        cls_pred = torch.sigmoid(output.cls_logits[:, -1]) > 0.5
        cls_acc = (cls_pred == target_is_bottom).float().mean().item()

    return ColtLossBreakdown(
        total=total, bce=bce, cls=cls, ce=ce, policy=pol,
        frac_bottom=frac_bottom, cls_acc=cls_acc,
    )
