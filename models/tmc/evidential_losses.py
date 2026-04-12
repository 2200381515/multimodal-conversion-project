from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass
class LossOutput:
    total: torch.Tensor
    fused_bce: torch.Tensor
    view_bce: torch.Tensor
    evidence_reg: torch.Tensor


def binary_bce_from_prob(
    prob_pos: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    """
    prob_pos: [B]
    target: [B] int {0,1}
    """
    target = target.float()
    prob_pos = prob_pos.clamp(1e-6, 1 - 1e-6)

    weight = torch.ones_like(target)
    weight[target > 0.5] = pos_weight

    loss = F.binary_cross_entropy(prob_pos, target, reduction="none")
    loss = loss * weight
    return loss.mean()


def evidence_regularizer(
    evidences: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    简单版 evidence 正则：
    避免 evidence 无限放大
    """
    reg = 0.0
    count = 0
    for e in evidences.values():
        reg = reg + e.mean()
        count += 1
    if count == 0:
        return torch.tensor(0.0, device=next(iter(evidences.values())).device)
    return reg / count


def compute_tmc_loss(
    fused_prob: torch.Tensor,
    view_probs: Dict[str, torch.Tensor],
    view_evidences: Dict[str, torch.Tensor],
    target: torch.Tensor,
    pos_weight: float = 1.0,
    lambda_view: float = 0.5,
    lambda_evidence: float = 1e-4,
) -> LossOutput:
    fused_bce = binary_bce_from_prob(
        prob_pos=fused_prob,
        target=target,
        pos_weight=pos_weight,
    )

    view_losses = []
    for _, p in view_probs.items():
        # p shape [B, 2]
        view_losses.append(
            binary_bce_from_prob(
                prob_pos=p[:, 1],
                target=target,
                pos_weight=pos_weight,
            )
        )

    if len(view_losses) > 0:
        view_bce = torch.stack(view_losses).mean()
    else:
        view_bce = fused_bce.new_tensor(0.0)

    e_reg = evidence_regularizer(view_evidences)

    total = fused_bce + lambda_view * view_bce + lambda_evidence * e_reg

    return LossOutput(
        total=total,
        fused_bce=fused_bce,
        view_bce=view_bce,
        evidence_reg=e_reg,
    )