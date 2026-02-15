from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn.functional as F

from .subjective_logic_utils import check_no_nan_inf, dirichlet_strength


@dataclass
class TMCLossOutput:
    total: torch.Tensor
    fused_ce: torch.Tensor
    view_ce_mean: torch.Tensor
    evidence_reg_mean: torch.Tensor
    strength_mean: torch.Tensor


def _ce_from_prob(prob: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    CE using Dirichlet mean prob (not logits).
    prob: [B,K] in (0,1) sum=1
    y: [B] int64
    """
    prob = prob.clamp_min(eps)
    logp = torch.log(prob)
    return F.nll_loss(logp, y, reduction="mean")


def evidence_l1_regularizer(evidence: torch.Tensor) -> torch.Tensor:
    """
    Simple evidence penalty to avoid exploding evidence (stabilizer).
    evidence: [B,K]
    """
    return evidence.mean()


def compute_tmc_multitask_loss(
    fused_prob: torch.Tensor,
    fused_evidence: Optional[torch.Tensor],
    view_probs: torch.Tensor,
    view_evidences: Optional[torch.Tensor],
    y: torch.Tensor,
    lambda_view: float = 1.0,
    lambda_fused: float = 1.0,
    lambda_evidence: float = 1e-3,
) -> TMCLossOutput:
    """
    Multi-task loss:
      L = lambda_fused * CE(fused_prob, y) + lambda_view * mean_m CE(view_prob_m, y)
          + lambda_evidence * mean_m evidence_reg(view_evidence_m) [optional]
    """
    if y.dim() != 1:
        y = y.view(-1)
    if fused_prob.dim() != 2:
        raise ValueError("fused_prob must be [B,K]")
    if view_probs.dim() != 3:
        raise ValueError("view_probs must be [M,B,K]")

    check_no_nan_inf(fused_prob, "fused_prob")
    check_no_nan_inf(view_probs, "view_probs")

    fused_ce = _ce_from_prob(fused_prob, y)

    # per-view CE
    M = view_probs.shape[0]
    view_ces = []
    for m in range(M):
        view_ces.append(_ce_from_prob(view_probs[m], y))
    view_ce_mean = torch.stack(view_ces).mean()

    evidence_reg_mean = torch.zeros_like(fused_ce)
    strength_mean = torch.zeros_like(fused_ce)

    if view_evidences is not None:
        if view_evidences.shape[:2] != view_probs.shape[:2]:
            raise ValueError("view_evidences must be [M,B,K]")
        regs = []
        strengths = []
        for m in range(M):
            regs.append(evidence_l1_regularizer(view_evidences[m]))
            # strength proxy: sum(alpha)=sum(evidence+1) = sum(evidence)+K
            strengths.append(view_evidences[m].sum(dim=-1).mean())
        evidence_reg_mean = torch.stack(regs).mean()
        strength_mean = torch.stack(strengths).mean()

    total = lambda_fused * fused_ce + lambda_view * view_ce_mean + lambda_evidence * evidence_reg_mean
    return TMCLossOutput(
        total=total,
        fused_ce=fused_ce,
        view_ce_mean=view_ce_mean,
        evidence_reg_mean=evidence_reg_mean,
        strength_mean=strength_mean,
    )
