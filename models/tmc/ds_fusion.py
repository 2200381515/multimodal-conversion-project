from __future__ import annotations

from typing import Tuple

import torch

from .subjective_logic_utils import check_no_nan_inf


def _normalize_opinion(belief: torch.Tensor, uncertainty: torch.Tensor, eps: float = 1e-12):
    """
    Ensure opinion is numerically stable and roughly satisfies sum(b)+u=1.

    belief: [B,K]
    uncertainty: [B,1] or [B]
    """
    if uncertainty.dim() == 1:
        uncertainty = uncertainty.unsqueeze(-1)
    if belief.dim() != 2 or uncertainty.dim() != 2:
        raise ValueError(f"belief must be [B,K], u must be [B,1]; got {belief.shape}, {uncertainty.shape}")

    belief = belief.clamp_min(0.0)
    uncertainty = uncertainty.clamp(0.0, 1.0)

    s = belief.sum(dim=-1, keepdim=True) + uncertainty
    # Avoid division by zero; if s deviates, renormalize gently.
    s = s.clamp_min(eps)
    belief = belief / s
    uncertainty = uncertainty / s

    check_no_nan_inf(belief, "belief(norm)")
    check_no_nan_inf(uncertainty, "uncertainty(norm)")
    return belief, uncertainty


def ds_combine_two(
    b1: torch.Tensor,
    u1: torch.Tensor,
    b2: torch.Tensor,
    u2: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reduced Dempster's rule for K-class opinions (TMC paper).
    Combine two opinions (b1,u1) and (b2,u2).

    Inputs:
      b1,b2: [B,K]
      u1,u2: [B,1] or [B]
    Returns:
      b: [B,K]
      u: [B,1]
      C: [B,1] conflict mass

    Formulas (Reduced DS rule):
      C = sum_{i!=j} b1_i * b2_j
      b_k = (b1_k*b2_k + b1_k*u2 + b2_k*u1) / (1 - C)
      u   = (u1*u2) / (1 - C)
    """
    if b1.shape != b2.shape:
        raise ValueError(f"b1 and b2 shape mismatch: {b1.shape} vs {b2.shape}")
    if b1.dim() != 2:
        raise ValueError(f"belief must be [B,K], got {b1.shape}")

    if u1.dim() == 1:
        u1 = u1.unsqueeze(-1)
    if u2.dim() == 1:
        u2 = u2.unsqueeze(-1)

    # Normalize inputs (robustness)
    b1, u1 = _normalize_opinion(b1, u1, eps=eps)
    b2, u2 = _normalize_opinion(b2, u2, eps=eps)

    # Conflict: C = sum_{i!=j} b1_i b2_j = (sum_i b1_i)(sum_j b2_j) - sum_k b1_k b2_k
    bb = (b1 * b2).sum(dim=-1, keepdim=True)  # [B,1]
    s1 = b1.sum(dim=-1, keepdim=True)         # [B,1]
    s2 = b2.sum(dim=-1, keepdim=True)         # [B,1]
    C = (s1 * s2 - bb).clamp_min(0.0)         # [B,1], keep non-negative
    check_no_nan_inf(C, "conflict C")

    denom = (1.0 - C).clamp_min(eps)          # avoid division by zero
    b = (b1 * b2 + b1 * u2 + b2 * u1) / denom
    u = (u1 * u2) / denom

    # Re-normalize final opinion for numerical stability
    b, u = _normalize_opinion(b, u, eps=eps)
    return b, u, C


def ds_combine_many(
    beliefs: torch.Tensor,
    uncertainties: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Iteratively combine M views.

    beliefs: [M,B,K]
    uncertainties: [M,B,1] or [M,B]
    Returns:
      fused_b: [B,K]
      fused_u: [B,1]
      conflicts: [M-1,B,1] conflict mass at each combine step
    """
    if beliefs.dim() != 3:
        raise ValueError(f"beliefs must be [M,B,K], got {beliefs.shape}")
    M, B, K = beliefs.shape

    if uncertainties.dim() == 2:
        uncertainties = uncertainties.unsqueeze(-1)
    if uncertainties.shape[0] != M or uncertainties.shape[1] != B:
        raise ValueError(f"uncertainties must be [M,B,1], got {uncertainties.shape}")

    fused_b = beliefs[0]
    fused_u = uncertainties[0]
    conflicts = []

    for m in range(1, M):
        fused_b, fused_u, C = ds_combine_two(fused_b, fused_u, beliefs[m], uncertainties[m], eps=eps)
        conflicts.append(C)

    if len(conflicts) == 0:
        conflicts_t = torch.zeros((0, B, 1), device=beliefs.device, dtype=beliefs.dtype)
    else:
        conflicts_t = torch.stack(conflicts, dim=0)

    return fused_b, fused_u, conflicts_t
