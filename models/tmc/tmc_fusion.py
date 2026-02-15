from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from .subjective_logic_utils import (
    alpha_to_opinion,
    check_no_nan_inf,
)
from .ds_fusion import ds_combine_many
from .missing_view_handler import apply_missing_mask_to_alpha


def fuse_alphas_tmc(
    alphas: List[torch.Tensor],
    modality_masks: Optional[List[torch.Tensor]] = None,
    prior: float = 1.0,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """
    High-level TMC fusion:
      inputs: list of alpha^m, each [B,K]
      optional modality_masks: list of [B] indicating missingness per view
      returns fused belief/uncertainty + per-view belief/uncertainty + conflicts

    Strategy:
      - if mask provided: for missing view, alpha -> prior (high uncertainty)
      - compute per-view (b,u,S)
      - DS combine across views iteratively
    """
    if len(alphas) == 0:
        raise ValueError("alphas list is empty")
    B, K = alphas[0].shape
    for a in alphas:
        if a.shape != (B, K):
            raise ValueError("All alphas must share shape [B,K]")

    if modality_masks is not None and len(modality_masks) != len(alphas):
        raise ValueError("modality_masks length must match alphas length")

    per_view_b = []
    per_view_u = []
    per_view_S = []
    used_alphas = []

    for i, alpha in enumerate(alphas):
        if modality_masks is not None:
            alpha = apply_missing_mask_to_alpha(alpha, modality_masks[i], prior=prior)
        used_alphas.append(alpha)

        b, u, S = alpha_to_opinion(alpha, eps=eps)
        per_view_b.append(b)
        per_view_u.append(u)
        per_view_S.append(S)

    beliefs = torch.stack(per_view_b, dim=0)          # [M,B,K]
    uncertainties = torch.stack(per_view_u, dim=0)    # [M,B,1]

    fused_b, fused_u, conflicts = ds_combine_many(beliefs, uncertainties, eps=eps)

    check_no_nan_inf(fused_b, "fused_belief")
    check_no_nan_inf(fused_u, "fused_uncertainty")

    return {
        "fused_belief": fused_b,
        "fused_uncertainty": fused_u,
        "per_view_belief": beliefs,
        "per_view_uncertainty": uncertainties,
        "per_view_strength": torch.stack(per_view_S, dim=0),  # [M,B,1]
        "conflicts": conflicts,                                # [M-1,B,1]
        "used_alphas": torch.stack(used_alphas, dim=0),        # [M,B,K]
    }
