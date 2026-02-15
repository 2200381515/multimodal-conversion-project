from __future__ import annotations

from typing import Tuple

import torch

from .subjective_logic_utils import check_no_nan_inf


def apply_missing_mask_to_alpha(
    alpha: torch.Tensor,
    modality_mask: torch.Tensor,
    prior: float = 1.0,
) -> torch.Tensor:
    """
    If modality_mask[b]==0, replace alpha[b,:] with prior (i.e., evidence=0 -> alpha=prior),
    leading to high uncertainty.

    alpha: [B,K]
    modality_mask: [B] bool/int (1=present,0=missing)
    prior: >0, default 1.0
    """
    if alpha.dim() != 2:
        raise ValueError(f"alpha must be [B,K], got {alpha.shape}")
    if modality_mask.dim() != 1 or modality_mask.shape[0] != alpha.shape[0]:
        raise ValueError(f"modality_mask must be [B], got {modality_mask.shape}")

    mask = modality_mask.to(alpha.device).float().unsqueeze(-1)  # [B,1]
    # present -> keep alpha; missing -> set to prior
    out = alpha * mask + prior * (1.0 - mask)
    check_no_nan_inf(out, "alpha_after_missing_mask")
    return out


def apply_missing_mask_to_opinion(
    belief: torch.Tensor,
    uncertainty: torch.Tensor,
    modality_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    If missing, set belief=0 and uncertainty=1.
    """
    if belief.dim() != 2:
        raise ValueError(f"belief must be [B,K], got {belief.shape}")
    if uncertainty.dim() == 1:
        uncertainty = uncertainty.unsqueeze(-1)
    if uncertainty.shape[0] != belief.shape[0]:
        raise ValueError("belief and uncertainty batch mismatch")
    if modality_mask.dim() != 1 or modality_mask.shape[0] != belief.shape[0]:
        raise ValueError("modality_mask must be [B]")

    mask = modality_mask.to(belief.device).float().unsqueeze(-1)  # [B,1]
    belief_out = belief * mask
    u_out = uncertainty * mask + (1.0 - mask)  # missing -> 1
    check_no_nan_inf(belief_out, "belief_after_missing_mask")
    check_no_nan_inf(u_out, "uncertainty_after_missing_mask")
    return belief_out, u_out
