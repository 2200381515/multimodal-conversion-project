from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Opinion:
    """
    Subjective Logic opinion for a K-class classification:
    - belief: [B, K]
    - uncertainty: [B, 1] or [B]
    - strength: [B, 1] or [B]  (Dirichlet strength S = sum(alpha))
    - alpha: [B, K]
    - evidence: [B, K]
    - prob: [B, K] (Dirichlet mean)
    """
    belief: torch.Tensor
    uncertainty: torch.Tensor
    strength: torch.Tensor
    alpha: torch.Tensor
    evidence: torch.Tensor
    prob: torch.Tensor


def check_no_nan_inf(x: torch.Tensor, name: str) -> None:
    if not torch.isfinite(x).all():
        raise ValueError(f"[TMC] Tensor '{name}' contains NaN or Inf. "
                         f"min={x.min().item():.4g}, max={x.max().item():.4g}")


def dirichlet_strength(alpha: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    """
    S = sum_k alpha_k
    alpha: [B, K]
    """
    return alpha.sum(dim=-1, keepdim=keepdim)


def dirichlet_mean(alpha: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Dirichlet mean: p_k = alpha_k / sum_j alpha_j
    """
    s = dirichlet_strength(alpha, keepdim=True).clamp_min(eps)
    return alpha / s


def logits_to_evidence(
    logits: torch.Tensor,
    activation: str = "softplus",
    clamp_max: Optional[float] = None,
) -> torch.Tensor:
    """
    Map logits -> non-negative evidence.

    activation:
      - "relu": evidence = relu(logits)
      - "softplus": evidence = softplus(logits)  (smoother, avoids dead units)
      - "exp": evidence = exp(logits)  (can explode; use with care)

    clamp_max: optionally clamp evidence to avoid numerical explosion.
    """
    if activation == "relu":
        evidence = F.relu(logits)
    elif activation == "softplus":
        evidence = F.softplus(logits)
    elif activation == "exp":
        evidence = torch.exp(logits)
    else:
        raise ValueError(f"Unknown activation='{activation}'. Use relu/softplus/exp.")

    if clamp_max is not None:
        evidence = evidence.clamp_max(clamp_max)

    check_no_nan_inf(evidence, "evidence")
    return evidence


def evidence_to_alpha(
    evidence: torch.Tensor,
    prior: float = 1.0,
) -> torch.Tensor:
    """
    alpha = evidence + prior
    prior default 1.0 -> uniform base rate.
    """
    if prior <= 0:
        raise ValueError("prior must be > 0.")

    alpha = evidence + prior
    check_no_nan_inf(alpha, "alpha")
    return alpha


def alpha_to_opinion(
    alpha: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert Dirichlet alpha -> (belief, uncertainty, strength) following Subjective Logic.

    For K classes:
      S = sum(alpha)
      evidence e = alpha - 1
      belief b_k = e_k / S
      uncertainty u = K / S

    Note: This is the common form used in EDL / TMC style papers.
    """
    if alpha.dim() != 2:
        raise ValueError(f"alpha must be 2D [B,K], got shape={tuple(alpha.shape)}")

    bsz, k = alpha.shape
    strength = dirichlet_strength(alpha, keepdim=True).clamp_min(eps)  # [B,1]
    evidence = (alpha - 1.0).clamp_min(0.0)  # ensure non-negative due to possible numeric issues

    belief = evidence / strength  # [B,K]
    uncertainty = (float(k) / strength).clamp(0.0, 1.0)  # [B,1]

    check_no_nan_inf(belief, "belief")
    check_no_nan_inf(uncertainty, "uncertainty")
    check_no_nan_inf(strength, "strength")
    return belief, uncertainty, strength
