from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .subjective_logic_utils import (
    logits_to_evidence,
    evidence_to_alpha,
    alpha_to_opinion,
    dirichlet_mean,
    check_no_nan_inf,
)


class EvidenceHead(nn.Module):
    """
    Evidence head for K-class classification.

    Input:
      pooled features: [B, D]
    Output:
      dict with keys:
        - logits: [B, K]
        - evidence: [B, K] (>=0)
        - alpha: [B, K] (>=prior)
        - prob: [B, K] Dirichlet mean
        - belief: [B, K]
        - uncertainty: [B, 1]
        - strength: [B, 1]
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        activation: str = "softplus",
        prior: float = 1.0,
        evidence_clamp_max: Optional[float] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError("in_dim must be > 0")
        if num_classes <= 1:
            raise ValueError("num_classes must be >= 2")
        if prior <= 0:
            raise ValueError("prior must be > 0")

        self.in_dim = in_dim
        self.num_classes = num_classes
        self.activation = activation
        self.prior = prior
        self.evidence_clamp_max = evidence_clamp_max

        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_dim, num_classes))
        self.proj = nn.Sequential(*layers)

    def forward(self, pooled: torch.Tensor) -> Dict[str, torch.Tensor]:
        if pooled.dim() != 2:
            raise ValueError(f"pooled must be 2D [B,D], got shape={tuple(pooled.shape)}")
        if pooled.size(-1) != self.in_dim:
            raise ValueError(
                f"pooled last dim mismatch: expected {self.in_dim}, got {pooled.size(-1)}"
            )

        logits = self.proj(pooled)  # [B,K]
        check_no_nan_inf(logits, "logits")

        evidence = logits_to_evidence(
            logits, activation=self.activation, clamp_max=self.evidence_clamp_max
        )
        alpha = evidence_to_alpha(evidence, prior=self.prior)
        prob = dirichlet_mean(alpha)

        belief, uncertainty, strength = alpha_to_opinion(alpha)

        out = {
            "logits": logits,
            "evidence": evidence,
            "alpha": alpha,
            "prob": prob,
            "belief": belief,
            "uncertainty": uncertainty,
            "strength": strength,
        }
        return out
