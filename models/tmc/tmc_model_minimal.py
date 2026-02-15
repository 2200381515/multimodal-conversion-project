from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .evidence_head import EvidenceHead
from .subjective_logic_utils import dirichlet_mean
from .tmc_fusion import fuse_alphas_tmc


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TMCForwardOutput:
    fused_prob: torch.Tensor                 # [B,K]
    fused_uncertainty: torch.Tensor          # [B,1]
    view_probs: torch.Tensor                 # [M,B,K]
    view_uncertainties: torch.Tensor         # [M,B,1]
    view_evidences: torch.Tensor             # [M,B,K]
    view_alphas: torch.Tensor                # [M,B,K]
    conflicts: torch.Tensor                  # [M-1,B,1]


class TMCMinimalModel(nn.Module):
    """
    Minimal trainable TMC model:
      - Encode each modality -> pooled feature
      - EvidenceHead -> evidence/alpha/prob/belief/u
      - Fuse alphas using DS rule
      - Output fused_prob + uncertainties
    """

    def __init__(
        self,
        scale_dim: int,
        sc_dim: int,
        fc_dim: int,
        embed_dim: int = 128,
        hidden: int = 256,
        num_classes: int = 2,
        prior: float = 1.0,
        evidence_activation: str = "softplus",
        evidence_clamp_max: Optional[float] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.prior = prior

        self.scale_enc = MLPEncoder(scale_dim, hidden, embed_dim, dropout=dropout)
        self.sc_enc = MLPEncoder(sc_dim, hidden, embed_dim, dropout=dropout)
        self.fc_enc = MLPEncoder(fc_dim, hidden, embed_dim, dropout=dropout)

        self.head_scale = EvidenceHead(embed_dim, num_classes, activation=evidence_activation,
                                       prior=prior, evidence_clamp_max=evidence_clamp_max, dropout=0.0)
        self.head_sc = EvidenceHead(embed_dim, num_classes, activation=evidence_activation,
                                    prior=prior, evidence_clamp_max=evidence_clamp_max, dropout=0.0)
        self.head_fc = EvidenceHead(embed_dim, num_classes, activation=evidence_activation,
                                    prior=prior, evidence_clamp_max=evidence_clamp_max, dropout=0.0)

    @staticmethod
    def _flatten_matrix(x: torch.Tensor) -> torch.Tensor:
        # x: [B,N,N] -> [B, N*N]
        if x.dim() != 3:
            raise ValueError(f"Expected [B,N,N], got {x.shape}")
        return x.reshape(x.size(0), -1)

    def forward(
        self,
        x_scale: torch.Tensor,                # [B,scale_dim]
        x_sc: torch.Tensor,                   # [B,N,N]
        x_fc: torch.Tensor,                   # [B,N,N]
        modality_masks: Optional[List[torch.Tensor]] = None,  # 3 x [B]
    ) -> TMCForwardOutput:

        sc_flat = self._flatten_matrix(x_sc)
        fc_flat = self._flatten_matrix(x_fc)

        z_scale = self.scale_enc(x_scale)
        z_sc = self.sc_enc(sc_flat)
        z_fc = self.fc_enc(fc_flat)

        o_scale = self.head_scale(z_scale)
        o_sc = self.head_sc(z_sc)
        o_fc = self.head_fc(z_fc)

        alphas = [o_scale["alpha"], o_sc["alpha"], o_fc["alpha"]]
        res = fuse_alphas_tmc(alphas, modality_masks=modality_masks, prior=self.prior)

        # fused belief+u -> fused prob
        # We can convert fused belief+u back to a pseudo alpha is optional; here simply use:
        # prob ≈ fused belief + u * base_rate (uniform)
        fused_b = res["fused_belief"]          # [B,K]
        fused_u = res["fused_uncertainty"]     # [B,1]
        base_rate = torch.full_like(fused_b, 1.0 / fused_b.size(-1))
        fused_prob = fused_b + fused_u * base_rate

        view_probs = torch.stack([o_scale["prob"], o_sc["prob"], o_fc["prob"]], dim=0)
        view_uncertainties = torch.stack([o_scale["uncertainty"], o_sc["uncertainty"], o_fc["uncertainty"]], dim=0)
        view_evidences = torch.stack([o_scale["evidence"], o_sc["evidence"], o_fc["evidence"]], dim=0)
        view_alphas = torch.stack(alphas, dim=0)

        return TMCForwardOutput(
            fused_prob=fused_prob,
            fused_uncertainty=fused_u,
            view_probs=view_probs,
            view_uncertainties=view_uncertainties,
            view_evidences=view_evidences,
            view_alphas=view_alphas,
            conflicts=res["conflicts"],
        )
