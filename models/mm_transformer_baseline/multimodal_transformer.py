from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .encoders import MatrixTokenEncoder, ScaleEncoder
from .fusion import ExplicitCrossAttention3Way


class MultimodalTransformerBaseline(nn.Module):
    """
    Multimodal transformer baseline for dementia conversion prediction.

    Pipeline:
        Scale / SC / FC
            -> modality-specific token encoders
            -> explicit cross-attention fusion
            -> modality representations
            -> fused representation
            -> binary classifier

    The model returns modality-specific representations as well, so the same
    backbone can later be reused for baseline + TMC.
    """

    def __init__(
        self,
        scale_dim: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        scale_tokens: int = 1,
        matrix_tokens: int = 8,
        matrix_pool_width: int = 32,
        classifier_hidden: int = 128,
    ):
        super().__init__()

        self.scale_encoder = ScaleEncoder(
            in_dim=scale_dim,
            embed_dim=embed_dim,
            num_tokens=scale_tokens,
            hidden_dim=max(embed_dim * 2, classifier_hidden),
            dropout=dropout,
        )

        self.sc_encoder = MatrixTokenEncoder(
            embed_dim=embed_dim,
            num_tokens=matrix_tokens,
            pool_width=matrix_pool_width,
            dropout=dropout,
        )

        self.fc_encoder = MatrixTokenEncoder(
            embed_dim=embed_dim,
            num_tokens=matrix_tokens,
            pool_width=matrix_pool_width,
            dropout=dropout,
        )

        self.fusion = ExplicitCrossAttention3Way(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 4, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1),
        )

    @staticmethod
    def _masked_pool(tokens: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        """
        tokens:  [B, T, D]
        present: [B]
        return:  [B, D]
        """
        pooled = tokens.mean(dim=1)
        return pooled * present.unsqueeze(-1)

    def forward(
        self,
        x_scale: torch.Tensor,
        x_sc: torch.Tensor,
        x_fc: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor | Dict]:
        # 1) modality-specific tokenization
        scale_tokens = self.scale_encoder(x_scale)  # [B, Ts, D]
        sc_tokens = self.sc_encoder(x_sc)          # [B, Tm, D]
        fc_tokens = self.fc_encoder(x_fc)          # [B, Tm, D]

        # 2) explicit cross-attention fusion
        scale_tokens, sc_tokens, fc_tokens, attn_dict = self.fusion(
            scale_tokens,
            sc_tokens,
            fc_tokens,
        )

        # 3) mask handling
        if modality_mask is None:
            modality_mask = torch.ones(
                x_scale.size(0),
                3,
                device=x_scale.device,
                dtype=x_scale.dtype,
            )

        modality_mask = modality_mask.float()

        # 4) modality-level representations
        scale_repr = self._masked_pool(scale_tokens, modality_mask[:, 0])
        sc_repr = self._masked_pool(sc_tokens, modality_mask[:, 1])
        fc_repr = self._masked_pool(fc_tokens, modality_mask[:, 2])

        # 5) fused representation
        denom = modality_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        fused_repr = (scale_repr + sc_repr + fc_repr) / denom

        # 6) binary classification
        classifier_in = torch.cat(
            [scale_repr, sc_repr, fc_repr, fused_repr],
            dim=-1,
        )
        logits = self.classifier(classifier_in).squeeze(-1)
        prob = torch.sigmoid(logits)

        return {
            "logits": logits,
            "prob": prob,
            "scale_repr": scale_repr,
            "sc_repr": sc_repr,
            "fc_repr": fc_repr,
            "fused_repr": fused_repr,
            "attn_dict": attn_dict,
        }