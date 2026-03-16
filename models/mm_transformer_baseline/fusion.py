from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class _CrossAttentionUnit(nn.Module):
    """
    A small wrapper around nn.MultiheadAttention with batch_first=True.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out, weights = self.attn(
            q,
            kv,
            kv,
            need_weights=True,
            average_attn_weights=False,
        )
        return out, weights


class _TargetFusionBlock(nn.Module):
    """
    For one target modality:
      - query itself
      - query modality 2
      - query modality 3
    Then combine the three outputs with learned mixing weights.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn_self = _CrossAttentionUnit(embed_dim, num_heads, dropout)
        self.attn_m2 = _CrossAttentionUnit(embed_dim, num_heads, dropout)
        self.attn_m3 = _CrossAttentionUnit(embed_dim, num_heads, dropout)

        self.mix_logits = nn.Parameter(torch.zeros(3))

        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        target: torch.Tensor,
        src_self: torch.Tensor,
        src_2: torch.Tensor,
        src_3: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        o_self, w_self = self.attn_self(target, src_self)
        o_2, w_2 = self.attn_m2(target, src_2)
        o_3, w_3 = self.attn_m3(target, src_3)

        mix = torch.softmax(self.mix_logits, dim=0)
        fused = mix[0] * o_self + mix[1] * o_2 + mix[2] * o_3

        x = self.norm1(target + self.drop(fused))
        y = self.norm2(x + self.drop(self.ffn(x)))

        info = {
            "mix_weights": mix.detach(),
            "attn_self": w_self.detach(),
            "attn_cross_2": w_2.detach(),
            "attn_cross_3": w_3.detach(),
        }
        return y, info


class ExplicitCrossAttention3Way(nn.Module):
    """
    Three-modality explicit cross-attention block.

    This keeps the core idea borrowed from the old project:
    each target modality queries itself and the other two modalities,
    then combines the three outputs with learned target-specific weights.

    In this project:
        modality 1 -> scale
        modality 2 -> sc
        modality 3 -> fc
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.m1_block = _TargetFusionBlock(embed_dim, num_heads, dropout)
        self.m2_block = _TargetFusionBlock(embed_dim, num_heads, dropout)
        self.m3_block = _TargetFusionBlock(embed_dim, num_heads, dropout)

    def forward(
        self,
        m1: torch.Tensor,
        m2: torch.Tensor,
        m3: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]:
        o1, i1 = self.m1_block(m1, m1, m2, m3)
        o2, i2 = self.m2_block(m2, m2, m1, m3)
        o3, i3 = self.m3_block(m3, m3, m1, m2)

        attn_dict = {
            "scale": i1,
            "sc": i2,
            "fc": i3,
        }
        return o1, o2, o3, attn_dict