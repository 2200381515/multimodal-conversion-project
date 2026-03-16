from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleEncoder(nn.Module):
    """
    Encode clinical scale features into one or more tokens.

    Input:
        x: [B, M]
    Output:
        tokens: [B, T, D]
    """

    def __init__(
        self,
        in_dim: int,
        embed_dim: int = 128,
        num_tokens: int = 1,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.embed_dim = int(embed_dim)

        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_tokens * self.embed_dim),
        )
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(
                f"ScaleEncoder expects input with shape [B, M], got {tuple(x.shape)}"
            )

        z = self.proj(x)  # [B, T*D]
        z = z.view(x.size(0), self.num_tokens, self.embed_dim)  # [B, T, D]
        z = self.norm(z)
        return z


class MatrixTokenEncoder(nn.Module):
    """
    Lightweight encoder for SC / FC matrices.

    Instead of flattening the full [N, N] matrix into an extremely large vector,
    we first apply adaptive average pooling to obtain a compact
    [num_tokens, pool_width] representation, and then project each pooled row
    into the transformer embedding space.

    Input:
        x: [B, N, N]
    Output:
        tokens: [B, T, D]
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_tokens: int = 8,
        pool_width: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.pool_width = int(pool_width)

        self.token_proj = nn.Sequential(
            nn.Linear(self.pool_width, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"MatrixTokenEncoder expects input with shape [B, N, N], got {tuple(x.shape)}"
            )

        # [B, N, N] -> [B, 1, N, N] -> adaptive pool -> [B, 1, T, W] -> [B, T, W]
        pooled = F.adaptive_avg_pool2d(
            x.unsqueeze(1),
            output_size=(self.num_tokens, self.pool_width),
        ).squeeze(1)

        # project each pooled row into embedding dim
        tokens = self.token_proj(pooled)  # [B, T, D]
        tokens = self.norm(tokens)
        return tokens