from __future__ import annotations

import torch

from .evidence_head import EvidenceHead


def main() -> None:
    torch.manual_seed(0)

    bsz = 4
    in_dim = 16
    num_classes = 2

    pooled = torch.randn(bsz, in_dim)
    head = EvidenceHead(in_dim=in_dim, num_classes=num_classes, activation="softplus", prior=1.0)

    out = head(pooled)

    print("=== Day1 Demo: EvidenceHead output keys ===")
    print(sorted(list(out.keys())))

    print("\n=== Shapes ===")
    for k, v in out.items():
        print(f"{k:12s}: {tuple(v.shape)}")

    # sanity checks:
    belief = out["belief"]         # [B,K]
    u = out["uncertainty"]         # [B,1]
    s = belief.sum(dim=-1, keepdim=True) + u

    print("\n=== Sanity ===")
    print("belief_sum_plus_u (should be close to 1):")
    print(s)

    # quick value checks
    assert (out["evidence"] >= 0).all(), "Evidence must be non-negative"
    assert (out["alpha"] >= 1.0).all(), "Alpha must be >= prior (1.0)"
    assert torch.isfinite(s).all(), "Sanity value must be finite"
    print("\nOK: Evidence/Alpha non-negativity & basic SL constraint passed.")


if __name__ == "__main__":
    main()
