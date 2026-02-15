from __future__ import annotations

import torch

from .evidence_head import EvidenceHead
from .tmc_fusion import fuse_alphas_tmc
from .subjective_logic_utils import evidence_to_alpha, logits_to_evidence


def _make_alpha_from_logits(logits: torch.Tensor, prior: float = 1.0):
    e = logits_to_evidence(logits, activation="softplus")
    return evidence_to_alpha(e, prior=prior)


def main() -> None:
    torch.manual_seed(0)
    B = 6
    K = 2
    prior = 1.0

    # Simulate 3 views (scale, sc, fc)
    # View0 (good): confident towards class 1
    logits0 = torch.tensor([[ -1.0,  3.0]]).repeat(B, 1) + 0.1 * torch.randn(B, K)

    # View1 (noisy/bad): random logits -> higher uncertainty
    logits1 = 0.2 * torch.randn(B, K)

    # View2 (conflicting): confident towards class 0 (conflicts with view0)
    logits2 = torch.tensor([[  3.0, -1.0]]).repeat(B, 1) + 0.1 * torch.randn(B, K)

    alpha0 = _make_alpha_from_logits(logits0, prior=prior)
    alpha1 = _make_alpha_from_logits(logits1, prior=prior)
    alpha2 = _make_alpha_from_logits(logits2, prior=prior)

    # Case A: all present
    res_all = fuse_alphas_tmc([alpha0, alpha1, alpha2], modality_masks=None, prior=prior)
    print("=== Day2 Demo: All present (3 views) ===")
    print("per-view uncertainty mean:", res_all["per_view_uncertainty"].mean(dim=(1, 0)).squeeze(-1))
    print("fused uncertainty mean:", res_all["fused_uncertainty"].mean().item())
    print("conflict mean per step:", res_all["conflicts"].mean(dim=(1, 0)).squeeze(-1))

    # Case B: view2 missing -> should reduce conflict, increase reliance on view0+view1
    mask0 = torch.ones(B, dtype=torch.long)
    mask1 = torch.ones(B, dtype=torch.long)
    mask2 = torch.zeros(B, dtype=torch.long)  # missing
    res_miss = fuse_alphas_tmc([alpha0, alpha1, alpha2], modality_masks=[mask0, mask1, mask2], prior=prior)
    print("\n=== Day2 Demo: View2 missing ===")
    print("per-view uncertainty mean:", res_miss["per_view_uncertainty"].mean(dim=(1, 0)).squeeze(-1))
    print("fused uncertainty mean:", res_miss["fused_uncertainty"].mean().item())
    print("conflict mean per step:", res_miss["conflicts"].mean(dim=(1, 0)).squeeze(-1))

    # Quick constraint check: sum(b)+u approx 1
    fused_sum = res_all["fused_belief"].sum(dim=-1, keepdim=True) + res_all["fused_uncertainty"]
    print("\nSanity fused sum(b)+u (should be ~1):", fused_sum[:3].squeeze(-1))


if __name__ == "__main__":
    main()
