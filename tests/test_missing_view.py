import torch
from models.tmc.missing_view_handler import apply_missing_mask_to_alpha
from models.tmc.subjective_logic_utils import alpha_to_opinion

def test_missing_alpha_prior_high_uncertainty():
    torch.manual_seed(0)
    B, K = 5, 2
    alpha = torch.ones(B, K) * 10.0
    mask = torch.tensor([1, 0, 1, 0, 1], dtype=torch.long)

    out = apply_missing_mask_to_alpha(alpha, mask, prior=1.0)
    b, u, S = alpha_to_opinion(out)

    # missing indices should have alpha=1 -> strength=K -> u=1
    missing_idx = (mask == 0).nonzero(as_tuple=False).squeeze(-1)
    assert torch.allclose(u[missing_idx], torch.ones_like(u[missing_idx]), atol=1e-5)
