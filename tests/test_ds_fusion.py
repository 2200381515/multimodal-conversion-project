import torch
from models.tmc.ds_fusion import ds_combine_two

def test_ds_fusion_sum_constraint():
    torch.manual_seed(0)
    B, K = 8, 3

    # random opinions
    b1 = torch.rand(B, K)
    u1 = torch.rand(B, 1) * 0.5
    # normalize roughly
    s1 = b1.sum(dim=-1, keepdim=True) + u1
    b1 = b1 / s1
    u1 = u1 / s1

    b2 = torch.rand(B, K)
    u2 = torch.rand(B, 1) * 0.5
    s2 = b2.sum(dim=-1, keepdim=True) + u2
    b2 = b2 / s2
    u2 = u2 / s2

    b, u, C = ds_combine_two(b1, u1, b2, u2)
    s = b.sum(dim=-1, keepdim=True) + u
    assert torch.allclose(s, torch.ones_like(s), atol=1e-4)
    assert torch.isfinite(C).all()
