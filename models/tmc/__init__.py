"""
TMC (Trusted Multi-view Classification) package.

Day1:
- EvidenceHead: map pooled features -> evidence -> Dirichlet alpha -> (belief, uncertainty)
- Subjective Logic utilities: alpha <-> evidence <-> opinion conversions
- Modality output protocol (markdown) for multi-modal alignment

Day2:
- Reduced DS fusion rules and multi-view fusion API
- Missing view handling utilities
"""
from .evidence_head import EvidenceHead
from .subjective_logic_utils import (
    logits_to_evidence,
    evidence_to_alpha,
    alpha_to_opinion,
    dirichlet_mean,
    dirichlet_strength,
    check_no_nan_inf,
)
from .ds_fusion import ds_combine_two, ds_combine_many
from .missing_view_handler import apply_missing_mask_to_alpha, apply_missing_mask_to_opinion
from .tmc_fusion import fuse_alphas_tmc
from .losses_tmc import compute_tmc_multitask_loss
from .tmc_model_minimal import TMCMinimalModel