from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tmc.evidence_head import EvidenceHead
from models.tmc.ds_fusion import ds_combine_two
from models.tmc.subjective_logic_utils import alpha_to_opinion  # 导入 alpha_to_opinion

# 这里尽量兼容你现有 baseline backbone 的类名
try:
    from models.mm_transformer_baseline.multimodal_transformer import MultimodalTransformer
except ImportError:
    try:
        from models.mm_transformer_baseline.multimodal_transformer import MultimodalTransformerBaseline as MultimodalTransformer
    except ImportError as e:
        raise ImportError(
            "Cannot import baseline backbone. "
            "Please check your class name in "
            "models/mm_transformer_baseline/multimodal_transformer.py"
        ) from e


@dataclass
class TMCOutput:
    logits: torch.Tensor
    probs: torch.Tensor
    preds: torch.Tensor

    fused_belief: torch.Tensor
    fused_uncertainty: torch.Tensor
    fused_alpha: torch.Tensor
    conflict: torch.Tensor

    view_probs: Dict[str, torch.Tensor]
    view_uncertainties: Dict[str, torch.Tensor]
    view_beliefs: Dict[str, torch.Tensor]
    view_alphas: Dict[str, torch.Tensor]
    view_evidences: Dict[str, torch.Tensor]

    reprs: Dict[str, torch.Tensor]
    backbone_outputs: Dict[str, Any]


class MMTransformerTMC(nn.Module):
    """
    正式版：baseline backbone + TMC head
    流程：
      x_scale/x_sc/x_fc
        -> baseline backbone
        -> scale_repr/sc_repr/fc_repr
        -> per-view EvidenceHead
        -> DS fusion
        -> fused prob / uncertainty
    """

    def __init__(
        self,
        backbone_kwargs: Dict[str, Any],
        n_classes: int = 2,
        evidence_hidden_dim: int = 128,
        evidence_dropout: float = 0.1,
        use_fused_feature_view: bool = False,
    ) -> None:
        super().__init__()

        if n_classes != 2:
            raise ValueError("Current implementation is for binary classification (n_classes=2).")

        self.n_classes = n_classes
        self.use_fused_feature_view = use_fused_feature_view

        self.backbone = MultimodalTransformer(**backbone_kwargs)

        embed_dim = backbone_kwargs.get("embed_dim", None)
        if embed_dim is None:
            raise ValueError("backbone_kwargs must contain embed_dim.")

        self.scale_head = EvidenceHead(
            in_dim=embed_dim,
            num_classes=n_classes,
            activation="softplus",
            prior=1.0,
            evidence_clamp_max=None,
            dropout=evidence_dropout,
        )
        self.sc_head = EvidenceHead(
            in_dim=embed_dim,
            num_classes=n_classes,
            activation="softplus",
            prior=1.0,
            evidence_clamp_max=None,
            dropout=evidence_dropout,
        )
        self.fc_head = EvidenceHead(
            in_dim=embed_dim,
            num_classes=n_classes,
            activation="softplus",
            prior=1.0,
            evidence_clamp_max=None,
            dropout=evidence_dropout,
        )

        if self.use_fused_feature_view:
            self.fused_view_head = EvidenceHead(
                in_dim=embed_dim,
                num_classes=n_classes,
                activation="softplus",
                prior=1.0,
                evidence_clamp_max=None,
                dropout=evidence_dropout,
            )
        else:
            self.fused_view_head = None

    @staticmethod
    def _safe_get_repr(backbone_out: Dict[str, Any], key: str) -> torch.Tensor:
        if key not in backbone_out:
            raise KeyError(
                f"Backbone output missing key: {key}. "
                f"Please ensure baseline backbone returns {key}."
            )
        return backbone_out[key]

    @staticmethod
    def _prob_from_belief_u(belief: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
        # uniform base rate
        k = belief.shape[-1]
        return belief + uncertainty / k

    @staticmethod
    def _alpha_from_belief_u(belief: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
        # S = K / u, alpha = e + 1, e = b * S
        eps = 1e-8
        k = belief.shape[-1]
        strength = k / uncertainty.clamp_min(eps)
        evidence = belief * strength
        return evidence + 1.0

    def _apply_missing_mask(
        self,
        alpha: torch.Tensor,
        present_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        present_mask: [B]  1=present, 0=missing
        missing 时退回 Dirichlet prior alpha=1
        """
        prior = torch.ones_like(alpha)
        present_mask = present_mask.float().unsqueeze(-1)
        return alpha * present_mask + prior * (1.0 - present_mask)

    def forward(
            self,
            x_scale: torch.Tensor,
            x_sc: torch.Tensor,
            x_fc: torch.Tensor,
            modality_mask: Optional[torch.Tensor] = None,
            pos_weight: Optional[torch.Tensor] = None,  # 允许传递pos_weight
    ) -> TMCOutput:
        backbone_out = self.backbone(
            x_scale=x_scale,
            x_sc=x_sc,
            x_fc=x_fc,
            modality_mask=modality_mask,
        )

        scale_repr = self._safe_get_repr(backbone_out, "scale_repr")
        sc_repr = self._safe_get_repr(backbone_out, "sc_repr")
        fc_repr = self._safe_get_repr(backbone_out, "fc_repr")

        reprs = {
            "scale": scale_repr,
            "sc": sc_repr,
            "fc": fc_repr,
        }

        if self.use_fused_feature_view and "fused_repr" in backbone_out:
            reprs["fused_feature"] = backbone_out["fused_repr"]

        scale_out = self.scale_head(scale_repr)
        sc_out = self.sc_head(sc_repr)
        fc_out = self.fc_head(fc_repr)

        view_evidences = {
            "scale": scale_out["evidence"],
            "sc": sc_out["evidence"],
            "fc": fc_out["evidence"],
        }
        view_alphas = {
            "scale": scale_out["alpha"],
            "sc": sc_out["alpha"],
            "fc": fc_out["alpha"],
        }
        view_probs = {
            "scale": scale_out["prob"],
            "sc": sc_out["prob"],
            "fc": fc_out["prob"],
        }

        if self.use_fused_feature_view and self.fused_view_head is not None:
            fused_out = self.fused_view_head(reprs["fused_feature"])
            view_evidences["fused_feature"] = fused_out["evidence"]
            view_alphas["fused_feature"] = fused_out["alpha"]
            view_probs["fused_feature"] = fused_out["prob"]

        if modality_mask is not None:
            # 约定: modality_mask shape [B, 3], 顺序 scale/sc/fc
            view_alphas["scale"] = self._apply_missing_mask(view_alphas["scale"], modality_mask[:, 0])
            view_alphas["sc"] = self._apply_missing_mask(view_alphas["sc"], modality_mask[:, 1])
            view_alphas["fc"] = self._apply_missing_mask(view_alphas["fc"], modality_mask[:, 2])

        view_beliefs = {}
        view_uncertainties = {}
        view_strengths = {}

        for name, alpha in view_alphas.items():
            b, u, s = alpha_to_opinion(alpha)  # 使用 alpha_to_opinion 重新计算 belief/uncertainty/strength
            view_beliefs[name] = b
            view_uncertainties[name] = u
            view_strengths[name] = s

        combine_order: List[str] = ["scale", "sc", "fc"]
        if self.use_fused_feature_view and "fused_feature" in view_alphas:
            combine_order.append("fused_feature")

        cur_b = view_beliefs[combine_order[0]]
        cur_u = view_uncertainties[combine_order[0]]
        conflict_terms = []

        for name in combine_order[1:]:
            nxt_b = view_beliefs[name]
            nxt_u = view_uncertainties[name]
            cur_b, cur_u, c = ds_combine_two(cur_b, cur_u, nxt_b, nxt_u)
            conflict_terms.append(c)

        fused_belief = cur_b
        fused_uncertainty = cur_u
        fused_alpha = self._alpha_from_belief_u(fused_belief, fused_uncertainty)
        fused_prob = self._prob_from_belief_u(fused_belief, fused_uncertainty)

        positive_prob = fused_prob[:, 1]
        logits = torch.logit(positive_prob.clamp(1e-6, 1 - 1e-6))
        preds = (positive_prob >= 0.5).long()

        if len(conflict_terms) == 0:
            conflict = torch.zeros_like(positive_prob)
        else:
            conflict = torch.stack(conflict_terms, dim=0).mean(dim=0).squeeze(-1)

        return TMCOutput(
            logits=logits,
            probs=positive_prob,
            preds=preds,
            fused_belief=fused_belief,
            fused_uncertainty=fused_uncertainty,
            fused_alpha=fused_alpha,
            conflict=conflict,
            view_probs=view_probs,
            view_uncertainties=view_uncertainties,
            view_beliefs=view_beliefs,
            view_alphas=view_alphas,
            view_evidences=view_evidences,
            reprs=reprs,
            backbone_outputs=backbone_out,
        )