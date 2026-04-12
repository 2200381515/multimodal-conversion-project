from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from torch.utils.data import DataLoader

from pipeline.dataset import PatientT0Dataset
from models.tmc.mm_transformer_tmc import MMTransformerTMC


def load_yaml(path: str) -> Dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


@torch.no_grad()
def predict_on_dataset(model, loader, device):
    model.eval()
    rows = []

    for batch in loader:
        batch = batch_to_device(batch, device)
        out = model(
            x_scale=batch["x_scale"],
            x_sc=batch["x_sc"],
            x_fc=batch["x_fc"],
            modality_mask=batch.get("modality_mask", None),
        )

        y = batch["y"].detach().cpu().numpy()
        probs = out.probs.detach().cpu().numpy()
        preds = (probs >= 0.5).astype(int)

        fused_u = out.fused_uncertainty
        if fused_u.ndim == 2:
            fused_u = fused_u.squeeze(-1)
        fused_u = fused_u.detach().cpu().numpy()

        conflict = out.conflict.detach().cpu().numpy()

        metas = batch.get("meta", None)
        if isinstance(metas, list):
            batch_subject_ids = [m.get("subject_id", "") for m in metas]
        else:
            batch_subject_ids = [""] * len(y)

        for i in range(len(y)):
            row = {
                "subject_id": batch_subject_ids[i],
                "y_true": int(y[i]),
                "y_prob": float(probs[i]),
                "y_pred": int(preds[i]),
                "uncertainty": float(fused_u[i]),
                "conflict": float(conflict[i]),
            }
            for name, vp in out.view_probs.items():
                row[f"view_prob_{name}"] = float(vp.detach().cpu().numpy()[i, 1])
            for name, vu in out.view_uncertainties.items():
                vu_np = vu.detach().cpu().numpy()
                if vu_np.ndim == 2:
                    vu_np = vu_np.squeeze(-1)
                row[f"view_u_{name}"] = float(vu_np[i])
            rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml",
        type=str,
        default="configs/mm_transformer_tmc/transformer_tmc.yaml",
    )
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.yaml)
    device = torch.device(args.device or cfg["train"]["device"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = PatientT0Dataset(
        cohort_csv=cfg["data"]["cohort_csv"],
        require_eligible=False,
        drop_qc_excluded=False,
        fill_missing=True,
        sc_norm="zscore_global",
        fc_norm="zscore_global",
        scale_norm_stats=None,
        drop_scale_cols=["DATASET-DIAG2"],
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    model = MMTransformerTMC(
        backbone_kwargs=cfg["model"]["backbone_kwargs"],
        n_classes=2,
        evidence_hidden_dim=cfg["model"]["evidence_hidden_dim"],
        evidence_dropout=cfg["model"]["evidence_dropout"],
        use_fused_feature_view=cfg["model"]["use_fused_feature_view"],
    ).to(device)

    state = torch.load(args.ckpt, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=True)
    else:
        model.load_state_dict(state, strict=True)

    pred_df = predict_on_dataset(model, loader, device)
    out_csv = out_dir / "predictions.csv"
    pred_df.to_csv(out_csv, index=False)
    print(f"Predictions saved to {out_csv}")


if __name__ == "__main__":
    main()