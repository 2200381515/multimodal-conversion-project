from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from pipeline.dataset import PatientT0Dataset
from models.tmc.mm_transformer_tmc import MMTransformerTMC


def load_yaml(path: str) -> Dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataset(cfg: Dict) -> PatientT0Dataset:
    return PatientT0Dataset(
        cohort_csv=cfg["data"]["cohort_csv"],
        require_eligible=False,
        drop_qc_excluded=False,
        fill_missing=True,
        sc_norm="zscore_global",
        fc_norm="zscore_global",
        scale_norm_stats=None,
        drop_scale_cols=["DATASET-DIAG2"],
    )


def infer_subject_ids(dataset: PatientT0Dataset) -> List[str]:
    subject_ids = []
    for i in range(len(dataset)):
        item = dataset[i]
        meta = item.get("meta", {})
        sid = meta.get("subject_id", None)
        if sid is None:
            raise KeyError("Dataset item meta missing subject_id")
        subject_ids.append(str(sid))
    return subject_ids


def attach_fold_ids(fold_index_csv: str) -> pd.DataFrame:
    fold_df = pd.read_csv(fold_index_csv)
    return fold_df[["subject_id", "test_fold_id"]].copy()


def batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


@torch.no_grad()
def eval_one_fold(model, loader, device):
    model.eval()

    y_true_all, y_prob_all, y_pred_all = [], [], []
    u_all, conflict_all = [], []
    rows = []

    for batch in loader:
        batch = batch_to_device(batch, device)
        out = model(
            x_scale=batch["x_scale"],
            x_sc=batch["x_sc"],
            x_fc=batch["x_fc"],
            modality_mask=batch.get("modality_mask", None),
        )

        y = batch["y"].long().detach().cpu().numpy()
        y_prob = out.probs.detach().cpu().numpy()
        y_pred = (y_prob >= 0.5).astype(int)

        fused_u = out.fused_uncertainty
        if fused_u.ndim == 2:
            fused_u = fused_u.squeeze(-1)
        fused_u = fused_u.detach().cpu().numpy()

        conflict = out.conflict.detach().cpu().numpy()

        metas = batch.get("meta", None)
        if isinstance(metas, list):
            batch_subject_ids = [m.get("subject_id", "") for m in metas]
        else:
            # fallback
            batch_subject_ids = [""] * len(y)

        for i in range(len(y)):
            row = {
                "subject_id": batch_subject_ids[i],
                "y_true": int(y[i]),
                "y_prob": float(y_prob[i]),
                "y_pred": int(y_pred[i]),
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

        y_true_all.extend(y.tolist())
        y_prob_all.extend(y_prob.tolist())
        y_pred_all.extend(y_pred.tolist())
        u_all.extend(fused_u.tolist())
        conflict_all.extend(conflict.tolist())

    y_true = np.asarray(y_true_all).astype(int)
    y_prob = np.asarray(y_prob_all).astype(float)
    y_pred = np.asarray(y_pred_all).astype(int)

    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        accuracy_score,
        f1_score,
        confusion_matrix,
    )

    auc = float("nan")
    ap = float("nan")
    if len(np.unique(y_true)) >= 2:
        auc = roc_auc_score(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "AUC": float(auc),
        "PR_AUC": float(ap),
        "F1": float(f1),
        "ACC": float(acc),
        "SEN": float(sen),
        "SPE": float(spe),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "mean_u": float(np.mean(u_all)) if len(u_all) else float("nan"),
        "u_p90": float(np.quantile(u_all, 0.9)) if len(u_all) else float("nan"),
        "mean_conflict": float(np.mean(conflict_all)) if len(conflict_all) else float("nan"),
    }

    pred_df = pd.DataFrame(rows)
    return metrics, pred_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml",
        type=str,
        default="configs/mm_transformer_tmc/transformer_tmc.yaml",
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.yaml)
    device = torch.device(args.device or cfg["train"]["device"])

    out_root = Path(cfg["train"]["out_dir"])
    eval_root = out_root / "eval_results"
    eval_root.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(cfg)
    subject_ids = infer_subject_ids(dataset)
    fold_df = attach_fold_ids(cfg["data"]["fold_index_csv"])

    sid_to_fold = {
        str(r["subject_id"]): int(r["test_fold_id"])
        for _, r in fold_df.iterrows()
    }
    test_fold_ids = [sid_to_fold[str(sid)] for sid in subject_ids]

    all_metrics = []
    all_pred_dfs = []

    for fold_id in sorted(set(test_fold_ids)):
        ckpt_path = out_root / f"fold_{fold_id}" / "model.pt"
        if not ckpt_path.exists():
            print(f"[WARN] missing checkpoint for fold {fold_id}: {ckpt_path}")
            continue

        model = MMTransformerTMC(
            backbone_kwargs=cfg["model"]["backbone_kwargs"],
            n_classes=2,
            evidence_hidden_dim=cfg["model"]["evidence_hidden_dim"],
            evidence_dropout=cfg["model"]["evidence_dropout"],
            use_fused_feature_view=cfg["model"]["use_fused_feature_view"],
        ).to(device)

        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state_dict"], strict=True)

        test_idx = [i for i, f in enumerate(test_fold_ids) if f == fold_id]
        loader = DataLoader(
            Subset(dataset, test_idx),
            batch_size=cfg["train"]["batch_size"],
            shuffle=False,
            num_workers=0,
        )

        metrics, pred_df = eval_one_fold(model, loader, device)
        metrics["fold_id"] = fold_id

        fold_dir = eval_root / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        pred_df.to_csv(fold_dir / "predictions.csv", index=False)
        with open(fold_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        print(
            f"[Eval] fold={fold_id} | "
            f"AUC={metrics['AUC']:.4f} | "
            f"F1={metrics['F1']:.4f} | "
            f"mean_u={metrics['mean_u']:.4f} | "
            f"u(p90)={metrics['u_p90']:.4f} | "
            f"ckpt={ckpt_path}"
        )

        all_metrics.append(metrics)
        pred_df["fold_id"] = fold_id
        all_pred_dfs.append(pred_df)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(eval_root / "metrics_all_folds.csv", index=False)

    if len(all_pred_dfs) > 0:
        pd.concat(all_pred_dfs, axis=0, ignore_index=True).to_csv(
            eval_root / "predictions_all_folds.csv",
            index=False,
        )

    if len(metrics_df) > 0:
        mean_row = metrics_df.mean(numeric_only=True).to_dict()
        with open(eval_root / "summary_mean.json", "w", encoding="utf-8") as f:
            json.dump(mean_row, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()