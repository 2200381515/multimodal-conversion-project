from __future__ import annotations

import os
import json
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import average_precision_score

from pipeline.dataset import PatientT0Dataset
from baselines.metrics import binary_metrics
from models.mm_transformer_baseline.multimodal_transformer import (
    MultimodalTransformerBaseline,
)
from training.common.visualize import (
    ensure_dir,
    plot_confusion_binary,
    plot_pr_curve,
    plot_roc_curve,
)


def load_fold_index_map(fold_index_csv: str) -> Dict[str, int]:
    df = pd.read_csv(fold_index_csv)
    if "subject_id" not in df.columns or "test_fold_id" not in df.columns:
        raise ValueError(
            f"fold_index_csv 缺少必须列: subject_id/test_fold_id, got columns={list(df.columns)}"
        )
    return dict(zip(df["subject_id"].astype(str), df["test_fold_id"].astype(int)))


def build_dataset(cohort_csv: str) -> PatientT0Dataset:
    ds = PatientT0Dataset(
        cohort_csv=cohort_csv,
        require_eligible=False,
        drop_qc_excluded=False,
        fill_missing=True,
        sc_norm="zscore_global",
        fc_norm="zscore_global",
        scale_norm_stats=None,
        drop_scale_cols=["DATASET-DIAG2"],
    )
    return ds


def add_fold_column(df: pd.DataFrame, fold_map: Dict[str, int]) -> pd.DataFrame:
    df = df.copy()
    df["subject_id"] = df["subject_id"].astype(str)
    df["test_fold_id"] = df["subject_id"].map(fold_map)
    if df["test_fold_id"].isna().any():
        miss = df.loc[df["test_fold_id"].isna(), "subject_id"].tolist()[:10]
        raise RuntimeError(f"有样本在 fold_index.csv 中找不到 test_fold_id，例如: {miss}")
    df["test_fold_id"] = df["test_fold_id"].astype(int)
    return df


def infer_scale_dim(dataset: PatientT0Dataset) -> int:
    sample = dataset[0]
    x_scale = sample["x_scale"]
    if x_scale.dim() != 1:
        raise ValueError(f"x_scale 应为 1 维, got shape={tuple(x_scale.shape)}")
    return int(x_scale.numel())


def make_loader(
    dataset,
    indices: List[int],
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def move_batch_to_device(batch: Dict, device: str) -> Dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def build_model_from_ckpt(ckpt: Dict, device: str) -> MultimodalTransformerBaseline:
    cfg = ckpt["config"]
    model = MultimodalTransformerBaseline(
        scale_dim=int(ckpt["scale_dim"]),
        embed_dim=int(cfg["embed_dim"]),
        num_heads=int(cfg["num_heads"]),
        dropout=float(cfg["dropout"]),
        scale_tokens=int(cfg["scale_tokens"]),
        matrix_tokens=int(cfg["matrix_tokens"]),
        matrix_pool_width=int(cfg["matrix_pool_width"]),
        classifier_hidden=int(cfg["classifier_hidden"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Dict:
    y_true_all = []
    y_prob_all = []
    subject_ids = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        out = model(
            x_scale=batch["x_scale"].float(),
            x_sc=batch["x_sc"].float(),
            x_fc=batch["x_fc"].float(),
            modality_mask=batch["modality_mask"].float(),
        )

        y_prob = out["prob"].detach().cpu().numpy()
        y_true = batch["y"].detach().cpu().numpy().astype(int)

        y_prob_all.extend(y_prob.tolist())
        y_true_all.extend(y_true.tolist())

        metas = batch["meta"]
        if isinstance(metas, dict) and "subject_id" in metas:
            subject_ids.extend([str(s) for s in metas["subject_id"]])
        else:
            subject_ids.extend(["unknown"] * len(y_true))

    y_true_arr = np.asarray(y_true_all).astype(int)
    y_prob_arr = np.asarray(y_prob_all).astype(float)
    y_pred_arr = (y_prob_arr >= 0.5).astype(int)

    metrics = binary_metrics(y_true_arr, y_prob_arr, thr=0.5)
    if len(np.unique(y_true_arr)) > 1:
        metrics["PR_AUC"] = float(average_precision_score(y_true_arr, y_prob_arr))
    else:
        metrics["PR_AUC"] = float("nan")

    pred_df = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "y_true": y_true_arr,
            "y_prob": y_prob_arr,
            "y_pred": y_pred_arr,
        }
    )

    return {
        "metrics": metrics,
        "pred_df": pred_df,
    }


def save_json(obj: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def summarize_metrics_across_folds(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "AUC", "PR_AUC", "F1", "ACC", "SEN", "SPE",
        "TP", "FP", "TN", "FN",
        "best_val_auc",
        "best_epoch",
    ]
    rows = []
    for c in metric_cols:
        if c not in fold_metrics.columns:
            continue
        vals = pd.to_numeric(fold_metrics[c], errors="coerce")
        rows.append(
            {
                "metric": c,
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "min": float(vals.min()),
                "max": float(vals.max()),
            }
        )
    return pd.DataFrame(rows)


def run_eval(
    cohort_csv: str,
    fold_index_csv: str,
    ckpt_dir: str,
    out_dir: str,
    batch_size: int = 8,
    num_workers: int = 0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    ensure_dir(out_dir)

    dataset = build_dataset(cohort_csv)
    df = dataset.df.copy().reset_index(drop=True)

    fold_map = load_fold_index_map(fold_index_csv)
    df = add_fold_column(df, fold_map)

    scale_dim = infer_scale_dim(dataset)
    all_fold_ids = sorted(df["test_fold_id"].unique().tolist())

    print(f"[INFO] dataset size = {len(dataset)}")
    print(f"[INFO] scale_dim = {scale_dim}")
    print(f"[INFO] folds = {all_fold_ids}")
    print(f"[INFO] device = {device}")

    fold_rows = []

    for fold_id in all_fold_ids:
        fold_ckpt_path = os.path.join(ckpt_dir, f"fold_{fold_id}", "model.pt")
        if not os.path.exists(fold_ckpt_path):
            print(f"[WARN] fold {fold_id} 缺少 checkpoint，跳过: {fold_ckpt_path}")
            continue

        fold_out_dir = os.path.join(out_dir, f"fold_{fold_id}")
        fig_dir = os.path.join(fold_out_dir, "figures")
        ensure_dir(fold_out_dir)
        ensure_dir(fig_dir)

        test_indices = df.index[df["test_fold_id"] == fold_id].tolist()
        test_loader = make_loader(
            dataset=dataset,
            indices=test_indices,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        ckpt = torch.load(fold_ckpt_path, map_location=device)
        model = build_model_from_ckpt(ckpt, device=device)

        res = evaluate_model(model=model, loader=test_loader, device=device)
        metrics = res["metrics"]
        pred_df = res["pred_df"]
        pred_df["fold_id"] = fold_id

        pred_path = os.path.join(fold_out_dir, "predictions.csv")
        pred_df.to_csv(pred_path, index=False)

        y_true = pred_df["y_true"].to_numpy()
        y_prob = pred_df["y_prob"].to_numpy()
        y_pred = pred_df["y_pred"].to_numpy()

        plot_confusion_binary(
            y_true=y_true,
            y_pred=y_pred,
            out_path=os.path.join(fig_dir, "confusion_matrix.png"),
        )
        if len(np.unique(y_true)) > 1:
            plot_roc_curve(
                y_true=y_true,
                y_prob=y_prob,
                out_path=os.path.join(fig_dir, "roc.png"),
                title=f"Fold {fold_id} ROC",
            )
            plot_pr_curve(
                y_true=y_true,
                y_prob=y_prob,
                out_path=os.path.join(fig_dir, "pr.png"),
                title=f"Fold {fold_id} PR",
            )

        fold_metrics = dict(metrics)
        fold_metrics["fold_id"] = int(fold_id)
        fold_metrics["best_epoch"] = int(ckpt.get("best_epoch", -1))
        fold_metrics["best_val_auc"] = float(ckpt.get("best_val_auc", float("nan")))

        save_json(fold_metrics, os.path.join(fold_out_dir, "metrics.json"))

        print(
            f"[fold {fold_id}] "
            f"AUC={metrics['AUC']:.4f} | "
            f"PR_AUC={metrics['PR_AUC']:.4f} | "
            f"F1={metrics['F1']:.4f} | "
            f"SEN={metrics['SEN']:.4f} | "
            f"SPE={metrics['SPE']:.4f}"
        )

        fold_rows.append(fold_metrics)

    if not fold_rows:
        raise RuntimeError("没有成功评估任何 fold，请检查 ckpt_dir 是否正确。")

    fold_metrics_df = pd.DataFrame(fold_rows)
    fold_metrics_df.to_csv(os.path.join(out_dir, "fold_metrics.csv"), index=False)

    summary_df = summarize_metrics_across_folds(fold_metrics_df)
    summary_df.to_csv(os.path.join(out_dir, "summary_metrics.csv"), index=False)

    print("\n[EVAL DONE] fold-level metrics:")
    print(fold_metrics_df)
    print("\n[EVAL DONE] summary metrics:")
    print(summary_df)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort_csv", type=str, required=True)
    ap.add_argument("--fold_index_csv", type=str, required=True)
    ap.add_argument("--ckpt_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = ap.parse_args()

    run_eval(
        cohort_csv=args.cohort_csv,
        fold_index_csv=args.fold_index_csv,
        ckpt_dir=args.ckpt_dir,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )


if __name__ == "__main__":
    main()