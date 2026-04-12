from __future__ import annotations

import os
import json
import pandas as pd
import torch
import numpy as np
from typing import Dict

from pipeline.dataset import PatientT0Dataset
from models.mm_transformer_baseline.multimodal_transformer import (
    MultimodalTransformerBaseline,
)
from training.common.visualize import ensure_dir


def load_model_from_ckpt(ckpt_path: str, device: str) -> MultimodalTransformerBaseline:
    """
    加载训练好的模型 checkpoint。
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    model = MultimodalTransformerBaseline(
        scale_dim=int(ckpt["scale_dim"]),
        embed_dim=int(ckpt["config"]["embed_dim"]),
        num_heads=int(ckpt["config"]["num_heads"]),
        dropout=float(ckpt["config"]["dropout"]),
        scale_tokens=int(ckpt["config"]["scale_tokens"]),
        matrix_tokens=int(ckpt["config"]["matrix_tokens"]),
        matrix_pool_width=int(ckpt["config"]["matrix_pool_width"]),
        classifier_hidden=int(ckpt["config"]["classifier_hidden"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def build_dataset(cohort_csv: str) -> PatientT0Dataset:
    """
    根据 cohort_csv 生成 PatientT0Dataset 数据集。
    """
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


def make_loader(dataset: PatientT0Dataset, indices: list, batch_size: int) -> torch.utils.data.DataLoader:
    """
    创建数据加载器
    """
    from torch.utils.data import DataLoader, Subset

    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def move_batch_to_device(batch: Dict, device: str) -> Dict:
    """
    将 batch 中的所有 tensor 移动到指定的设备。
    """
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


@torch.no_grad()
def predict_on_dataset(
    model: MultimodalTransformerBaseline,
    dataset: PatientT0Dataset,
    batch_size: int,
    device: str,
    cohort_csv: str,
) -> pd.DataFrame:
    """
    在整个数据集上进行预测，并返回包含预测结果的 DataFrame。
    """
    loader = make_loader(dataset, list(range(len(dataset))), batch_size)
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

        y_true = batch["y"].cpu().numpy()
        y_prob = out["prob"].cpu().numpy()

        y_true_all.extend(y_true.tolist())
        y_prob_all.extend(y_prob.tolist())

        metas = batch["meta"]
        if isinstance(metas, dict) and "subject_id" in metas:
            subject_ids.extend([str(s) for s in metas["subject_id"]])
        else:
            subject_ids.extend(["unknown"] * len(y_true))

    y_true_arr = np.asarray(y_true_all).astype(int)
    y_prob_arr = np.asarray(y_prob_all).astype(float)
    y_pred_arr = (y_prob_arr >= 0.5).astype(int)

    pred_df = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "y_true": y_true_arr,
            "y_prob": y_prob_arr,
            "y_pred": y_pred_arr,
        }
    )

    return pred_df


def save_predictions(pred_df: pd.DataFrame, out_path: str):
    """
    保存预测结果到指定路径。
    """
    pred_df.to_csv(out_path, index=False)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort_csv", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True, help="Model checkpoint path")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # 确保输出目录存在
    ensure_dir(args.out_dir)

    # 加载数据集
    dataset = build_dataset(args.cohort_csv)

    # 加载模型
    model = load_model_from_ckpt(args.ckpt, device=args.device)

    # 预测并保存结果
    pred_df = predict_on_dataset(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        device=args.device,
        cohort_csv=args.cohort_csv,
    )

    pred_out_path = os.path.join(args.out_dir, "predictions.csv")
    save_predictions(pred_df, pred_out_path)

    print(f"Predictions saved to {pred_out_path}")


if __name__ == "__main__":
    main()