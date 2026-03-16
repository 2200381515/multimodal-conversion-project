from __future__ import annotations

import os
import json
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

from pipeline.dataset import PatientT0Dataset
from baselines.metrics import binary_metrics
from models.mm_transformer_baseline.multimodal_transformer import (
    MultimodalTransformerBaseline,
)
from training.common.visualize import (
    ensure_dir,
    plot_confusion_binary,
    plot_roc_curve,
    plot_pr_curve,
)


@dataclass
class TrainConfig:
    cohort_csv: str
    fold_index_csv: str
    out_dir: str

    batch_size: int = 8  # 增加 batch_size，提高训练效率
    epochs: int = 50
    lr: float = 0.0005
    weight_decay: float = 0.0005

    embed_dim: int = 128
    num_heads: int = 4
    dropout: float = 0.3
    scale_tokens: int = 1
    matrix_tokens: int = 8
    matrix_pool_width: int = 32
    classifier_hidden: int = 128

    val_ratio: float = 0.2
    seed: int = 42
    num_workers: int = 0  # 禁用多线程加载
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 新增 lr_scheduler 参数
    lr_scheduler: Dict = None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config_from_yaml(yaml_path: str) -> TrainConfig:
    print("Loading config from YAML...")
    if yaml is None:
        raise RuntimeError("未安装 pyyaml，请先执行: pip install pyyaml")
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    print("Config loaded successfully.")
    return TrainConfig(**cfg_dict)


def load_fold_index_map(fold_index_csv: str) -> Dict[str, int]:
    print("Loading fold index map...")
    df = pd.read_csv(fold_index_csv)
    if "subject_id" not in df.columns or "test_fold_id" not in df.columns:
        raise ValueError(f"fold_index_csv 缺少必须列: subject_id/test_fold_id, got columns={list(df.columns)}")
    print("Fold index map loaded.")
    return dict(zip(df["subject_id"].astype(str), df["test_fold_id"].astype(int)))


def build_dataset_and_df(cfg: TrainConfig) -> Tuple[PatientT0Dataset, pd.DataFrame]:
    print("Building dataset and dataframe...")
    ds = PatientT0Dataset(
        cohort_csv=cfg.cohort_csv,
        require_eligible=False,
        drop_qc_excluded=False,
        fill_missing=True,
        sc_norm="zscore_global",
        fc_norm="zscore_global",
        scale_norm_stats=None,
        drop_scale_cols=["DATASET-DIAG2"],
    )
    df = ds.df.copy().reset_index(drop=True)
    print(f"Dataset built with {len(df)} samples.")
    return ds, df


def add_fold_column(df: pd.DataFrame, fold_map: Dict[str, int]) -> pd.DataFrame:
    print("Adding fold column...")
    df = df.copy()
    df["subject_id"] = df["subject_id"].astype(str)
    df["test_fold_id"] = df["subject_id"].map(fold_map)
    if df["test_fold_id"].isna().any():
        miss = df.loc[df["test_fold_id"].isna(), "subject_id"].tolist()[:10]
        raise RuntimeError(f"有样本在 fold_index.csv 中找不到 test_fold_id，例如: {miss}")
    df["test_fold_id"] = df["test_fold_id"].astype(int)
    print("Fold column added.")
    return df


def split_train_val_indices(
    train_indices: List[int],
    labels: np.ndarray,
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    print("Splitting train and validation indices...")
    if len(train_indices) < 2:
        return train_indices, []

    y_train = labels[train_indices]
    idx_arr = np.array(train_indices)

    try:
        tr_idx, va_idx = train_test_split(
            idx_arr,
            test_size=val_ratio,
            random_state=seed,
            stratify=y_train,
        )
    except Exception:
        tr_idx, va_idx = train_test_split(
            idx_arr,
            test_size=val_ratio,
            random_state=seed,
            shuffle=True,
        )
    print(f"Train/Val split completed: {len(tr_idx)} train, {len(va_idx)} validation.")
    return tr_idx.tolist(), va_idx.tolist()


def make_loader(
    dataset,
    indices: List[int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    print("Creating data loader...")
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def infer_scale_dim(dataset: PatientT0Dataset) -> int:
    sample = dataset[0]
    x_scale = sample["x_scale"]
    if x_scale.dim() != 1:
        raise ValueError(f"x_scale 应为 1 维, got shape={tuple(x_scale.shape)}")
    return int(x_scale.numel())


def build_model(cfg: TrainConfig, scale_dim: int) -> MultimodalTransformerBaseline:
    print("Building model...")
    model = MultimodalTransformerBaseline(
        scale_dim=scale_dim,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        scale_tokens=cfg.scale_tokens,
        matrix_tokens=cfg.matrix_tokens,
        matrix_pool_width=cfg.matrix_pool_width,
        classifier_hidden=cfg.classifier_hidden,
    )
    print("Model built.")
    return model


def move_batch_to_device(batch: Dict, device: str) -> Dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    criterion: nn.Module,
) -> Dict:
    print("Evaluating model...")
    model.eval()

    losses = []
    y_true_all = []
    y_prob_all = []
    meta_subjects = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        out = model(
            x_scale=batch["x_scale"].float(),
            x_sc=batch["x_sc"].float(),
            x_fc=batch["x_fc"].float(),
            modality_mask=batch["modality_mask"].float(),
        )
        logits = out["logits"]
        y = batch["y"].float()

        loss = criterion(logits, y)
        losses.append(float(loss.item()))

        y_prob = torch.sigmoid(logits).detach().cpu().numpy()
        y_true = y.detach().cpu().numpy()

        y_prob_all.extend(y_prob.tolist())
        y_true_all.extend(y_true.astype(int).tolist())

        metas = batch["meta"]
        if isinstance(metas, dict) and "subject_id" in metas:
            meta_subjects.extend([str(s) for s in metas["subject_id"]])
        else:
            meta_subjects.extend(["unknown"] * len(y_true))

    y_true_arr = np.asarray(y_true_all).astype(int)
    y_prob_arr = np.asarray(y_prob_all).astype(float)
    y_pred_arr = (y_prob_arr >= 0.5).astype(int)

    m = binary_metrics(y_true_arr, y_prob_arr, thr=0.5)
    if len(np.unique(y_true_arr)) > 1:
        pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))
    else:
        pr_auc = float("nan")

    m["PR_AUC"] = pr_auc
    m["loss"] = float(np.mean(losses)) if losses else float("nan")

    pred_df = pd.DataFrame(
        {
            "subject_id": meta_subjects,
            "y_true": y_true_arr,
            "y_prob": y_prob_arr,
            "y_pred": y_pred_arr,
        }
    )

    return {
        "metrics": m,
        "pred_df": pred_df,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler=None,
) -> float:
    print("Training one epoch...")
    model.train()
    losses = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad()

        out = model(
            x_scale=batch["x_scale"].float(),
            x_sc=batch["x_sc"].float(),
            x_fc=batch["x_fc"].float(),
            modality_mask=batch["modality_mask"].float(),
        )
        logits = out["logits"]
        y = batch["y"].float()

        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

    if scheduler:
        scheduler.step()  # 更新学习率

    print(f"Epoch training complete, average loss: {np.mean(losses):.4f}")
    return float(np.mean(losses)) if losses else float("nan")


def save_json(obj: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def summarize_metrics_across_folds(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    print("Summarizing metrics across folds...")
    metric_cols = [
        "AUC", "PR_AUC", "F1", "ACC", "SEN", "SPE",
        "TP", "FP", "TN", "FN",
        "best_val_auc",
        "test_loss",
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


def train_one_fold(
    cfg: TrainConfig,
    dataset: PatientT0Dataset,
    df: pd.DataFrame,
    fold_id: int,
    scale_dim: int,
) -> Dict:
    fold_out_dir = os.path.join(cfg.out_dir, f"fold_{fold_id}")
    fig_dir = os.path.join(fold_out_dir, "figures")
    ensure_dir(fold_out_dir)
    ensure_dir(fig_dir)

    test_indices = df.index[df["test_fold_id"] == fold_id].tolist()
    trainval_indices = df.index[df["test_fold_id"] != fold_id].tolist()

    labels_all = df["label_convert"].astype(int).to_numpy()
    train_indices, val_indices = split_train_val_indices(
        train_indices=trainval_indices,
        labels=labels_all,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed + fold_id,
    )

    if len(val_indices) == 0:
        raise RuntimeError(f"fold {fold_id} 的 val 集为空，请检查数据量或 val_ratio。")

    train_loader = make_loader(
        dataset,
        train_indices,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = make_loader(
        dataset,
        val_indices,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    test_loader = make_loader(
        dataset,
        test_indices,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    model = build_model(cfg, scale_dim).to(cfg.device)

    y_train = labels_all[train_indices]
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight_value = float(n_neg / max(n_pos, 1))
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=cfg.device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    history_rows = []
    best_val_auc = -1.0
    best_epoch = -1
    best_state_dict = None

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            device=cfg.device,
            optimizer=optimizer,
            criterion=criterion,
        )

        val_res = evaluate_epoch(
            model=model,
            loader=val_loader,
            device=cfg.device,
            criterion=criterion,
        )
        val_metrics = val_res["metrics"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_auc": val_metrics["AUC"],
            "val_pr_auc": val_metrics["PR_AUC"],
            "val_f1": val_metrics["F1"],
            "val_acc": val_metrics["ACC"],
            "val_sen": val_metrics["SEN"],
            "val_spe": val_metrics["SPE"],
        }
        history_rows.append(row)

        current_val_auc = val_metrics["AUC"]
        if np.isnan(current_val_auc):
            current_val_auc = -1.0

        if current_val_auc > best_val_auc:
            best_val_auc = float(current_val_auc)
            best_epoch = epoch
            best_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        print(
            f"[fold {fold_id}] epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_auc={val_metrics['AUC']:.4f} | "
            f"val_f1={val_metrics['F1']:.4f}"
        )

    if best_state_dict is None:
        raise RuntimeError(f"fold {fold_id} 没有成功保存 best_state_dict")

    # 保存 history
    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(os.path.join(fold_out_dir, "history.csv"), index=False)

    # 加载最佳模型
    model.load_state_dict(best_state_dict)

    # 测试集评估
    test_res = evaluate_epoch(
        model=model,
        loader=test_loader,
        device=cfg.device,
        criterion=criterion,
    )
    test_metrics = test_res["metrics"]
    pred_df = test_res["pred_df"]
    pred_df["fold_id"] = fold_id
    pred_df.to_csv(os.path.join(fold_out_dir, "predictions.csv"), index=False)

    # 图
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

    # 保存 checkpoint
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": asdict(cfg),
        "fold_id": fold_id,
        "scale_dim": scale_dim,
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
    }
    torch.save(ckpt, os.path.join(fold_out_dir, "model.pt"))

    metrics_out = dict(test_metrics)
    metrics_out["fold_id"] = fold_id
    metrics_out["best_epoch"] = int(best_epoch)
    metrics_out["best_val_auc"] = float(best_val_auc)
    save_json(metrics_out, os.path.join(fold_out_dir, "metrics.json"))

    print(
        f"[fold {fold_id}] TEST | "
        f"AUC={test_metrics['AUC']:.4f} | "
        f"PR_AUC={test_metrics['PR_AUC']:.4f} | "
        f"F1={test_metrics['F1']:.4f} | "
        f"SEN={test_metrics['SEN']:.4f} | "
        f"SPE={test_metrics['SPE']:.4f}"
    )

    return {
        "fold_id": fold_id,
        "AUC": test_metrics["AUC"],
        "PR_AUC": test_metrics["PR_AUC"],
        "F1": test_metrics["F1"],
        "ACC": test_metrics["ACC"],
        "SEN": test_metrics["SEN"],
        "SPE": test_metrics["SPE"],
        "TP": test_metrics["TP"],
        "FP": test_metrics["FP"],
        "TN": test_metrics["TN"],
        "FN": test_metrics["FN"],
        "test_loss": test_metrics["loss"],
    }