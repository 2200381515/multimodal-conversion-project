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
    split_csv: str
    out_dir: str

    batch_size: int = 8
    epochs: int = 25
    lr: float = 0.0003
    weight_decay: float = 0.0005

    embed_dim: int = 64
    num_heads: int = 2
    dropout: float = 0.5
    scale_tokens: int = 1
    matrix_tokens: int = 8
    matrix_pool_width: int = 32
    classifier_hidden: int = 64

    val_ratio: float = 0.2
    seed: int = 42
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config_from_yaml(yaml_path: str) -> TrainConfig:
    if yaml is None:
        raise RuntimeError("未安装 pyyaml，请先执行: pip install pyyaml")
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    return TrainConfig(**cfg_dict)


def load_split_df(split_csv: str) -> pd.DataFrame:
    """
    split_csv 需要至少包含:
    - subject_id
    - split  (取值: trainval / test)
    """
    df = pd.read_csv(split_csv)
    required = {"subject_id", "split"}
    if not required.issubset(set(df.columns)):
        raise ValueError(
            f"split_csv 缺少必须列: {sorted(required)}, got columns={list(df.columns)}"
        )

    df = df.copy()
    df["subject_id"] = df["subject_id"].astype(str)
    df["split"] = df["split"].astype(str)

    valid_splits = {"trainval", "test"}
    bad = sorted(set(df["split"].unique()) - valid_splits)
    if bad:
        raise ValueError(f"split_csv 存在非法 split 值: {bad}, 只允许 {sorted(valid_splits)}")

    return df


def build_dataset_and_df(cfg: TrainConfig) -> Tuple[PatientT0Dataset, pd.DataFrame]:
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
    df["subject_id"] = df["subject_id"].astype(str)
    df["label_convert"] = df["label_convert"].astype(int)
    return ds, df


def add_split_column(df: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["subject_id"] = df["subject_id"].astype(str)

    split_df = split_df[["subject_id", "split"]].drop_duplicates().copy()
    split_df["subject_id"] = split_df["subject_id"].astype(str)

    df = df.merge(split_df, on="subject_id", how="left")

    if df["split"].isna().any():
        miss = df.loc[df["split"].isna(), "subject_id"].tolist()[:10]
        raise RuntimeError(f"有样本在 split_csv 中找不到 split，例如: {miss}")

    return df


def split_train_val_indices(
    train_indices: List[int],
    labels: np.ndarray,
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """
    在 trainval 集内再切一个 val。
    如果标签太极端导致 stratify 失败，则退化为随机切分。
    """
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

    return tr_idx.tolist(), va_idx.tolist()


def make_loader(
    dataset,
    indices: List[int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
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
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    criterion: nn.Module,
) -> Dict:
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
    loss_mean = float(np.mean(losses)) if losses else float("nan")

    return {
        "y_true": y_true_arr,
        "y_prob": y_prob_arr,
        "subject_id": meta_subjects,
        "loss": loss_mean,
    }


def build_eval_result(
    y_true_arr: np.ndarray,
    y_prob_arr: np.ndarray,
    subject_ids: List[str],
    loss_value: float,
    thr: float,
) -> Dict:
    y_pred_arr = (y_prob_arr >= thr).astype(int)

    m = binary_metrics(y_true_arr, y_prob_arr, thr=thr)
    if len(np.unique(y_true_arr)) > 1:
        pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))
    else:
        pr_auc = float("nan")

    m["PR_AUC"] = pr_auc
    m["loss"] = float(loss_value)
    m["thr"] = float(thr)

    pred_df = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "y_true": y_true_arr,
            "y_prob": y_prob_arr,
            "y_pred": y_pred_arr,
            "used_thr": float(thr),
        }
    )

    return {
        "metrics": m,
        "pred_df": pred_df,
    }


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    criterion: nn.Module,
    thr: float = 0.5,
) -> Dict:
    collected = collect_predictions(
        model=model,
        loader=loader,
        device=device,
        criterion=criterion,
    )
    return build_eval_result(
        y_true_arr=collected["y_true"],
        y_prob_arr=collected["y_prob"],
        subject_ids=collected["subject_id"],
        loss_value=collected["loss"],
        thr=thr,
    )


def choose_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """
    更稳的阈值选择：
    直接看验证集负类分数分布，
    用负类高分位数来定阈值，优先控制假阳性。
    """
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5

    neg_probs = y_prob[y_true == 0]
    pos_probs = y_prob[y_true == 1]

    if len(neg_probs) == 0:
        return 0.5

    # 主阈值：负类 85% 分位点
    thr = float(np.quantile(neg_probs, 0.875))

    # 防止阈值太极端
    thr = max(0.30, min(0.80, thr))

    print(
        f"[THR SEARCH] thr={thr:.4f} | "
        f"neg_q875={np.quantile(neg_probs, 0.875):.4f} | "
        f"neg_mean={neg_probs.mean():.4f} | "
        f"pos_mean={pos_probs.mean():.4f}"
    )
    return thr


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
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

    return float(np.mean(losses)) if losses else float("nan")


def save_json(obj: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def run_train(cfg: TrainConfig):
    ensure_dir(cfg.out_dir)
    set_seed(cfg.seed)

    save_json(asdict(cfg), os.path.join(cfg.out_dir, "config_snapshot.json"))

    dataset, df = build_dataset_and_df(cfg)
    split_df = load_split_df(cfg.split_csv)
    df = add_split_column(df, split_df)

    scale_dim = infer_scale_dim(dataset)

    print(f"[INFO] dataset size = {len(dataset)}")
    print(f"[INFO] scale_dim = {scale_dim}")
    print(f"[INFO] device = {cfg.device}")
    print("[INFO] split counts:")
    print(df["split"].value_counts())
    print("[INFO] test label counts:")
    print(df.loc[df["split"] == "test", "label_convert"].value_counts())

    test_indices = df.index[df["split"] == "test"].tolist()
    trainval_indices = df.index[df["split"] == "trainval"].tolist()

    if len(test_indices) == 0:
        raise RuntimeError("test 集为空，请检查 split_csv。")
    if len(trainval_indices) == 0:
        raise RuntimeError("trainval 集为空，请检查 split_csv。")

    labels_all = df["label_convert"].astype(int).to_numpy()
    train_indices, val_indices = split_train_val_indices(
        train_indices=trainval_indices,
        labels=labels_all,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
    )

    if len(val_indices) == 0:
        raise RuntimeError("val 集为空，请检查 trainval 数量或 val_ratio。")

    print(f"[INFO] n_train={len(train_indices)}, n_val={len(val_indices)}, n_test={len(test_indices)}")
    print("[INFO] train label counts:")
    print(pd.Series(labels_all[train_indices]).value_counts())
    print("[INFO] val label counts:")
    print(pd.Series(labels_all[val_indices]).value_counts())
    print("[INFO] test label counts:")
    print(pd.Series(labels_all[test_indices]).value_counts())

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

    raw_pos_weight = float(n_neg / max(n_pos, 1))
    pos_weight_value = min(raw_pos_weight, 4.0)
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

    patience = 10
    min_epochs = 15
    no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            device=cfg.device,
            optimizer=optimizer,
            criterion=criterion,
        )

        val_res_05 = evaluate_epoch(
            model=model,
            loader=val_loader,
            device=cfg.device,
            criterion=criterion,
            thr=0.5,
        )
        val_metrics_05 = val_res_05["metrics"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics_05["loss"],
            "val_auc": val_metrics_05["AUC"],
            "val_pr_auc": val_metrics_05["PR_AUC"],
            "val_f1_thr_05": val_metrics_05["F1"],
            "val_acc_thr_05": val_metrics_05["ACC"],
            "val_sen_thr_05": val_metrics_05["SEN"],
            "val_spe_thr_05": val_metrics_05["SPE"],
        }
        history_rows.append(row)

        current_val_auc = float(val_metrics_05["AUC"])
        current_val_pr_auc = float(val_metrics_05["PR_AUC"])

        if np.isnan(current_val_auc):
            current_val_auc = -1.0
        if np.isnan(current_val_pr_auc):
            current_val_pr_auc = -1.0

        current_val_score = 0.35 * current_val_auc + 0.65 * current_val_pr_auc

        if current_val_score > best_val_auc:
            best_val_auc = float(current_val_score)
            best_epoch = epoch
            best_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"[single] epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_auc={val_metrics_05['AUC']:.4f} | "
            f"val_pr_auc={val_metrics_05['PR_AUC']:.4f} | "
            f"val_f1@0.5={val_metrics_05['F1']:.4f}"
        )

        if epoch >= min_epochs and no_improve >= patience:
            print(f"[EARLY STOP] no improvement in val_loss for {patience} epochs, stop at epoch {epoch}")
            break


    if best_state_dict is None:
        raise RuntimeError("没有成功保存 best_state_dict")

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(os.path.join(cfg.out_dir, "history.csv"), index=False)

    model.load_state_dict(best_state_dict)

    # 先在验证集上找最佳阈值

    val_raw = collect_predictions(
        model=model,
        loader=val_loader,
        device=cfg.device,
        criterion=criterion,
    )
    best_thr = choose_best_threshold(val_raw["y_true"], val_raw["y_prob"])


    val_res_best = build_eval_result(
        y_true_arr=val_raw["y_true"],
        y_prob_arr=val_raw["y_prob"],
        subject_ids=val_raw["subject_id"],
        loss_value=val_raw["loss"],
        thr=best_thr,
    )
    val_metrics_best = val_res_best["metrics"]
    val_pred_df = val_res_best["pred_df"]
    val_pred_df.to_csv(os.path.join(cfg.out_dir, "predictions_val.csv"), index=False)
    print(f"[VAL] pred_pos_rate={val_pred_df['y_pred'].mean():.4f}")




    # 测试集用验证集选出的阈值
    test_res = evaluate_epoch(
        model=model,
        loader=test_loader,
        device=cfg.device,
        criterion=criterion,
        thr=best_thr,
    )
    test_metrics = test_res["metrics"]
    pred_df = test_res["pred_df"]
    pred_df.to_csv(os.path.join(cfg.out_dir, "predictions_test.csv"), index=False)
    print(f"[TEST] pred_pos_rate={pred_df['y_pred'].mean():.4f}")


    # 图
    fig_dir = os.path.join(cfg.out_dir, "figures")
    ensure_dir(fig_dir)

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
            title="Single Split ROC",
        )
        plot_pr_curve(
            y_true=y_true,
            y_prob=y_prob,
            out_path=os.path.join(fig_dir, "pr.png"),
            title="Single Split PR",
        )

    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": asdict(cfg),
        "scale_dim": scale_dim,
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "best_thr": best_thr,
        "n_train": len(train_indices),
        "n_val": len(val_indices),
        "n_test": len(test_indices),
        "pos_weight": pos_weight_value,
    }
    torch.save(ckpt, os.path.join(cfg.out_dir, "model.pt"))

    metrics_out = {
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
        "best_epoch": int(best_epoch),
        "best_val_auc": float(best_val_auc),
        "best_thr": float(best_thr),
        "n_train": len(train_indices),
        "n_val": len(val_indices),
        "n_test": len(test_indices),
        "pos_weight": float(pos_weight_value),
    }
    save_json(metrics_out, os.path.join(cfg.out_dir, "metrics_test.json"))

    val_metrics_out = {
        "AUC": val_metrics_best["AUC"],
        "PR_AUC": val_metrics_best["PR_AUC"],
        "F1": val_metrics_best["F1"],
        "ACC": val_metrics_best["ACC"],
        "SEN": val_metrics_best["SEN"],
        "SPE": val_metrics_best["SPE"],
        "loss": val_metrics_best["loss"],
        "best_thr": float(best_thr),
    }
    save_json(val_metrics_out, os.path.join(cfg.out_dir, "metrics_val.json"))

    summary_df = pd.DataFrame(
        [
            {
                "metric": "AUC",
                "value": test_metrics["AUC"],
            },
            {
                "metric": "PR_AUC",
                "value": test_metrics["PR_AUC"],
            },
            {
                "metric": "F1",
                "value": test_metrics["F1"],
            },
            {
                "metric": "ACC",
                "value": test_metrics["ACC"],
            },
            {
                "metric": "SEN",
                "value": test_metrics["SEN"],
            },
            {
                "metric": "SPE",
                "value": test_metrics["SPE"],
            },
            {
                "metric": "TP",
                "value": test_metrics["TP"],
            },
            {
                "metric": "FP",
                "value": test_metrics["FP"],
            },
            {
                "metric": "TN",
                "value": test_metrics["TN"],
            },
            {
                "metric": "FN",
                "value": test_metrics["FN"],
            },
            {
                "metric": "best_val_auc",
                "value": best_val_auc,
            },
            {
                "metric": "best_thr",
                "value": best_thr,
            },
            {
                "metric": "test_loss",
                "value": test_metrics["loss"],
            },
        ]
    )
    summary_df.to_csv(os.path.join(cfg.out_dir, "summary_metrics.csv"), index=False)

    print("\n[TRAIN DONE] validation (using best_thr):")
    print(pd.DataFrame([val_metrics_out]))

    print("\n[TRAIN DONE] test metrics:")
    print(pd.DataFrame([metrics_out]))

    print(
        f"\n[SINGLE TEST] "
        f"AUC={test_metrics['AUC']:.4f} | "
        f"PR_AUC={test_metrics['PR_AUC']:.4f} | "
        f"F1={test_metrics['F1']:.4f} | "
        f"SEN={test_metrics['SEN']:.4f} | "
        f"SPE={test_metrics['SPE']:.4f} | "
        f"best_thr={best_thr:.4f}"
    )


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--yaml",
        type=str,
        required=True,
        help="例如: configs/mm_transformer_baseline/transformer_baseline_single.yaml",
    )
    args = ap.parse_args()

    cfg = load_config_from_yaml(args.yaml)
    run_train(cfg)


if __name__ == "__main__":
    main()