from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from pipeline.dataset import PatientT0Dataset
from models.tmc.mm_transformer_tmc import MMTransformerTMC
from models.tmc.evidential_losses import compute_tmc_loss


def load_yaml(path: str) -> Dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

def attach_fold_ids(dataset: PatientT0Dataset, fold_index_csv: str) -> pd.DataFrame:
    fold_df = pd.read_csv(fold_index_csv)
    if "subject_id" not in fold_df.columns:
        raise ValueError("fold_index.csv must contain column subject_id")
    if "test_fold_id" not in fold_df.columns:
        raise ValueError("fold_index.csv must contain column test_fold_id")
    return fold_df[["subject_id", "test_fold_id"]].copy()


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


def split_train_val(
    train_indices: List[int],
    subject_ids: List[str],
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """
    按 subject 切分，避免泄漏
    """
    rng = np.random.default_rng(seed)

    train_subjects = sorted(list({subject_ids[i] for i in train_indices}))
    rng.shuffle(train_subjects)

    n_val_subj = max(1, int(round(len(train_subjects) * val_ratio)))
    val_subject_set = set(train_subjects[:n_val_subj])

    tr_idx, va_idx = [], []
    for i in train_indices:
        if subject_ids[i] in val_subject_set:
            va_idx.append(i)
        else:
            tr_idx.append(i)

    if len(tr_idx) == 0 or len(va_idx) == 0:
        raise RuntimeError("train/val split failed; got empty split.")

    return tr_idx, va_idx


def make_loader(dataset, indices, batch_size, shuffle):
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


@torch.no_grad()
def evaluate_epoch(model, loader, device):
    model.eval()

    y_true_all, y_prob_all, u_all = [], [], []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch_to_device(batch, device)
        out = model(
            x_scale=batch["x_scale"],
            x_sc=batch["x_sc"],
            x_fc=batch["x_fc"],
            modality_mask=batch.get("modality_mask", None),
        )

        y = batch["y"].long()

        loss_out = compute_tmc_loss(
            fused_prob=out.probs,
            view_probs=out.view_probs,
            view_evidences=out.view_evidences,
            target=y,
            pos_weight=1.0,  # 验证阶段不必加权到 loss 里选模
            lambda_view=0.5,
            lambda_evidence=1e-4,
        )

        total_loss += loss_out.fused_bce.item()
        n_batches += 1

        y_true_all.extend(y.detach().cpu().tolist())
        y_prob_all.extend(out.probs.detach().cpu().tolist())
        y_u = out.fused_uncertainty.squeeze(-1) if out.fused_uncertainty.ndim == 2 else out.fused_uncertainty
        y_u = y_u.detach().cpu().tolist()
        if isinstance(y_u, float):
            y_u = [y_u]
        u_all.extend(y_u)

    y_true = np.asarray(y_true_all).astype(int)
    y_prob = np.asarray(y_prob_all).astype(float)
    u_arr = np.asarray(u_all).astype(float)

    from sklearn.metrics import roc_auc_score

    auc = float("nan")
    if len(np.unique(y_true)) >= 2:
        auc = roc_auc_score(y_true, y_prob)

    mean_loss = total_loss / max(n_batches, 1)
    mean_u = float(np.mean(u_arr)) if len(u_arr) > 0 else float("nan")

    return {
        "val_fused_bce": mean_loss,
        "val_auc": auc,
        "val_mean_u": mean_u,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml",
        type=str,
        default="configs/mm_transformer_tmc/transformer_tmc.yaml",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.yaml)
    seed_everything(cfg["train"]["seed"])

    device = torch.device(cfg["train"]["device"])
    out_root = Path(cfg["train"]["out_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(cfg)
    fold_df = attach_fold_ids(dataset, cfg["data"]["fold_index_csv"])
    subject_ids = infer_subject_ids(dataset)

    sid_to_fold = {
        str(r["subject_id"]): int(r["test_fold_id"])
        for _, r in fold_df.iterrows()
    }

    test_fold_ids = []
    for sid in subject_ids:
        if sid not in sid_to_fold:
            raise KeyError(f"subject_id {sid} not found in fold_index.csv")
        test_fold_ids.append(sid_to_fold[sid])

    all_records = []

    for fold_id in sorted(set(test_fold_ids)):
        test_idx = [i for i, f in enumerate(test_fold_ids) if f == fold_id]
        train_pool_idx = [i for i, f in enumerate(test_fold_ids) if f != fold_id]

        train_idx, val_idx = split_train_val(
            train_indices=train_pool_idx,
            subject_ids=subject_ids,
            val_ratio=cfg["train"]["val_ratio"],
            seed=cfg["train"]["seed"] + fold_id,
        )

        train_loader = make_loader(dataset, train_idx, cfg["train"]["batch_size"], shuffle=True)
        val_loader = make_loader(dataset, val_idx, cfg["train"]["batch_size"], shuffle=False)

        model = MMTransformerTMC(
            backbone_kwargs=cfg["model"]["backbone_kwargs"],
            n_classes=2,
            evidence_hidden_dim=cfg["model"]["evidence_hidden_dim"],
            evidence_dropout=cfg["model"]["evidence_dropout"],
            use_fused_feature_view=cfg["model"]["use_fused_feature_view"],
        ).to(device)

        y_train = []
        for idx in train_idx:
            item = dataset[idx]
            y_train.append(int(item["y"]))
        y_train = np.asarray(y_train)

        n_pos = max(int((y_train == 1).sum()), 1)
        n_neg = max(int((y_train == 0).sum()), 1)
        dynamic_pos_weight = n_neg / n_pos

        pos_weight_mode = cfg["train"].get("pos_weight_mode", "dynamic")
        pos_weight_cap = float(cfg["train"].get("pos_weight_cap", 6.0))

        if pos_weight_mode == "fixed":
            pos_weight = float(cfg["train"].get("pos_weight", 1.0))
        elif pos_weight_mode == "dynamic_cap":
            pos_weight = min(dynamic_pos_weight, pos_weight_cap)
        else:
            pos_weight = dynamic_pos_weight

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["train"]["lr"],
            weight_decay=cfg["train"]["weight_decay"],
        )

        best_auc = -1.0
        best_state = None
        history = []

        for epoch in range(1, cfg["train"]["epochs"] + 1):
            model.train()

            tr_total = tr_fused = tr_view = tr_reg = 0.0
            n_batches = 0

            for batch in train_loader:
                batch = batch_to_device(batch, device)

                optimizer.zero_grad()

                out = model(
                    x_scale=batch["x_scale"],
                    x_sc=batch["x_sc"],
                    x_fc=batch["x_fc"],
                    modality_mask=batch.get("modality_mask", None),
                )

                y = batch["y"].long()

                loss_out = compute_tmc_loss(
                    fused_prob=out.probs,
                    view_probs=out.view_probs,
                    view_evidences=out.view_evidences,
                    target=y,
                    pos_weight=pos_weight,
                    lambda_view=cfg["train"]["lambda_view"],
                    lambda_evidence=cfg["train"]["lambda_evidence"],
                )

                loss_out.total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                optimizer.step()

                tr_total += loss_out.total.item()
                tr_fused += loss_out.fused_bce.item()
                tr_view += loss_out.view_bce.item()
                tr_reg += loss_out.evidence_reg.item()
                n_batches += 1

            val_metrics = evaluate_epoch(model, val_loader, device)

            record = {
                "fold_id": fold_id,
                "epoch": epoch,
                "train_loss": tr_total / max(n_batches, 1),
                "train_fused_bce": tr_fused / max(n_batches, 1),
                "train_view_bce": tr_view / max(n_batches, 1),
                "train_e_reg": tr_reg / max(n_batches, 1),
                **val_metrics,
            }
            history.append(record)

            print(
                f"[fold {fold_id}] epoch {epoch:03d} | "
                f"train_loss={record['train_loss']:.4f} | "
                f"val_auc={record['val_auc']:.4f} | "
                f"val_fused_bce={record['val_fused_bce']:.4f} | "
                f"val_mean_u={record['val_mean_u']:.4f}"
            )

            score = record["val_auc"]
            if np.isnan(score):
                score = -1.0

            if score > best_auc:
                best_auc = score
                best_state = {
                    "model_state_dict": model.state_dict(),
                    "fold_id": fold_id,
                    "best_val_auc": best_auc,
                    "cfg": cfg,
                }

        fold_dir = out_root / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        hist_df = pd.DataFrame(history)
        hist_df.to_csv(fold_dir / "history.csv", index=False)

        if best_state is None:
            raise RuntimeError(f"fold {fold_id} failed to produce a best checkpoint.")

        ckpt_path = fold_dir / "model.pt"
        torch.save(best_state, ckpt_path)

        print(f"[fold {fold_id}] saved best checkpoint to {ckpt_path}")
        all_records.extend(history)

    pd.DataFrame(all_records).to_csv(out_root / "all_history.csv", index=False)
    with open(out_root / "done.json", "w", encoding="utf-8") as f:
        json.dump({"status": "ok"}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()