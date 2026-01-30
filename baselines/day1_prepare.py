###输入：python -m baselines.day1_prepare --yaml configs/week2_day1.yaml



from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


@dataclass
class Day1Config:
    cohort_csv: str
    out_dir: str
    require_eligible: bool = True
    drop_qc_excluded: bool = True
    require_modality_mask: Optional[str] = "111"
    labels_allowed: List[int] = None
    n_splits: int = 5
    seed: int = 42
    subject_id_col: str = "subject_id"
    label_col: str = "label_convert"


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_config_from_yaml(yaml_path: str) -> Dict:
    # 轻量：不强依赖 pyyaml，避免环境问题
    # 允许你也可以不使用 yaml，直接在 main 里传参
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "未安装 pyyaml。你可以：pip install pyyaml，或直接不用yaml，改成命令行传参。"
        ) from e

    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def filter_cohort(df: pd.DataFrame, cfg: Day1Config) -> pd.DataFrame:
    df0 = df.copy()

    # label 过滤
    if cfg.labels_allowed is None:
        cfg.labels_allowed = [0, 1]
    df0 = df0[df0[cfg.label_col].isin(cfg.labels_allowed)]

    # eligible / qc
    if cfg.require_eligible and "eligible_flag" in df0.columns:
        df0 = df0[df0["eligible_flag"] == 1]
    if cfg.drop_qc_excluded and "qc_exclude_flag" in df0.columns:
        df0 = df0[df0["qc_exclude_flag"] == 0]

    # modality_mask
    if cfg.require_modality_mask is not None and "modality_mask" in df0.columns:
        df0 = df0[df0["modality_mask"].astype(str) == str(cfg.require_modality_mask)]

    df0 = df0.reset_index(drop=True)
    return df0


def compute_stats(df: pd.DataFrame, cfg: Day1Config) -> Dict:
    y = df[cfg.label_col].astype(int).to_numpy()
    n = int(len(df))
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    pos_rate = float(pos / max(1, n))

    stats = {
        "n_samples": n,
        "n_pos": pos,
        "n_neg": neg,
        "pos_rate": pos_rate,
        "filters": {
            "require_eligible": cfg.require_eligible,
            "drop_qc_excluded": cfg.drop_qc_excluded,
            "require_modality_mask": cfg.require_modality_mask,
            "labels_allowed": cfg.labels_allowed,
        },
        "columns_present": list(df.columns),
    }

    if "modality_mask" in df.columns:
        stats["modality_mask_counts"] = df["modality_mask"].astype(str).value_counts(dropna=False).to_dict()

    # 可选：质量字段摘要（若存在）
    for col in ["dvars", "homogeneity", "quality_score", "delta_t_years"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) > 0:
                stats[f"{col}_summary"] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "p10": float(vals.quantile(0.10)),
                    "p50": float(vals.quantile(0.50)),
                    "p90": float(vals.quantile(0.90)),
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                }
            else:
                stats[f"{col}_summary"] = "all_nan_or_missing"

    return stats


def make_group_folds(df: pd.DataFrame, cfg: Day1Config):
    # GroupKFold 是确定性的，不依赖 seed；这里留 seed 只是为了与后续模块统一
    gkf = GroupKFold(n_splits=cfg.n_splits)

    idx = np.arange(len(df))
    groups = df[cfg.subject_id_col].astype(str).to_numpy()

    folds = []
    for fold_id, (tr, te) in enumerate(gkf.split(idx, groups=groups)):
        folds.append((fold_id, tr, te))
    return folds


def leakage_check(train_subjects: set, test_subjects: set):
    inter = train_subjects & test_subjects
    if inter:
        raise RuntimeError(f"Leakage detected! train∩test has {len(inter)} subjects, e.g. {list(sorted(inter))[:10]}")


def fold_label_sanity(df: pd.DataFrame, te_idx: np.ndarray, cfg: Day1Config) -> str:
    y = df.iloc[te_idx][cfg.label_col].astype(int)
    vc = y.value_counts().to_dict()
    # 提示：若某折 test 没有正类/负类，AUC 会变成 nan 或无意义
    if 0 not in vc or 1 not in vc:
        return f"WARNING: test fold label distribution={vc} (AUC可能无意义)"
    return f"OK: test fold label distribution={vc}"


def save_folds(df: pd.DataFrame, folds, cfg: Day1Config):
    out_dir = cfg.out_dir
    fold_dir = os.path.join(out_dir, "folds")
    _ensure_dir(fold_dir)

    # 给每一行样本标注 fold（test_fold_id，非test为-1）
    fold_index = df[[cfg.subject_id_col, cfg.label_col]].copy()
    fold_index["test_fold_id"] = -1

    for fold_id, tr, te in folds:
        tr_sub = set(df.iloc[tr][cfg.subject_id_col].astype(str).tolist())
        te_sub = set(df.iloc[te][cfg.subject_id_col].astype(str).tolist())
        leakage_check(tr_sub, te_sub)

        # 保存 subject 列表（审计/复现关键）
        with open(os.path.join(fold_dir, f"fold_{fold_id}_train_subjects.txt"), "w", encoding="utf-8") as f:
            for s in sorted(tr_sub):
                f.write(s + "\n")
        with open(os.path.join(fold_dir, f"fold_{fold_id}_test_subjects.txt"), "w", encoding="utf-8") as f:
            for s in sorted(te_sub):
                f.write(s + "\n")

        # 标记 test_fold_id
        fold_index.loc[te, "test_fold_id"] = fold_id

        print(f"[FOLD {fold_id}] train_subjects={len(tr_sub)} test_subjects={len(te_sub)} | {fold_label_sanity(df, te, cfg)}")

    fold_index_path = os.path.join(out_dir, "fold_index.csv")
    fold_index.to_csv(fold_index_path, index=False)
    print("[OK] wrote:", fold_index_path)


def run_day1(cfg: Day1Config):
    _ensure_dir(cfg.out_dir)

    df = pd.read_csv(cfg.cohort_csv)
    print("[LOAD] cohort_csv:", cfg.cohort_csv, "N =", len(df))

    df_f = filter_cohort(df, cfg)
    print("[FILTER] N =", len(df_f))
    if len(df_f) == 0:
        raise RuntimeError("过滤后样本数为0，请检查 eligible_flag/qc_exclude_flag/modality_mask/label_convert 等字段与规则。")

    # 保存过滤后的队列（第2周所有 baseline 都只用它）
    cohort_filtered_path = os.path.join(cfg.out_dir, "cohort_filtered.csv")
    df_f.to_csv(cohort_filtered_path, index=False)
    print("[OK] wrote:", cohort_filtered_path)

    # 统计并保存
    stats = compute_stats(df_f, cfg)
    stats_path = os.path.join(cfg.out_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("[OK] wrote:", stats_path)

    print("[STATS] n_samples =", stats["n_samples"], "pos_rate =", round(stats["pos_rate"], 4))

    # 切分并保存折信息
    folds = make_group_folds(df_f, cfg)
    save_folds(df_f, folds, cfg)

    print("\n[DAY1 DONE] Next: Day2 开始跑 Scale-only 的 LogReg/SVM baseline（读 cohort_filtered.csv + fold_index.csv）。")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort_csv", type=str, required=False)
    ap.add_argument("--out_dir", type=str, required=False)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--require_modality_mask", type=str, default="111")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--yaml", type=str, default=None, help="可选：传入 configs/week2_day1.yaml")

    args = ap.parse_args()

    if args.yaml:
        cfg_dict = load_config_from_yaml(args.yaml)
        cfg = Day1Config(
            cohort_csv=cfg_dict["cohort_csv"],
            out_dir=cfg_dict["out_dir"],
            require_eligible=cfg_dict.get("require_eligible", True),
            drop_qc_excluded=cfg_dict.get("drop_qc_excluded", True),
            require_modality_mask=cfg_dict.get("require_modality_mask", "111"),
            labels_allowed=cfg_dict.get("labels_allowed", [0, 1]),
            n_splits=int(cfg_dict.get("n_splits", args.n_splits)),
            seed=int(cfg_dict.get("seed", args.seed)),
            subject_id_col=cfg_dict.get("subject_id_col", "subject_id"),
            label_col=cfg_dict.get("label_col", "label_convert"),
        )
    else:
        if not args.cohort_csv or not args.out_dir:
            raise SystemExit("你要么传 --yaml configs/week2_day1.yaml，要么同时传 --cohort_csv 和 --out_dir。")
        cfg = Day1Config(
            cohort_csv=args.cohort_csv,
            out_dir=args.out_dir,
            n_splits=args.n_splits,
            seed=args.seed,
            require_modality_mask=args.require_modality_mask,
            labels_allowed=[0, 1],
        )

    run_day1(cfg)


if __name__ == "__main__":
    main()
