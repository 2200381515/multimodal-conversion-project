# baselines/run_week2.py

from __future__ import annotations

import os
from typing import Any, Dict, List, Union, Optional

import pandas as pd
import numpy as np

from baselines.featurize import build_scale_matrix, build_sc_matrix, build_fc_matrix
from baselines.train_sklearn import train_logreg, train_svm_rbf, predict_prob_positive
from baselines.metrics import binary_metrics


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _summarize_mean_std(df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    rows = []
    for model, g in df.groupby("model"):
        r = {"model": model}
        for c in metric_cols:
            v = pd.to_numeric(g[c], errors="coerce")
            r[f"{c}_mean"] = float(v.mean())
            r[f"{c}_std"] = float(v.std())
        rows.append(r)
    return pd.DataFrame(rows)


def _load_day1_outputs(day1_dir: str):
    cohort_path = os.path.join(day1_dir, "cohort_filtered.csv")
    fold_index_path = os.path.join(day1_dir, "fold_index.csv")

    if not os.path.exists(cohort_path):
        raise FileNotFoundError(f"missing: {cohort_path}")
    if not os.path.exists(fold_index_path):
        raise FileNotFoundError(f"missing: {fold_index_path}")

    df = pd.read_csv(cohort_path)
    fold_index = pd.read_csv(fold_index_path)

    if len(df) != len(fold_index):
        raise RuntimeError("cohort_filtered 与 fold_index 行数不一致，请确认 Day1 输出一致。")
    if "test_fold_id" not in fold_index.columns:
        raise RuntimeError("fold_index.csv 缺少 test_fold_id 列。")

    test_fold_ids = fold_index["test_fold_id"].astype(int).to_numpy()
    n_folds = int(test_fold_ids.max() + 1)

    for col in ["label_convert", "subject_id"]:
        if col not in df.columns:
            raise RuntimeError(f"cohort_filtered.csv 缺少 {col} 列。")

    y = df["label_convert"].astype(int).to_numpy()
    subject_id = df["subject_id"].astype(str).to_numpy()

    return df, y, subject_id, test_fold_ids, n_folds


def _get_feature_matrix(
    feature: str,
    df: pd.DataFrame,
    vectorize: str,
) -> np.ndarray:
    """
    feature:
      - scale
      - sc
      - fc
      - concat: [scale, sc, fc] 拼接
    """
    feature = feature.lower().strip()
    vectorize = vectorize.lower().strip()

    if feature == "scale":
        if "scale_path" not in df.columns:
            raise RuntimeError("cohort_filtered.csv 缺少 scale_path 列。")
        return build_scale_matrix(df, scale_col="scale_path")

    if feature == "sc":
        if "sc_path" not in df.columns:
            raise RuntimeError("cohort_filtered.csv 缺少 sc_path 列（SC-only 需要）。")
        return build_sc_matrix(df, sc_col="sc_path", vectorize=vectorize)

    if feature == "fc":
        if "fc_path" not in df.columns:
            raise RuntimeError("cohort_filtered.csv 缺少 fc_path 列（FC-only 需要）。")
        return build_fc_matrix(df, fc_col="fc_path", vectorize=vectorize)

    if feature == "concat":
        # 必须三模态都在
        for col in ["scale_path", "sc_path", "fc_path"]:
            if col not in df.columns:
                raise RuntimeError(f"cohort_filtered.csv 缺少 {col} 列（concat 需要）。")

        X_scale = build_scale_matrix(df, scale_col="scale_path")
        X_sc = build_sc_matrix(df, sc_col="sc_path", vectorize=vectorize)
        X_fc = build_fc_matrix(df, fc_col="fc_path", vectorize=vectorize)
        return np.concatenate([X_scale, X_sc, X_fc], axis=1)

    raise ValueError(f"unknown feature={feature}, use scale/sc/fc/concat")


def _train_svm_linear_with_optional_pca(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    pca_components: Optional[int] = None,
):
    """
    Concat 特征维度通常很高，默认用线性核更稳更快；
    可选加 PCA 以降低维度（pca_components=None 表示不做 PCA）。
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.decomposition import PCA

    steps = [
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ]
    if pca_components is not None and int(pca_components) > 0:
        steps.append(("pca", PCA(n_components=int(pca_components), random_state=42)))

    steps.append(("svm", SVC(
        kernel="linear",
        C=1.0,
        probability=True,
        class_weight="balanced"
    )))

    clf = Pipeline(steps)
    clf.fit(Xtr, ytr)
    return clf


def _train_and_predict(
    model_name: str,
    feature: str,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    pca_components: Optional[int] = None,
) -> np.ndarray:
    """
    约定：
      - *_SVM 用 SVM
      - 其余用 LogReg
    备注：
      - concat + SVM 默认使用 linear（可选 PCA），其他场景仍沿用 train_svm_rbf。
    """
    if model_name.lower().endswith("_svm"):
        if feature.lower().strip() == "concat":
            clf = _train_svm_linear_with_optional_pca(Xtr, ytr, pca_components=pca_components)
        else:
            clf = train_svm_rbf(Xtr, ytr)
    else:
        clf = train_logreg(Xtr, ytr)

    return predict_prob_positive(clf, Xte)


def run(
    day1_dir: str,
    out_dir: str,
    feature: str,
    models_csv: str,
    tag: str,
    thr: float = 0.5,
    vectorize: str = "upper",
    pca_components: Optional[int] = None,
):
    _ensure_dir(out_dir)
    preds_dir = os.path.join(out_dir, "preds")
    _ensure_dir(preds_dir)

    df, y, subject_id, test_fold_ids, n_folds = _load_day1_outputs(day1_dir)
    X = _get_feature_matrix(feature, df, vectorize=vectorize)

    models = [m.strip() for m in models_csv.split(",") if m.strip()]
    if not models:
        raise ValueError("未指定模型，请使用 YAML 的 models 或命令行 --models ModelA,ModelB")

    all_rows = []
    metric_cols = ["AUC", "F1", "ACC", "SEN", "SPE"]

    for model_name in models:
        for fold_id in range(n_folds):
            te_mask = (test_fold_ids == fold_id)
            tr_mask = ~te_mask

            Xtr, ytr = X[tr_mask], y[tr_mask]
            Xte, yte = X[te_mask], y[te_mask]
            sub_te = subject_id[te_mask]

            prob = _train_and_predict(
                model_name=model_name,
                feature=feature,
                Xtr=Xtr,
                ytr=ytr,
                Xte=Xte,
                pca_components=pca_components,
            )

            pred_path = os.path.join(preds_dir, f"{model_name}_fold{fold_id}.csv")
            pd.DataFrame({
                "subject_id": sub_te,
                "y_true": yte,
                "y_prob": prob,
                "fold": fold_id,
                "model": model_name
            }).to_csv(pred_path, index=False)

            m = binary_metrics(yte, prob, thr=thr)
            m.update({
                "model": model_name,
                "fold": fold_id,
                "n_test": int(te_mask.sum()),
                "n_train": int(tr_mask.sum()),
                "pos_test": int((yte == 1).sum()),
                "neg_test": int((yte == 0).sum()),
                "feature": feature,
                "vectorize": vectorize,
                "tag": tag,
                "pca_components": (int(pca_components) if pca_components is not None else None),
            })
            all_rows.append(m)

            if m["pos_test"] == 0 or m["neg_test"] == 0:
                print(f"[WARN] {tag} | {model_name} fold={fold_id} test 单一类别 "
                      f"(pos={m['pos_test']} neg={m['neg_test']}), AUC 可能为 NaN。")
            print(f"[OK] {tag} | {model_name} fold={fold_id} "
                  f"AUC={m['AUC']:.4f} F1={m['F1']:.4f} SEN={m['SEN']:.4f} SPE={m['SPE']:.4f}")

    folds_df = pd.DataFrame(all_rows)
    folds_path = os.path.join(out_dir, f"{tag}_folds.csv")
    folds_df.to_csv(folds_path, index=False)

    summary_df = _summarize_mean_std(folds_df, metric_cols)
    summary_path = os.path.join(out_dir, f"{tag}_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("[DONE] wrote:", folds_path)
    print("[DONE] wrote:", summary_path)


# ---------------- YAML Support：方式A（单 YAML 单 run） ----------------

def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("未安装 pyyaml。请执行：pip install pyyaml") from e

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("YAML 顶层必须是 dict（键值对）。")
    if "runs" in cfg:
        raise ValueError("当前为方式A：单 YAML 单 run，不支持 runs 列表。")
    return cfg


def _normalize_models(models_field: Union[str, List[Any]]) -> str:
    if isinstance(models_field, str):
        return models_field
    if isinstance(models_field, list):
        return ",".join([str(x).strip() for x in models_field if str(x).strip()])
    raise ValueError(f"models 字段格式不支持：{type(models_field)}；请用字符串或列表。")


def run_from_yaml(yaml_path: str):
    cfg = _load_yaml(yaml_path)

    day1_dir = cfg["day1_dir"]
    out_dir = cfg["out_dir"]
    feature = cfg["feature"]
    tag = cfg["tag"]
    models_csv = _normalize_models(cfg["models"])

    thr = float(cfg.get("thr", 0.5))
    vectorize = str(cfg.get("vectorize", "upper"))
    pca_components = cfg.get("pca_components", None)
    if pca_components is not None:
        pca_components = int(pca_components)

    run(
        day1_dir=day1_dir,
        out_dir=out_dir,
        feature=feature,
        models_csv=models_csv,
        tag=tag,
        thr=thr,
        vectorize=vectorize,
        pca_components=pca_components,
    )


def main():
    import argparse
    ap = argparse.ArgumentParser()

    ap.add_argument("--yaml", type=str, default=None, help="方式A：单 YAML 单 run，例如 configs/week2_day4_concat.yaml")

    # 兼容命令行参数模式
    ap.add_argument("--day1_dir", required=False)
    ap.add_argument("--out_dir", required=False)
    ap.add_argument("--feature", required=False, choices=["scale", "sc", "fc", "concat"])
    ap.add_argument("--models", required=False, help="逗号分隔：ModelA,ModelB")
    ap.add_argument("--tag", required=False)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--vectorize", type=str, default="upper", choices=["upper", "flatten"])
    ap.add_argument("--pca_components", type=int, default=None, help="仅 concat+SVM 推荐设置，例如 256；不设置则不做 PCA")

    args = ap.parse_args()

    if args.yaml:
        run_from_yaml(args.yaml)
        return

    if not (args.day1_dir and args.out_dir and args.feature and args.models and args.tag):
        raise SystemExit("未提供 --yaml 时，必须同时提供 --day1_dir --out_dir --feature --models --tag")

    run(
        day1_dir=args.day1_dir,
        out_dir=args.out_dir,
        feature=args.feature,
        models_csv=args.models,
        tag=args.tag,
        thr=args.thr,
        vectorize=args.vectorize,
        pca_components=args.pca_components,
    )


if __name__ == "__main__":
    main()
