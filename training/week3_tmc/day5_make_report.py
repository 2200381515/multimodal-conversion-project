from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np

# pandas / matplotlib 可能在你的环境里都有；如果没有，脚本会提示如何处理
try:
    import pandas as pd
except Exception as e:
    raise RuntimeError(
        "Day5 requires pandas. Please install via: pip install pandas"
    ) from e

try:
    import matplotlib.pyplot as plt
except Exception as e:
    raise RuntimeError(
        "Day5 requires matplotlib. Please install via: pip install matplotlib"
    ) from e


@dataclass
class RunInputs:
    name: str
    ablation_csv: str


DEFAULT_BASELINE_ABLATION = os.path.join("results", "week3_tmc", "day4_missingness_ablation.csv")
DEFAULT_MODDROP_ABLATION = os.path.join("results", "week3_tmc_moddrop", "day4_missingness_ablation.csv")
DEFAULT_OUT_DIR = os.path.join("results", "week3_tmc", "day5_report")


EXPECTED_COLUMNS = [
    "condition", "fold",
    "val_n", "val_pos", "val_neg",
    "auc", "f1_05", "best_f1", "best_thr",
    "mean_u", "p90_u",
    "prob_mean", "prob_min", "prob_max",
]


def _safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)


def _read_ablation_csv(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Ablation CSV not found: {path}")

    df = pd.read_csv(path)

    # 基本列检查（允许额外列存在，比如 ckpt）
    missing = [c for c in ["condition", "fold", "auc", "best_f1", "mean_u"] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Ablation CSV missing required columns: {missing}\n"
            f"Got columns: {list(df.columns)}\n"
            f"File: {path}"
        )

    # 统一类型
    df["fold"] = df["fold"].astype(int)
    df["condition"] = df["condition"].astype(str)

    # 如果没有 f1_05/best_thr 等列，也允许（用 NaN 补）
    for c in EXPECTED_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan

    return df


def _summarize_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    """
    输出每个 condition 的 mean/std，以及 val_pos/val_neg 的均值（用于解释不平衡）。
    """
    g = df.groupby("condition", dropna=False)

    out = g.agg(
        folds=("fold", "nunique"),
        auc_mean=("auc", "mean"),
        auc_std=("auc", "std"),
        best_f1_mean=("best_f1", "mean"),
        best_f1_std=("best_f1", "std"),
        f1_05_mean=("f1_05", "mean"),
        mean_u_mean=("mean_u", "mean"),
        mean_u_std=("mean_u", "std"),
        p90_u_mean=("p90_u", "mean"),
        prob_mean_mean=("prob_mean", "mean"),
        prob_min_min=("prob_min", "min"),
        prob_max_max=("prob_max", "max"),
        val_pos_mean=("val_pos", "mean"),
        val_neg_mean=("val_neg", "mean"),
        val_n_mean=("val_n", "mean"),
    ).reset_index()

    # 排序：把 none 放第一，其余按 auc_mean 降序
    def sort_key(cond: str) -> Tuple[int, str]:
        return (0, cond) if cond == "none" else (1, cond)

    out = out.sort_values(by=["condition"], key=lambda s: s.map(sort_key)).reset_index(drop=True)
    return out


def _compare_runs_on_none(
    base_df: pd.DataFrame,
    other_df: pd.DataFrame,
    base_name: str,
    other_name: str
) -> pd.DataFrame:
    """
    对比两套 run 在 condition=none 下的 fold-level 指标变化。
    输出：auc/best_f1/mean_u 的差值（other - base）。
    """
    b = base_df[base_df["condition"] == "none"].copy()
    o = other_df[other_df["condition"] == "none"].copy()

    if b.empty or o.empty:
        # 可能用户只跑了部分 condition，或者文件不是 all
        return pd.DataFrame(
            {"warning": [f"Cannot compare none-condition between {base_name} and {other_name}: missing rows."]}
        )

    merged = pd.merge(
        b[["fold", "auc", "best_f1", "mean_u", "p90_u", "prob_mean"]],
        o[["fold", "auc", "best_f1", "mean_u", "p90_u", "prob_mean"]],
        on="fold",
        suffixes=(f"_{base_name}", f"_{other_name}"),
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame({"warning": [f"No overlapping folds found between {base_name} and {other_name}."]})

    merged["delta_auc"] = merged[f"auc_{other_name}"] - merged[f"auc_{base_name}"]
    merged["delta_best_f1"] = merged[f"best_f1_{other_name}"] - merged[f"best_f1_{base_name}"]
    merged["delta_mean_u"] = merged[f"mean_u_{other_name}"] - merged[f"mean_u_{base_name}"]
    merged["delta_p90_u"] = merged[f"p90_u_{other_name}"] - merged[f"p90_u_{base_name}"]
    merged["delta_prob_mean"] = merged[f"prob_mean_{other_name}"] - merged[f"prob_mean_{base_name}"]

    # 加一行总体均值
    mean_row = {"fold": "MEAN"}
    for c in ["delta_auc", "delta_best_f1", "delta_mean_u", "delta_p90_u", "delta_prob_mean"]:
        mean_row[c] = merged[c].mean()
    merged = pd.concat([merged, pd.DataFrame([mean_row])], ignore_index=True)

    return merged


def _plot_condition_bars(
    base_summary: pd.DataFrame,
    other_summary: Optional[pd.DataFrame],
    out_path: str,
    metric: str,
    title: str,
    base_label: str = "baseline",
    other_label: str = "moddrop",
):
    """
    画 condition-wise 柱状图：baseline vs moddrop（如果提供）。
    metric 例如：auc_mean, best_f1_mean, mean_u_mean
    """
    conditions = base_summary["condition"].tolist()
    x = np.arange(len(conditions))
    width = 0.35

    plt.figure(figsize=(10, 4))
    plt.bar(x - width / 2, base_summary[metric].values, width, label=base_label)

    if other_summary is not None:
        # 对齐条件（以 base 的为准，缺的填 NaN）
        other_map = {r["condition"]: r[metric] for _, r in other_summary.iterrows()}
        other_vals = [other_map.get(c, np.nan) for c in conditions]
        plt.bar(x + width / 2, other_vals, width, label=other_label)

    plt.xticks(x, conditions, rotation=20, ha="right")
    plt.title(title)
    plt.xlabel("condition")
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _write_markdown_report(
    out_path: str,
    base_name: str,
    base_csv: str,
    base_summary: pd.DataFrame,
    other_name: Optional[str],
    other_csv: Optional[str],
    other_summary: Optional[pd.DataFrame],
    compare_none: Optional[pd.DataFrame],
):
    def _fmt(x: float) -> str:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "NA"
        return f"{x:.4f}"

    def _row(summary: pd.DataFrame, cond: str) -> Optional[Dict]:
        m = summary[summary["condition"] == cond]
        if m.empty:
            return None
        return m.iloc[0].to_dict()

    base_none = _row(base_summary, "none")
    other_none = _row(other_summary, "none") if other_summary is not None else None

    lines = []
    lines.append("# Week3 Day5 Report (Auto Generated)\n")
    lines.append("## Inputs\n")
    lines.append(f"- baseline: **{base_name}**  \n  CSV: `{base_csv}`\n")
    if other_name and other_csv:
        lines.append(f"- other: **{other_name}**  \n  CSV: `{other_csv}`\n")

    lines.append("\n## Key Numbers (condition = none)\n")
    if base_none:
        lines.append(
            f"- baseline none: AUC={_fmt(base_none['auc_mean'])} ± {_fmt(base_none['auc_std'])}, "
            f"best_F1={_fmt(base_none['best_f1_mean'])} ± {_fmt(base_none['best_f1_std'])}, "
            f"mean_u={_fmt(base_none['mean_u_mean'])}\n"
        )
    if other_none:
        lines.append(
            f"- {other_name} none: AUC={_fmt(other_none['auc_mean'])} ± {_fmt(other_none['auc_std'])}, "
            f"best_F1={_fmt(other_none['best_f1_mean'])} ± {_fmt(other_none['best_f1_std'])}, "
            f"mean_u={_fmt(other_none['mean_u_mean'])}\n"
        )

    lines.append("\n## Condition-wise Summary\n")
    lines.append("You can find full tables in the CSV outputs.\n")

    lines.append("\n### Takeaways (Template)\n")
    lines.append(
        "- If `drop_sc` is identical to `none`, it usually indicates the SC view has near-zero contribution in fusion.\n"
        "- If `drop_fc` increases uncertainty and drives prob toward ~0.5, FC is acting as the dominant predictive view.\n"
        "- Large fold-to-fold variance is expected here because val set per fold is small and class-imbalanced.\n"
    )

    if compare_none is not None and not compare_none.empty and "warning" not in compare_none.columns:
        mean_row = compare_none[compare_none["fold"] == "MEAN"]
        if not mean_row.empty:
            r = mean_row.iloc[0].to_dict()
            lines.append("\n### Baseline vs Other (none-condition mean delta)\n")
            lines.append(
                f"- ΔAUC (other-baseline): {_fmt(r.get('delta_auc'))}\n"
                f"- Δbest_F1 (other-baseline): {_fmt(r.get('delta_best_f1'))}\n"
                f"- Δmean_u (other-baseline): {_fmt(r.get('delta_mean_u'))}\n"
            )

    lines.append("\n## Files Produced\n")
    lines.append("- `baseline_condition_summary.csv`\n")
    if other_summary is not None:
        lines.append("- `other_condition_summary.csv`\n")
        lines.append("- `compare_none_fold_level.csv`\n")
    lines.append("- `plot_auc_by_condition.png`, `plot_best_f1_by_condition.png`, `plot_mean_u_by_condition.png`\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def parse_args():
    p = argparse.ArgumentParser(description="Day5: summarize Day4 missingness ablation into tables + plots + report")
    p.add_argument("--baseline_csv", type=str, default=DEFAULT_BASELINE_ABLATION,
                   help="Baseline ablation CSV (e.g., results/week3_tmc/day4_missingness_ablation.csv)")
    p.add_argument("--other_csv", type=str, default=DEFAULT_MODDROP_ABLATION,
                   help="Other ablation CSV (optional), e.g., results/week3_tmc_moddrop/day4_missingness_ablation.csv")
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR,
                   help="Output directory for Day5 report artifacts")
    p.add_argument("--baseline_name", type=str, default="week3_tmc",
                   help="Label for baseline run")
    p.add_argument("--other_name", type=str, default="week3_tmc_moddrop",
                   help="Label for other run")
    p.add_argument("--skip_other", action="store_true",
                   help="If set, only summarize baseline CSV")
    return p.parse_args()


def main():
    args = parse_args()
    _safe_makedirs(args.out_dir)

    # --- baseline ---
    base_df = _read_ablation_csv(args.baseline_csv)
    base_summary = _summarize_by_condition(base_df)
    base_summary_path = os.path.join(args.out_dir, "baseline_condition_summary.csv")
    base_summary.to_csv(base_summary_path, index=False, encoding="utf-8-sig")

    # --- other (optional) ---
    other_df = None
    other_summary = None
    compare_none = None

    if not args.skip_other and args.other_csv and os.path.exists(args.other_csv):
        other_df = _read_ablation_csv(args.other_csv)
        other_summary = _summarize_by_condition(other_df)
        other_summary_path = os.path.join(args.out_dir, "other_condition_summary.csv")
        other_summary.to_csv(other_summary_path, index=False, encoding="utf-8-sig")

        compare_none = _compare_runs_on_none(
            base_df=base_df,
            other_df=other_df,
            base_name=args.baseline_name,
            other_name=args.other_name,
        )
        compare_none_path = os.path.join(args.out_dir, "compare_none_fold_level.csv")
        compare_none.to_csv(compare_none_path, index=False, encoding="utf-8-sig")

    # --- plots ---
    _plot_condition_bars(
        base_summary=base_summary,
        other_summary=other_summary,
        out_path=os.path.join(args.out_dir, "plot_auc_by_condition.png"),
        metric="auc_mean",
        title="AUC mean by condition",
        base_label=args.baseline_name,
        other_label=args.other_name,
    )
    _plot_condition_bars(
        base_summary=base_summary,
        other_summary=other_summary,
        out_path=os.path.join(args.out_dir, "plot_best_f1_by_condition.png"),
        metric="best_f1_mean",
        title="Best F1 mean by condition",
        base_label=args.baseline_name,
        other_label=args.other_name,
    )
    _plot_condition_bars(
        base_summary=base_summary,
        other_summary=other_summary,
        out_path=os.path.join(args.out_dir, "plot_mean_u_by_condition.png"),
        metric="mean_u_mean",
        title="Mean uncertainty by condition",
        base_label=args.baseline_name,
        other_label=args.other_name,
    )

    # --- report.md ---
    report_path = os.path.join(args.out_dir, "report.md")
    _write_markdown_report(
        out_path=report_path,
        base_name=args.baseline_name,
        base_csv=args.baseline_csv,
        base_summary=base_summary,
        other_name=None if (args.skip_other or other_summary is None) else args.other_name,
        other_csv=None if (args.skip_other or other_summary is None) else args.other_csv,
        other_summary=other_summary,
        compare_none=compare_none,
    )

    print("[Day5] Report generated:")
    print(f"  - {base_summary_path}")
    if other_summary is not None:
        print(f"  - {os.path.join(args.out_dir, 'other_condition_summary.csv')}")
        print(f"  - {os.path.join(args.out_dir, 'compare_none_fold_level.csv')}")
    print(f"  - {os.path.join(args.out_dir, 'plot_auc_by_condition.png')}")
    print(f"  - {os.path.join(args.out_dir, 'plot_best_f1_by_condition.png')}")
    print(f"  - {os.path.join(args.out_dir, 'plot_mean_u_by_condition.png')}")
    print(f"  - {report_path}")


if __name__ == "__main__":
    main()
