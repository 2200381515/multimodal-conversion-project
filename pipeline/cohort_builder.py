# pipeline/cohort_builder.py
from __future__ import annotations
import argparse
import os
import pandas as pd

from pipeline.labeling import LabelingConfig, compute_label_and_time
from pipeline.qc import QCConfig, qc_decide


REQUIRED_COLS = ["subject_id", "sc_path", "fc_path", "scale_path", "t0_date", "followup_end_date", "conversion_date"]


def build(args):
    df = pd.read_csv(args.in_csv)
    # 基本列存在性（可按你们真实列名改）
    for c in ["subject_id"]:
        if c not in df.columns:
            raise ValueError(f"missing column: {c}")

    # 标签/纳排
    lcfg = LabelingConfig(
        window_years=args.window_years,
        include_boundary=not args.window_open,
        phase1_drop_insufficient_followup=not args.keep_insufficient_followup
    )

    labels, dts, eligible, lreason = [], [], [], []
    for _, row in df.iterrows():
        # ✅ 优先使用数据集中已提供的 label_convert（例如来自 DATASET-DIAG2）
        raw = row.get("label_convert", None)
        if raw is not None and not (isinstance(raw, float) and pd.isna(raw)) and str(raw).strip() != "":
            try:
                y0 = int(float(raw))
                if y0 in (0, 1):
                    labels.append(y0)
                    dts.append(None)  # 没有日期就先不算 delta_t
                    eligible.append(1)
                    lreason.append("provided_label_convert")
                    continue
            except Exception:
                pass

        # 否则走原来的“按日期窗打标签”逻辑
        y, dt, eg, rs = compute_label_and_time(
            row.get("t0_date", None),
            row.get("followup_end_date", None),
            row.get("conversion_date", None),
            lcfg
        )
        labels.append(y)
        dts.append(dt)
        eligible.append(eg)
        lreason.append(rs)

    df["label_convert"] = labels
    df["delta_t_years"] = dts
    df["eligible_flag"] = eligible
    df["labeling_reason"] = lreason

    # QC
    qcfg = QCConfig(
        dvars_threshold=args.dvars_threshold,
        homogeneity_min=args.homogeneity_min,
        enable_dvars=not args.disable_dvars,
        enable_homogeneity=not args.disable_homogeneity,
        manual_exclude_col=args.manual_exclude_col
    )

    qc_ex, qs, qreason = [], [], []
    for _, row in df.iterrows():
        ex, score, rs = qc_decide(row, qcfg)
        qc_ex.append(ex)
        qs.append(score)
        qreason.append(rs)

    df["qc_exclude_flag"] = qc_ex
    df["quality_score"] = qs
    df["qc_reason"] = qreason

    # modality_mask：默认三模态齐全=111；缺失按文件路径空/NaN判
    def _exists(p):
        return isinstance(p, str) and len(p.strip()) > 0 and os.path.exists(p)

    mask_list = []
    for _, row in df.iterrows():
        has_scale = _exists(row.get("scale_path", "")) or (isinstance(row.get("scale_path", ""), str) and row.get("scale_path", "").strip() != "")
        has_sc = _exists(row.get("sc_path", "")) or (isinstance(row.get("sc_path", ""), str) and row.get("sc_path", "").strip() != "")
        has_fc = _exists(row.get("fc_path", "")) or (isinstance(row.get("fc_path", ""), str) and row.get("fc_path", "").strip() != "")
        mask_list.append(f"{int(has_scale)}{int(has_sc)}{int(has_fc)}")
    df["modality_mask"] = mask_list

    # 生成 qc_report.csv（每人一行，可追溯原因）
    qc_report = df[[
        "subject_id", "modality_mask",
        "dvars", "homogeneity",
        "qc_exclude_flag", "qc_reason",
        "eligible_flag", "labeling_reason",
        "label_convert", "delta_t_years"
    ]].copy()

    os.makedirs(args.out_dir, exist_ok=True)
    qc_report_path = os.path.join(args.out_dir, "qc_report.csv")
    cohort_out_path = os.path.join(args.out_dir, "cohort_table_built.csv")

    qc_report.to_csv(qc_report_path, index=False)
    df.to_csv(cohort_out_path, index=False)

    print("[OK] wrote:", qc_report_path)
    print("[OK] wrote:", cohort_out_path)

    # 快速统计（给你验收用：shape/缺失/label分布）
    print("\n[STATS]")
    print("N total:", len(df))
    print("eligible:", int(df["eligible_flag"].sum()))
    print("qc_excluded:", int(df["qc_exclude_flag"].sum()))
    print("label=1 count:", int((df["label_convert"] == 1).sum()))
    print("label=0 count:", int((df["label_convert"] == 0).sum()))
    print("mask counts:\n", df["modality_mask"].value_counts(dropna=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="raw cohort_table.csv from data team")
    ap.add_argument("--out_dir", required=True, help="output folder")
    ap.add_argument("--window_years", type=float, default=5.0)
    ap.add_argument("--window_open", action="store_true", help="use < window instead of <= window")
    ap.add_argument("--keep_insufficient_followup", action="store_true", help="Phase-1: keep insufficient followup (NOT recommended)")
    ap.add_argument("--dvars_threshold", type=float, default=50.0)
    ap.add_argument("--homogeneity_min", type=float, default=0.0)
    ap.add_argument("--disable_dvars", action="store_true")
    ap.add_argument("--disable_homogeneity", action="store_true")
    ap.add_argument("--manual_exclude_col", type=str, default="qc_manual_exclude")
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    main()
