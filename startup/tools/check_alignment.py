# -*- coding: utf-8 -*-
"""
startup/tools/check_alignment.py

启动期对齐体检：
- 读取 cohort_table.csv（至少含 subject_id, sc_path, fc_path, scale_path）
- 抽样加载 sc/fc/scale
- 检查：
  1) 文件可读
  2) SC/FC 是否为方阵、维度一致
  3) 对称性误差
  4) NaN/Inf
  5) 数值摘要 (min/max/mean/std)
  6) scale 列一致性、关键字段是否存在 (DATASET-DIAG2 / INSOMNA 可选检查)
- 输出：
  out_dir/
    alignment_table.csv
    alignment_report.json
    roi_order_hash.txt
"""

import argparse
import json
import hashlib
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


# -------------------------
# IO helpers
# -------------------------
def load_matrix(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    suf = p.suffix.lower()
    if suf == ".npy":
        arr = np.load(p, allow_pickle=False)
        return np.asarray(arr)
    if suf == ".npz":
        z = np.load(p, allow_pickle=False)
        # pick the first array in npz
        keys = list(z.keys())
        if not keys:
            raise ValueError(f"Empty npz: {p}")
        return np.asarray(z[keys[0]])
    if suf in [".csv", ".tsv"]:
        sep = "," if suf == ".csv" else "\t"
        df = pd.read_csv(p, sep=sep, header=None)
        return df.values
    raise ValueError(f"Unsupported matrix format: {p}")


def load_scale(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    suf = p.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(p)
    if suf == ".tsv":
        return pd.read_csv(p, sep="\t")
    if suf in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    if suf == ".json":
        # allow either dict or list of dicts
        obj = pd.read_json(p)
        # normalize to one-row dataframe if it's a series/dict-like
        if isinstance(obj, pd.Series):
            return obj.to_frame().T
        return obj
    raise ValueError(f"Unsupported scale format: {p}")


# -------------------------
# checks
# -------------------------
def is_square_2d(a: np.ndarray) -> bool:
    return (a.ndim == 2) and (a.shape[0] == a.shape[1])


def symm_error(a: np.ndarray) -> float:
    if a.ndim != 2:
        return float("nan")
    if a.shape[0] != a.shape[1]:
        return float("nan")
    # max absolute difference to transpose
    diff = np.nanmax(np.abs(a - a.T))
    return float(diff)


def summary_stats(a: np.ndarray) -> dict:
    # ignore NaNs for stats
    a = np.asarray(a)
    out = {}
    out["shape"] = list(a.shape)
    out["dtype"] = str(a.dtype)

    out["nan_count"] = int(np.isnan(a).sum()) if np.issubdtype(a.dtype, np.floating) else 0
    out["inf_count"] = int(np.isinf(a).sum()) if np.issubdtype(a.dtype, np.floating) else 0

    # robust stats: if everything is nan -> nan
    with np.errstate(all="ignore"):
        out["min"] = float(np.nanmin(a)) if a.size else float("nan")
        out["max"] = float(np.nanmax(a)) if a.size else float("nan")
        out["mean"] = float(np.nanmean(a)) if a.size else float("nan")
        out["std"] = float(np.nanstd(a)) if a.size else float("nan")
    return out


def make_roi_order_hash(n_roi: int) -> str:
    """
    你现在 per-subject npy 没有 roi label 文件，
    所以这里用“n_roi + 上三角索引顺序合同”生成一个稳定 hash。
    后续如果你加入 roi_labels_path，就可以改为 hash label 文件内容。
    """
    iu = np.triu_indices(n_roi, k=1)
    payload = {
        "n_roi": int(n_roi),
        "upper_tri_first_10": [(int(iu[0][k]), int(iu[1][k])) for k in range(min(10, len(iu[0])))],
        "upper_tri_last_10": [(int(iu[0][-k-1]), int(iu[1][-k-1])) for k in range(min(10, len(iu[0])))],
        "count_edges": int(len(iu[0])),
    }
    s = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort_table", type=str, required=True, help="cohort_table.csv 路径（min10 或 full）")
    ap.add_argument("--out_dir", type=str, required=True, help="输出目录")
    ap.add_argument("--n_samples", type=int, default=10, help="抽样检查多少个样本（默认10）")
    ap.add_argument("--random_seed", type=int, default=0, help="抽样随机种子")
    ap.add_argument("--expected_n_roi", type=int, default=0, help="可选：期望 ROI 数（0 表示自动以首个成功样本为准）")
    ap.add_argument("--symm_tol", type=float, default=1e-3, help="对称性误差阈值（max|M-M^T|）")
    ap.add_argument("--require_scale_cols", type=str, default="DATASET-DIAG2,INSOMNA",
                    help="可选：要求 scale 必须包含的列（逗号分隔；为空则不检查）")
    args = ap.parse_args()

    cohort_path = Path(args.cohort_table).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cohort_path)
    # minimal columns
    for col in ["subject_id", "sc_path", "fc_path", "scale_path"]:
        if col not in df.columns:
            raise ValueError(f"Missing column in cohort_table: {col}")

    # sample
    df_use = df.copy()
    if args.n_samples > 0 and len(df_use) > args.n_samples:
        df_use = df_use.sample(n=args.n_samples, random_state=args.random_seed)

    require_cols = [c.strip() for c in args.require_scale_cols.split(",") if c.strip()]

    rows = []
    fail_reasons = Counter()

    inferred_n_roi = None
    inferred_scale_cols = None

    for _, r in df_use.iterrows():
        sid = str(r.get("subject_id", ""))
        sc_path = str(r.get("sc_path", "")).strip()
        fc_path = str(r.get("fc_path", "")).strip()
        scale_path = str(r.get("scale_path", "")).strip()

        rec = {
            "subject_id": sid,
            "sc_path": sc_path,
            "fc_path": fc_path,
            "scale_path": scale_path,

            "ok_sc": False,
            "ok_fc": False,
            "ok_scale": False,

            "sc_shape": "",
            "fc_shape": "",
            "scale_n_cols": "",

            "sc_symm_err": "",
            "fc_symm_err": "",

            "sc_nan": "",
            "sc_inf": "",
            "fc_nan": "",
            "fc_inf": "",

            "sc_min": "",
            "sc_max": "",
            "sc_mean": "",
            "sc_std": "",

            "fc_min": "",
            "fc_max": "",
            "fc_mean": "",
            "fc_std": "",

            "scale_missing_required_cols": "",
            "fail_reason": "",
        }

        # --- SC ---
        try:
            sc = load_matrix(sc_path) if sc_path else None
            if sc is None:
                raise FileNotFoundError("empty sc_path")
            st = summary_stats(sc)
            rec["sc_shape"] = str(st["shape"])
            rec["sc_nan"] = st["nan_count"]
            rec["sc_inf"] = st["inf_count"]
            rec["sc_min"] = st["min"]
            rec["sc_max"] = st["max"]
            rec["sc_mean"] = st["mean"]
            rec["sc_std"] = st["std"]

            if not is_square_2d(sc):
                raise ValueError(f"SC not square 2D: shape={sc.shape}")

            rec["sc_symm_err"] = symm_error(sc)

            if args.expected_n_roi and sc.shape[0] != args.expected_n_roi:
                raise ValueError(f"SC n_roi={sc.shape[0]} != expected {args.expected_n_roi}")

            if inferred_n_roi is None:
                inferred_n_roi = sc.shape[0]
            else:
                if sc.shape[0] != inferred_n_roi:
                    raise ValueError(f"SC n_roi={sc.shape[0]} != inferred baseline {inferred_n_roi}")

            if float(rec["sc_symm_err"]) > args.symm_tol:
                # 不直接判失败，但记录警告到 fail_reason（方便你看）
                rec["fail_reason"] += f"[WARN] SC symm_err {rec['sc_symm_err']} > tol {args.symm_tol};"

            rec["ok_sc"] = True

        except Exception as e:
            msg = f"SC_ERROR: {e}"
            rec["fail_reason"] += msg + ";"
            fail_reasons["SC_ERROR"] += 1

        # --- FC ---
        try:
            fc = load_matrix(fc_path) if fc_path else None
            if fc is None:
                raise FileNotFoundError("empty fc_path")
            st = summary_stats(fc)
            rec["fc_shape"] = str(st["shape"])
            rec["fc_nan"] = st["nan_count"]
            rec["fc_inf"] = st["inf_count"]
            rec["fc_min"] = st["min"]
            rec["fc_max"] = st["max"]
            rec["fc_mean"] = st["mean"]
            rec["fc_std"] = st["std"]

            if not is_square_2d(fc):
                raise ValueError(f"FC not square 2D: shape={fc.shape}")

            rec["fc_symm_err"] = symm_error(fc)

            if args.expected_n_roi and fc.shape[0] != args.expected_n_roi:
                raise ValueError(f"FC n_roi={fc.shape[0]} != expected {args.expected_n_roi}")

            if inferred_n_roi is None:
                inferred_n_roi = fc.shape[0]
            else:
                if fc.shape[0] != inferred_n_roi:
                    raise ValueError(f"FC n_roi={fc.shape[0]} != inferred baseline {inferred_n_roi}")

            if float(rec["fc_symm_err"]) > args.symm_tol:
                rec["fail_reason"] += f"[WARN] FC symm_err {rec['fc_symm_err']} > tol {args.symm_tol};"

            rec["ok_fc"] = True

        except Exception as e:
            msg = f"FC_ERROR: {e}"
            rec["fail_reason"] += msg + ";"
            fail_reasons["FC_ERROR"] += 1

        # --- SCALE ---
        try:
            sca = load_scale(scale_path) if scale_path else None
            if sca is None:
                raise FileNotFoundError("empty scale_path")

            # normalize: we expect 1 row per subject
            if len(sca) != 1:
                # 不直接失败，但记录警告（有些量表可能是一人多行）
                rec["fail_reason"] += f"[WARN] SCALE rows={len(sca)} (expected 1);"

            cols = list(sca.columns)
            rec["scale_n_cols"] = len(cols)

            if inferred_scale_cols is None:
                inferred_scale_cols = cols
            else:
                if cols != inferred_scale_cols:
                    rec["fail_reason"] += "[WARN] SCALE columns differ from baseline;"

            missing = [c for c in require_cols if c not in cols]
            if missing:
                rec["scale_missing_required_cols"] = ",".join(missing)
                # 不一定要失败（你可把 require_cols 设为空），这里按“提示”处理
                rec["fail_reason"] += f"[WARN] SCALE missing required cols: {missing};"

            rec["ok_scale"] = True

        except Exception as e:
            msg = f"SCALE_ERROR: {e}"
            rec["fail_reason"] += msg + ";"
            fail_reasons["SCALE_ERROR"] += 1

        # overall: if any critical failed
        rows.append(rec)

    out_table = out_dir / "alignment_table.csv"
    pd.DataFrame(rows).to_csv(out_table, index=False, encoding="utf-8-sig")

    # report summary
    ok_sc = sum(1 for x in rows if x["ok_sc"])
    ok_fc = sum(1 for x in rows if x["ok_fc"])
    ok_scale = sum(1 for x in rows if x["ok_scale"])

    report = {
        "cohort_table": str(cohort_path),
        "n_checked": len(rows),
        "ok_sc": ok_sc,
        "ok_fc": ok_fc,
        "ok_scale": ok_scale,
        "inferred_n_roi": int(inferred_n_roi) if inferred_n_roi is not None else None,
        "expected_n_roi_arg": int(args.expected_n_roi),
        "symm_tol": float(args.symm_tol),
        "require_scale_cols": require_cols,
        "fail_reason_counts": dict(fail_reasons),
        "notes": [
            "alignment_table.csv is the main per-subject check output.",
            "WARN entries in fail_reason indicate non-fatal issues you may still want to fix.",
        ],
    }

    out_report = out_dir / "alignment_report.json"
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # roi order hash
    if inferred_n_roi is not None:
        roi_hash = make_roi_order_hash(inferred_n_roi)
        (out_dir / "roi_order_hash.txt").write_text(
            f"n_roi={inferred_n_roi}\nroi_order_hash_sha256={roi_hash}\n",
            encoding="utf-8"
        )

    print(f"[DONE] wrote: {out_table}")
    print(f"[DONE] wrote: {out_report}")
    if inferred_n_roi is not None:
        print(f"[DONE] wrote: {out_dir / 'roi_order_hash.txt'}")
    print(f"[INFO] ok_sc={ok_sc}/{len(rows)}, ok_fc={ok_fc}/{len(rows)}, ok_scale={ok_scale}/{len(rows)}")


if __name__ == "__main__":
    main()
