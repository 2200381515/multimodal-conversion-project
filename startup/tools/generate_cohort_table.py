# -*- coding: utf-8 -*-
"""
generate_cohort_table.py
自动扫描 data_root 生成 cohort_table.csv（最小可用版）

默认假设目录结构类似：
data_root/
  sub-0001/  (或任意文件夹名，视为一个subject)
    sc.npy / sc.npz / sc.csv ...
    fc.npy / fc.npz / fc.csv ...
    scale.csv / scale.json ...
  sub-0002/
    ...

你可以通过参数指定关键词来匹配 sc/fc/scale。
"""

import argparse
import os
from pathlib import Path
import re
import pandas as pd

DEFAULT_SC_KEYS = ["sc", "struct", "structural", "connectome_sc"]
DEFAULT_FC_KEYS = ["fc", "func", "functional", "connectome_fc"]
DEFAULT_SCALE_KEYS = ["scale", "clinical", "cog", "mmse", "moca", "cdr", "faq"]

MATRIX_EXTS = [".npy", ".npz", ".csv", ".tsv"]
SCALE_EXTS = [".csv", ".json", ".xlsx", ".xls"]

def _lower(s: str) -> str:
    return s.lower() if isinstance(s, str) else ""

def find_first_match(files, keys, exts):
    """
    files: List[Path]
    keys: keywords list
    exts: allowed ext list
    """
    cand = []
    for f in files:
        name = _lower(f.name)
        if f.suffix.lower() not in exts:
            continue
        if any(k in name for k in keys):
            cand.append(f)
    # 优先更短文件名（更像主文件），其次按字母排序
    cand.sort(key=lambda p: (len(p.name), p.name))
    return cand[0] if cand else None

def is_subject_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    # 里面至少有一个矩阵或量表文件就算候选
    for f in p.iterdir():
        if f.is_file() and (f.suffix.lower() in MATRIX_EXTS + SCALE_EXTS):
            return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="数据根目录（包含各subject子文件夹）")
    ap.add_argument("--out_csv", type=str, default="cohort_table.csv", help="输出 cohort_table.csv 路径")
    ap.add_argument("--limit", type=int, default=0, help="只输出前 N 个subject（0表示全量）。启动期可设10")
    ap.add_argument("--sc_keys", type=str, default=",".join(DEFAULT_SC_KEYS), help="匹配SC文件名关键词（逗号分隔）")
    ap.add_argument("--fc_keys", type=str, default=",".join(DEFAULT_FC_KEYS), help="匹配FC文件名关键词（逗号分隔）")
    ap.add_argument("--scale_keys", type=str, default=",".join(DEFAULT_SCALE_KEYS), help="匹配量表文件名关键词（逗号分隔）")
    ap.add_argument("--subject_regex", type=str, default="", help="可选：只把匹配该正则的文件夹视为subject（例如 '^sub-'）")
    args = ap.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    sc_keys = [k.strip().lower() for k in args.sc_keys.split(",") if k.strip()]
    fc_keys = [k.strip().lower() for k in args.fc_keys.split(",") if k.strip()]
    scale_keys = [k.strip().lower() for k in args.scale_keys.split(",") if k.strip()]

    subj_re = re.compile(args.subject_regex) if args.subject_regex else None

    subj_dirs = []
    for p in sorted(data_root.iterdir(), key=lambda x: x.name):
        if not is_subject_dir(p):
            continue
        if subj_re and (subj_re.search(p.name) is None):
            continue
        subj_dirs.append(p)

    if not subj_dirs:
        raise RuntimeError(
            "No subject directories found. "
            "Check --data_root or use --subject_regex to match your folder names."
        )

    if args.limit and args.limit > 0:
        subj_dirs = subj_dirs[: args.limit]

    rows = []
    for sd in subj_dirs:
        files = [f for f in sd.iterdir() if f.is_file()]

        sc = find_first_match(files, sc_keys, MATRIX_EXTS)
        fc = find_first_match(files, fc_keys, MATRIX_EXTS)
        scale = find_first_match(files, scale_keys, SCALE_EXTS)

        # 可选：ROI labels
        roi_labels = None
        for f in files:
            if f.suffix.lower() in [".txt", ".csv"] and "roi" in _lower(f.name) and "label" in _lower(f.name):
                roi_labels = f
                break

        rows.append({
            "subject_id": sd.name,
            "sc_path": str(sc) if sc else "",
            "fc_path": str(fc) if fc else "",
            "scale_path": str(scale) if scale else "",
            "roi_labels_path": str(roi_labels) if roi_labels else "",
            # 下面这些启动期可以先空着，等数据方补齐
            "label_convert": "",
            "t0_date": "",
            "followup_end_date": "",
            "conversion_date": "",
            "dvars": "",
            "homogeneity": "",
        })

    out_csv = Path(args.out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"[DONE] wrote cohort_table: {out_csv}")
    print(f"[INFO] subjects: {len(rows)}")
    # 给启动期一个“最低可用提醒”
    missing_sc = sum(1 for r in rows if not r["sc_path"])
    missing_fc = sum(1 for r in rows if not r["fc_path"])
    print(f"[WARN] missing sc_path: {missing_sc}, missing fc_path: {missing_fc}")
    print("[NEXT] You can now run check_alignment.py with this cohort_table.")

if __name__ == "__main__":
    main()
