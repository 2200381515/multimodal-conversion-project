# pipeline/check_leakage.py
from __future__ import annotations
import argparse
import pandas as pd


def read_ids(path: str, id_col: str) -> set[str]:
    df = pd.read_csv(path)
    if id_col not in df.columns:
        raise ValueError(f"missing id_col={id_col} in {path}")
    return set(df[id_col].astype(str).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--id_col", default="subject_id")
    args = ap.parse_args()

    tr = read_ids(args.train_csv, args.id_col)
    va = read_ids(args.val_csv, args.id_col)
    te = read_ids(args.test_csv, args.id_col)

    inter_tr_va = tr & va
    inter_tr_te = tr & te
    inter_va_te = va & te

    ok = (len(inter_tr_va) == 0) and (len(inter_tr_te) == 0) and (len(inter_va_te) == 0)
    if ok:
        print("[OK] No subject leakage.")
    else:
        print("[FAIL] Leakage detected:")
        if inter_tr_va:
            print(" train∩val:", list(sorted(inter_tr_va))[:20])
        if inter_tr_te:
            print(" train∩test:", list(sorted(inter_tr_te))[:20])
        if inter_va_te:
            print(" val∩test:", list(sorted(inter_va_te))[:20])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
