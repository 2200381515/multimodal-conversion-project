# pipeline/dataset_demo.py
from __future__ import annotations
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from pipeline.dataset import PatientT0Dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort_csv", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    df = pd.read_csv(args.cohort_csv)
    print("[COHORT] N =", len(df))
    if "label_convert" in df.columns:
        print("[LABEL] value counts:\n", df["label_convert"].value_counts(dropna=False))
    if "modality_mask" in df.columns:
        print("[MASK] value counts:\n", df["modality_mask"].value_counts(dropna=False))
    if "qc_exclude_flag" in df.columns:
        print("[QC] excluded:", int(df["qc_exclude_flag"].sum()))
    if "eligible_flag" in df.columns:
        print("[ELIG] eligible:", int(df["eligible_flag"].sum()))

    ds = PatientT0Dataset(args.cohort_csv, require_eligible=True, drop_qc_excluded=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    batch = next(iter(dl))
    print("\n[BATCH KEYS]", list(batch.keys()))
    print("x_scale:", batch["x_scale"].shape)
    print("x_sc:", batch["x_sc"].shape)
    print("x_fc:", batch["x_fc"].shape)
    print("modality_mask:", batch["modality_mask"].shape, batch["modality_mask"][:2])
    print("quality_score:", batch["quality_score"].shape, batch["quality_score"][:2])
    print("y:", batch["y"].shape, batch["y"][:10])


if __name__ == "__main__":
    main()
