from pathlib import Path
import pandas as pd

in_csv = Path(r"F:\multimodal-conversion-project\startup\cohort_table\cohort_table_full_from_mat_v2.csv")
out_csv = Path(r"F:\multimodal-conversion-project\startup\cohort_table\cohort_table_full_from_mat_v2_with_label.csv")

df = pd.read_csv(in_csv)

labels = []
for _, row in df.iterrows():
    scale_path = row["scale_path"]
    s = pd.read_csv(scale_path)

    if "DATASET-DIAG2" in s.columns:
        v = s.loc[0, "DATASET-DIAG2"]
    elif "DATASET_DIAG2" in s.columns:
        v = s.loc[0, "DATASET_DIAG2"]
    else:
        v = None

    labels.append(v)

df["label_convert"] = pd.to_numeric(labels, errors="coerce")
df.to_csv(out_csv, index=False, encoding="utf-8-sig")

print("wrote:", out_csv)
print("label_convert value counts:")
print(df["label_convert"].value_counts(dropna=False))
print("unique values:", sorted(df["label_convert"].dropna().unique().tolist()))