from pathlib import Path
import pandas as pd

in_csv = Path(r"F:\multimodal-conversion-project\startup\cohort_table\cohort_table_min10.csv")
df = pd.read_csv(in_csv)

print("n_rows =", len(df))
print()

for i, row in df.head(10).iterrows():
    scale_path = row["scale_path"]
    print("=" * 100)
    print("subject_id:", row.get("subject_id", "N/A"))
    print("scale_path :", scale_path)

    s = pd.read_csv(scale_path)
    print("columns:", list(s.columns))

    col = None
    if "DATASET-DIAG2" in s.columns:
        col = "DATASET-DIAG2"
    elif "DATASET_DIAG2" in s.columns:
        col = "DATASET_DIAG2"

    print("label_col:", col)

    if col is None:
        print("[ERROR] label column not found")
        continue

    print("shape:", s.shape)
    print("raw value at row 0:", repr(s.loc[0, col]))
    print("entire first row:")
    print(s.iloc[0].to_dict())
    print()