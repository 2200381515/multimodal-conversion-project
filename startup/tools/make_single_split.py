from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

in_csv = Path(r"F:\multimodal-conversion-project\results\week2\day1\cohort_filtered.csv")
out_csv = Path(r"F:\multimodal-conversion-project\results\week2\day1\single_split.csv")

df = pd.read_csv(in_csv).copy()
df["subject_id"] = df["subject_id"].astype(str)
df["label_convert"] = df["label_convert"].astype(int)

# 先切 test: 20%
trainval_ids, test_ids = train_test_split(
    df["subject_id"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label_convert"]
)

split_col = []
for sid in df["subject_id"]:
    if sid in set(test_ids):
        split_col.append("test")
    else:
        split_col.append("trainval")

df["split"] = split_col
df.to_csv(out_csv, index=False, encoding="utf-8-sig")

print("wrote:", out_csv)
print(df["split"].value_counts())
print()
print("test label counts:")
print(df.loc[df["split"] == "test", "label_convert"].value_counts())
print()
print("trainval label counts:")
print(df.loc[df["split"] == "trainval", "label_convert"].value_counts())