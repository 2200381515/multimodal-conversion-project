from pathlib import Path
import pandas as pd

root = Path(r"F:\2025.12.13交大新项目\share_to_ZYF_v2\derived_per_subject")

n_total = 0
n_fixed = 0
n_skipped = 0

for scale_path in root.glob("*/scale.csv"):
    n_total += 1
    try:
        df = pd.read_csv(scale_path)

        old_cols = list(df.columns)
        new_cols = [c.strip() for c in old_cols]

        changed = False

        # 先做 strip 后统一改名
        renamed_cols = []
        for c in new_cols:
            if c == "DATASET_DIAG2":
                renamed_cols.append("DATASET-DIAG2")
                changed = True
            else:
                renamed_cols.append(c)

        if renamed_cols != old_cols:
            df.columns = renamed_cols
            df.to_csv(scale_path, index=False, encoding="utf-8-sig")
            n_fixed += 1
        else:
            n_skipped += 1

    except Exception as e:
        print(f"[ERROR] {scale_path}: {e}")

print(f"total = {n_total}")
print(f"fixed = {n_fixed}")
print(f"skipped = {n_skipped}")