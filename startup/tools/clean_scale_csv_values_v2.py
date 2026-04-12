from pathlib import Path
import pandas as pd
import re

root = Path(r"F:\2025.12.13交大新项目\share_to_ZYF_v2\derived_per_subject")

def parse_cell(x):
    if pd.isna(x):
        return x

    s = str(x).strip()

    # ['ADNI'] -> ADNI
    m = re.fullmatch(r"\['(.+)'\]", s)
    if m:
        return m.group(1)

    # [[0.]], [[1]], [[57.92]] -> 数字
    m = re.fullmatch(r"\[\[\s*([-+]?\d+(?:\.\d*)?)\s*\]\]", s)
    if m:
        num = float(m.group(1))
        if num.is_integer():
            return int(num)
        return num

    # 兜底：如果本身就是普通数字字符串
    try:
        num = float(s)
        if num.is_integer():
            return int(num)
        return num
    except:
        return s

n_total = 0
n_fixed = 0

for scale_path in root.glob("*/scale.csv"):
    n_total += 1
    df = pd.read_csv(scale_path)

    df2 = df.copy()
    for col in df2.columns:
        df2[col] = df2[col].map(parse_cell)

    # 顺手统一列名
    df2.columns = [c.strip().replace("DATASET_DIAG2", "DATASET-DIAG2") for c in df2.columns]

    df2.to_csv(scale_path, index=False, encoding="utf-8-sig")
    n_fixed += 1

print(f"total = {n_total}")
print(f"fixed = {n_fixed}")
print("done.")