# pipeline/dataset.py
from __future__ import annotations

import os
import json
import re
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# -------------------------
# IO helpers
# -------------------------
def load_matrix(path: str) -> np.ndarray:
    if path is None or (isinstance(path, float) and np.isnan(path)):
        raise FileNotFoundError("empty path")
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing file: {path}")

    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".npz"):
        return np.load(path)["arr_0"]
    if path.endswith(".csv"):
        return np.loadtxt(path, delimiter=",")
    raise ValueError(f"unsupported matrix file: {path}")


def _parse_first_number(x: Any) -> float:
    """
    Parse a value that may look like:
      0, 0.0, "0", "[[0.]]", "[1.]", "  2.3  "
    Return float. If no number found -> nan.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return float("nan")
    s = str(x).strip()
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not m:
        return float("nan")
    try:
        return float(m.group(0))
    except Exception:
        return float("nan")


def load_scale(
    path: str,
    drop_cols: Optional[set] = None,
    keep_cols: Optional[List[str]] = None,
) -> np.ndarray:
    """
    约定：scale_path 可以是
      - 单行 csv（带header）：读取为一维特征（本函数会做列名strip与数值解析）
      - json：{"f1":..., "f2":...} 读取value按key排序（建议你们后续固定顺序）
      - npy：一维向量

    参数：
      drop_cols: 需要丢掉的列（例如 {"DATASET-DIAG2"} 防止标签泄漏）
      keep_cols: 若提供，则仅按该列顺序取特征（用于保证所有样本一致的特征维度/顺序）
    """
    if path is None or (isinstance(path, float) and np.isnan(path)):
        raise FileNotFoundError("empty scale path")
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing file: {path}")

    drop_cols = drop_cols or set()

    if path.endswith(".npy"):
        x = np.load(path)
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        return x

    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        keys = sorted(obj.keys())
        vals = [obj[k] for k in keys]
        x = np.array([_parse_first_number(v) for v in vals], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x

    if path.endswith(".csv"):
        df = pd.read_csv(path)
        if len(df) != 1:
            raise ValueError(f"scale csv must be single-row: {path}")

        # strip column names (解决你遇到的 INSOMNA      / DATASET      这种尾部空格)
        df.columns = [c.strip() for c in df.columns]

        row0 = df.iloc[0]

        # 仅保留指定列顺序（保证每个样本特征对齐）
        if keep_cols is not None:
            # 缺失列填 NaN，后面统一填0
            for c in keep_cols:
                if c not in row0.index:
                    row0[c] = np.nan
            row0 = row0[keep_cols]

        # 丢掉指定列（例如标签列）
        if drop_cols:
            row0 = row0.drop(labels=[c for c in row0.index if c in drop_cols], errors="ignore")

        # 数值化（支持 '[[0.]]' 等字符串）
        vals = np.array([_parse_first_number(v) for v in row0.values], dtype=np.float32)
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        return vals

    raise ValueError(f"unsupported scale file: {path}")


# -------------------------
# Dataset
# -------------------------
class PatientT0Dataset(Dataset):
    """
    输出字段（对齐你文档第1周要求）：
      x_scale: [M]
      x_sc:    [N,N]
      x_fc:    [N,N]
      modality_mask: [3] (scale, sc, fc)
      quality_score: [1]
      y: scalar (0/1)
      meta: dict
    """

    def __init__(
        self,
        cohort_csv: str,
        require_eligible: bool = True,
        drop_qc_excluded: bool = True,
        fill_missing: bool = True,
        sc_norm: str = "zscore_global",   # 简化：全局zscore
        fc_norm: str = "zscore_global",
        scale_norm_stats: dict | None = None,  # {"mean":..., "std":...} for scale
        drop_scale_cols: Optional[List[str]] = None,  # 默认会丢掉标签列，避免泄漏
    ):
        df = pd.read_csv(cohort_csv)

        if require_eligible and "eligible_flag" in df.columns:
            df = df[df["eligible_flag"] == 1]
        if drop_qc_excluded and "qc_exclude_flag" in df.columns:
            df = df[df["qc_exclude_flag"] == 0]

        # 必须有 label_convert
        df = df[df["label_convert"].isin([0, 1])].reset_index(drop=True)

        self.df = df
        self.fill_missing = fill_missing
        self.sc_norm = sc_norm
        self.fc_norm = fc_norm
        self.scale_norm_stats = scale_norm_stats

        # scale: 默认丢掉标签列，防止泄漏（你用 DATASET-DIAG2 作为 y）
        if drop_scale_cols is None:
            drop_scale_cols = ["DATASET-DIAG2"]
        self.drop_scale_cols = set(drop_scale_cols)

        # 探测维度（用于缺失占位）
        self._N = None
        for p in df.get("sc_path", []):
            if isinstance(p, str) and len(p.strip()) > 0 and os.path.exists(p):
                a = load_matrix(p)
                if a.ndim == 2 and a.shape[0] == a.shape[1]:
                    self._N = a.shape[0]
                    break

        # 探测 scale 维度 + 固定特征列顺序（非常关键：保证所有样本 x_scale 对齐）
        self.scale_cols: Optional[List[str]] = None
        for p in df.get("scale_path", []):
            if isinstance(p, str) and len(p.strip()) > 0 and os.path.exists(p):
                try:
                    sdf = pd.read_csv(p)
                    if len(sdf) == 1:
                        sdf.columns = [c.strip() for c in sdf.columns]
                        cols = [c for c in sdf.columns if c not in self.drop_scale_cols]
                        self.scale_cols = cols
                        break
                except Exception:
                    continue

        # 如果完全探测不到 scale_cols，又允许缺失，那就退化为 1 维占位
        #（不过你现在 scale 都齐全，通常不会走到这）
        if self.scale_cols is None:
            self.scale_cols = None

    def __len__(self):
        return len(self.df)

    def _norm_mat(self, a: np.ndarray, kind: str) -> np.ndarray:
        a = a.astype(np.float32)
        if kind == "zscore_global":
            mu = float(a.mean())
            sd = float(a.std() + 1e-6)
            return (a - mu) / sd
        return a

    def _norm_scale(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        if self.scale_norm_stats is None:
            return x
        mu = self.scale_norm_stats["mean"]
        sd = self.scale_norm_stats["std"]
        return (x - mu) / (sd + 1e-6)

    @staticmethod
    def _path_ok(p: Any) -> bool:
        return isinstance(p, str) and len(p.strip()) > 0 and os.path.exists(p)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        sid = row.get("subject_id")

        # --- SCALE (mask[0]) ---
        scale_path = row.get("scale_path", None)
        x_scale = None
        has_scale = self._path_ok(scale_path)

        if has_scale:
            try:
                x_scale = load_scale(
                    scale_path,
                    drop_cols=self.drop_scale_cols,
                    keep_cols=self.scale_cols,  # 固定列顺序
                )
                x_scale = self._norm_scale(x_scale)
            except Exception:
                x_scale = None
                has_scale = False

        if x_scale is None:
            if not self.fill_missing:
                raise ValueError(f"missing/failed scale for {sid}")
            # 如果我们探测到了 scale_cols，就用对应维度占位，否则退化为1维
            M = len(self.scale_cols) if self.scale_cols is not None else 1
            x_scale = np.zeros((M,), dtype=np.float32)

        # --- SC / FC (mask[1], mask[2]) ---
        def _load_or_zero(path_key: str, norm_kind: str):
            p = row.get(path_key, None)
            if self._path_ok(p):
                try:
                    a = load_matrix(p)
                    a = self._norm_mat(a, norm_kind)
                    return a.astype(np.float32), True
                except Exception:
                    pass
            if not self.fill_missing:
                raise ValueError(f"missing/failed {path_key} for {sid}")
            N = self._N if self._N is not None else 1
            return np.zeros((N, N), dtype=np.float32), False

        x_sc, has_sc = _load_or_zero("sc_path", self.sc_norm)
        x_fc, has_fc = _load_or_zero("fc_path", self.fc_norm)

        # ✅ modality_mask 由“是否成功读取”决定，而不是信 cohort_table 里的字符串
        mask = np.array([1.0 if has_scale else 0.0, 1.0 if has_sc else 0.0, 1.0 if has_fc else 0.0], dtype=np.float32)

        y = int(row["label_convert"])
        q = float(row.get("quality_score", 1.0))

        meta = {
            "subject_id": sid,
            "delta_t_years": row.get("delta_t_years", None),
            "t0_date": row.get("t0_date", None),
        }

        return {
            "x_scale": torch.from_numpy(x_scale),
            "x_sc": torch.from_numpy(x_sc),
            "x_fc": torch.from_numpy(x_fc),
            "modality_mask": torch.from_numpy(mask),
            "quality_score": torch.tensor([q], dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.long),
            "meta": meta,
        }
