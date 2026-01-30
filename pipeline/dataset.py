# pipeline/dataset.py
from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_matrix(path: str) -> np.ndarray:
    if path is None or (isinstance(path, float) and np.isnan(path)):
        raise FileNotFoundError("empty path")
    path = str(path)
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".npz"):
        return np.load(path)["arr_0"]
    if path.endswith(".csv"):
        return np.loadtxt(path, delimiter=",")
    raise ValueError(f"unsupported matrix file: {path}")


def load_scale(path: str) -> np.ndarray:
    """
    约定：scale_path 可以是
      - 单行 csv（带header）：读取为一维特征
      - json：{"f1":..., "f2":...} 读取value按key排序（建议你们后续固定顺序）
      - npy：一维向量
    """
    path = str(path)
    if path.endswith(".npy"):
        x = np.load(path)
        return x.astype(np.float32).reshape(-1)
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        if len(df) != 1:
            raise ValueError(f"scale csv must be single-row: {path}")
        return df.iloc[0].to_numpy(dtype=np.float32)
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        keys = sorted(obj.keys())
        return np.array([obj[k] for k in keys], dtype=np.float32)
    raise ValueError(f"unsupported scale file: {path}")


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
        scale_norm_stats: dict | None = None  # {"mean":..., "std":...} for scale
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

        # 探测维度（用于缺失占位）
        self._N = None
        for p in df.get("sc_path", []):
            if isinstance(p, str) and len(p.strip()) > 0 and os.path.exists(p):
                a = load_matrix(p)
                self._N = a.shape[0]
                break

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

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # modality_mask: "111" -> [1,1,1]
        mask_str = str(row.get("modality_mask", "111"))
        if len(mask_str) != 3:
            mask_str = "111"
        mask = np.array([int(mask_str[0]), int(mask_str[1]), int(mask_str[2])], dtype=np.float32)

        # scale
        x_scale = None
        if mask[0] == 1 and isinstance(row.get("scale_path", None), str) and len(row["scale_path"].strip()) > 0:
            try:
                x_scale = load_scale(row["scale_path"])
                x_scale = self._norm_scale(x_scale)
            except Exception:
                x_scale = None
        if x_scale is None:
            if not self.fill_missing:
                raise ValueError(f"missing scale for {row.get('subject_id')}")
            x_scale = np.zeros((1,), dtype=np.float32)  # 最小占位；你后续可改成固定M维
            mask[0] = 0

        # sc/fc
        def _load_or_zero(path_key: str, norm_kind: str):
            p = row.get(path_key, None)
            if isinstance(p, str) and len(p.strip()) > 0 and os.path.exists(p):
                a = load_matrix(p)
                a = self._norm_mat(a, norm_kind)
                return a.astype(np.float32), 1.0
            if not self.fill_missing:
                raise ValueError(f"missing {path_key} for {row.get('subject_id')}")
            N = self._N if self._N is not None else 1
            return np.zeros((N, N), dtype=np.float32), 0.0

        x_sc, m_sc = _load_or_zero("sc_path", self.sc_norm)
        x_fc, m_fc = _load_or_zero("fc_path", self.fc_norm)
        mask[1] = m_sc
        mask[2] = m_fc

        y = int(row["label_convert"])
        q = float(row.get("quality_score", 1.0))

        meta = {
            "subject_id": row.get("subject_id"),
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
