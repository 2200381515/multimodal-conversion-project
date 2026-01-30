from __future__ import annotations

import json
import numpy as np
import pandas as pd


def load_scale(path: str) -> np.ndarray:
    """
    约定：
      - npy: 1D
      - csv: 单行（带header）
      - json: {k:v}，按 key 排序取 value
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

    raise ValueError(f"Unsupported scale file: {path}")


def build_scale_matrix(df: pd.DataFrame, scale_col: str = "scale_path") -> np.ndarray:
    xs = []
    for _, row in df.iterrows():
        xs.append(load_scale(row[scale_col]))
    # 需要所有样本维度一致
    return np.stack(xs, axis=0)


def load_matrix(path: str) -> np.ndarray:
    """
    约定：
      - npy: 2D
      - npz: arr_0 或显式键
      - csv: 2D
    """
    path = str(path)
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32)

    if path.endswith(".npz"):
        obj = np.load(path)
        # 常见：arr_0
        if "arr_0" in obj:
            return obj["arr_0"].astype(np.float32)
        # 兜底：取第一个 key
        keys = list(obj.keys())
        if not keys:
            raise ValueError(f"empty npz: {path}")
        return obj[keys[0]].astype(np.float32)

    if path.endswith(".csv"):
        return np.loadtxt(path, delimiter=",").astype(np.float32)

    raise ValueError(f"Unsupported matrix file: {path}")


def mat_to_vec_upper(mat: np.ndarray, drop_diag: bool = True) -> np.ndarray:
    """
    将对称矩阵向量化为上三角（默认去对角），减少冗余。
    """
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"matrix must be square 2D, got shape={mat.shape}")
    n = mat.shape[0]
    k = 1 if drop_diag else 0
    iu = np.triu_indices(n, k=k)
    return mat[iu].astype(np.float32)


def build_matrix_features(
    df: pd.DataFrame,
    path_col: str,
    vectorize: str = "upper",
) -> np.ndarray:
    """
    从 df[path_col] 读取矩阵，并向量化堆叠成 [N, D]
    vectorize:
      - "upper": 上三角（去对角）
      - "flatten": 全矩阵 flatten
    """
    vectorize = vectorize.lower().strip()
    xs = []
    for _, row in df.iterrows():
        mat = load_matrix(row[path_col])
        if vectorize == "upper":
            x = mat_to_vec_upper(mat, drop_diag=True)
        elif vectorize == "flatten":
            x = mat.reshape(-1).astype(np.float32)
        else:
            raise ValueError(f"unknown vectorize={vectorize}, use upper/flatten")
        xs.append(x)

    return np.stack(xs, axis=0)


def build_sc_matrix(df: pd.DataFrame, sc_col: str = "sc_path", vectorize: str = "upper") -> np.ndarray:
    return build_matrix_features(df, path_col=sc_col, vectorize=vectorize)


def build_fc_matrix(df: pd.DataFrame, fc_col: str = "fc_path", vectorize: str = "upper") -> np.ndarray:
    return build_matrix_features(df, path_col=fc_col, vectorize=vectorize)