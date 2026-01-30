# pipeline/qc.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd


@dataclass
class QCConfig:
    dvars_threshold: float = 50.0
    homogeneity_min: float = 0.0      # 你们拿到数据后再定（先留0表示不启用下限）
    enable_dvars: bool = True
    enable_homogeneity: bool = False  # 没阈值前先不硬剔除
    manual_exclude_col: str = "qc_manual_exclude"  # 1表示人工剔除


def _safe_float(x) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        return float(x)
    except Exception:
        return None


def qc_decide(row: pd.Series, cfg: QCConfig) -> Tuple[int, float, str]:
    """
    返回:
      qc_exclude_flag: 1剔除/0保留
      quality_score: [0,1] 简单质量分(可后续更复杂)
      qc_reason: 原因
    """
    # 人工剔除优先
    if cfg.manual_exclude_col in row and str(row[cfg.manual_exclude_col]) not in ["", "0", "0.0", "False", "false", "nan", "None"]:
        return 1, 0.0, "manual_exclude"

    dvars = _safe_float(row.get("dvars", None))
    homo = _safe_float(row.get("homogeneity", None))

    # 质量分（先给一个可用的朴素版本）
    quality = 1.0
    reasons = []

    if cfg.enable_dvars and dvars is not None:
        if dvars > cfg.dvars_threshold:
            return 1, 0.0, f"dvars>{cfg.dvars_threshold}"
        # 越接近阈值，质量越低（线性压缩）
        quality *= max(0.0, 1.0 - (dvars / (cfg.dvars_threshold * 2.0)))

    if cfg.enable_homogeneity and homo is not None:
        if homo < cfg.homogeneity_min:
            return 1, 0.0, f"homogeneity<{cfg.homogeneity_min}"
        # 越高越好（简单归一）
        quality *= min(1.0, max(0.0, homo))

    if dvars is None and homo is None:
        reasons.append("qc_missing")

    reason = "ok" if not reasons else ";".join(reasons)
    return 0, float(quality), reason
