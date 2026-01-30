# pipeline/labeling.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd


@dataclass
class LabelingConfig:
    window_years: float = 5.0               # 5或6
    include_boundary: bool = True           # <= window 还是 < window
    phase1_drop_insufficient_followup: bool = True  # Phase-1：随访不足直接排除（更干净）


def _to_dt(x) -> Optional[pd.Timestamp]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, pd.Timestamp):
        return x
    try:
        return pd.to_datetime(x)
    except Exception:
        return None


def compute_label_and_time(
    t0_date,
    followup_end_date,
    conversion_date,
    cfg: LabelingConfig
) -> Tuple[Optional[int], Optional[float], Optional[int], Optional[str]]:
    """
    返回:
      label_convert: 0/1/None (None表示无法确定或不纳入Phase-1)
      delta_t_years: 转化或末次随访距离T0的年数
      eligible_flag: 1可用于Phase-1二分类; 0不可
      reason: 不可用原因
    """
    t0 = _to_dt(t0_date)
    fu = _to_dt(followup_end_date)
    conv = _to_dt(conversion_date)

    if t0 is None:
        return None, None, 0, "missing_t0_date"

    # 若发生转化：用 conversion_date 计算
    if conv is not None:
        dt_years = (conv - t0).days / 365.25
        # 若转化在T0之前，数据有问题
        if dt_years < 0:
            return None, None, 0, "conversion_before_t0"
        # 判定是否在窗内
        if cfg.include_boundary:
            in_window = dt_years <= cfg.window_years
        else:
            in_window = dt_years < cfg.window_years
        label = 1 if in_window else 0  # “窗外转化”按0处理也可，但通常Phase-1会重新定义队列；这里先给0
        return label, dt_years, 1, "ok_converted"

    # 未转化：用末次随访
    if fu is None:
        # 没末次随访，无法判断是否“未转化”
        return None, None, 0, "missing_followup_end_date"

    dt_years = (fu - t0).days / 365.25
    if dt_years < 0:
        return None, None, 0, "followup_before_t0"

    # 随访是否足够覆盖窗
    if cfg.include_boundary:
        sufficient = dt_years >= cfg.window_years
    else:
        sufficient = dt_years > cfg.window_years

    if not sufficient and cfg.phase1_drop_insufficient_followup:
        return None, dt_years, 0, "insufficient_followup"

    # 够窗或允许保留（不建议Phase-1这么做）
    return 0, dt_years, 1, "ok_nonconvert"
