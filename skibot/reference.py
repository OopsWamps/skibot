"""三档水平参考区间：加载 reference_values.toml，查询与达标率计算。

决策依据 docs/DECISIONS.md 第 8 条。表内数值未校准前，报告必须带"参考值未校准"水印。
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LEVELS = ("novice", "intermediate", "advanced")


@dataclass
class MetricRef:
    lo: float
    hi: float
    verdict_low: str = ""
    verdict_high: str = ""
    verdict_ok: str = "在参考区间内"


def load(path: str | Path) -> dict[str, dict[str, MetricRef]]:
    """返回 {metric: {level: MetricRef}}。文件缺失或结构非法时抛异常。"""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    refs: dict[str, dict[str, MetricRef]] = {}
    for metric, levels in raw.items():
        refs[metric] = {}
        for level, table in levels.items():
            lo, hi = table["range"]
            refs[metric][level] = MetricRef(
                lo=lo,
                hi=hi,
                verdict_low=table.get("verdict_low", ""),
                verdict_high=table.get("verdict_high", ""),
                verdict_ok=table.get("verdict_ok", "在参考区间内"),
            )
    return refs


def lookup(refs: dict[str, dict[str, MetricRef]], metric: str, level: str) -> MetricRef:
    """查询某指标某档区间。缺档时回退到 novice（最宽区间，保守）。"""
    if metric not in refs:
        raise KeyError(f"参考区间表中没有指标: {metric}")
    by_level = refs[metric]
    if level not in by_level:
        if level not in LEVELS:
            raise ValueError(f"未知水平档: {level}，可选 {LEVELS}")
        return by_level["novice"]
    return by_level[level]


def pct_in_range(series: np.ndarray, ref: MetricRef) -> float:
    """有效帧中落在区间内的比例（0~1）。全无效返回 NaN。"""
    valid = series[~np.isnan(series)]
    if valid.size == 0:
        return np.nan
    in_range = (valid >= ref.lo) & (valid <= ref.hi)
    return float(in_range.mean())
