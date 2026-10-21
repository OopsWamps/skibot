"""评估流程：关键点 → 视角分类 → 五指标 → 参考区间 → 报告 dict。

主路径（cli 选完人后）与抢救路径（scripts/report_from_trc.py）共用同一条
流程 —— DECISIONS 10 的量角对账走抢救路径，行为必须与主路径一致。
二期弯相位划分（DECISIONS 42）在此插入。
"""

from __future__ import annotations

import numpy as np

from . import metrics as M
from .report import build_report
from .view import classify_view


def assess(keypoints: np.ndarray, *, level: str,
           refs: dict, label: str,
           person: dict | None = None, calibrated: bool = False,
           quality: dict | None = None) -> dict:
    """ keypoints: (frames, n_kp, >=3)，skeleton 契约。返回报告 dict（不落盘）。"""
    vr = classify_view(keypoints)
    series = M.compute_series(keypoints)
    return build_report(label, vr, series, refs, level,
                        person=person, calibrated=calibrated, quality=quality)
