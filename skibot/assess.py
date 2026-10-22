"""评估流程：关键点 → 视角分类 → 五指标 → 参考区间 → 评估结果。

主路径（cli 选完人后）与抢救路径（scripts/report_from_trc.py）共用同一条
流程 —— 一期 #10 的量角对账走抢救路径，行为必须与主路径一致。
Assessment 是唯一出口：报告、角度序列、帧级视角归属一次算齐整体交付，
render 等下游收现成结果，不再重算。二期弯相位划分（docs/DECISIONS.md
二期待办）在此插入。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import metrics as M
from .skeleton import CORE_JOINTS, size_score
from .report import build_report
from .view import classify_view, classify_view_frames

LOW_CONF_CORE_RATIO = 0.75  # 核心八点有效占比低于此值的帧记为低置信度帧


@dataclass
class Assessment:
    """评估结果三件套：报告 dict、逐帧指标序列、帧级视角归属。

    下游（render、对账脚本）消费这个整体，不要自己再算 series/frame_view
    —— 报告与叠加层必须出自同一次计算。
    """
    report: dict
    series: dict[str, np.ndarray]
    frame_view: np.ndarray


def compute_quality(keypoints: np.ndarray, frame_height: float | None = None) -> dict:
    """质量统计（验证进展 #3 的报告侧缓解输入）：人物高度占比 + 低置信度帧占比。

    trc 的 conf 是 0/1 存在标记，判核心 8 点而非全部 26 点 —— 雪场远景
    缺脸/手脚趾很常见，全点口径会让提示常驻失去区分度。判读阈值与文案
    在 report 层，这里只出统计。
    """
    quality: dict = {}
    if frame_height:
        quality["person_height_ratio"] = round(size_score(keypoints) / frame_height, 3)
    core_ok = (keypoints[:, CORE_JOINTS, 2] > 0).mean(axis=1)
    quality["low_conf_frame_ratio"] = round(float((core_ok < LOW_CONF_CORE_RATIO).mean()), 3)
    return quality


def assess(keypoints: np.ndarray, *, level: str,
           refs: dict, label: str,
           person: dict | None = None, calibrated: bool = False,
           quality: dict | None = None, frame_height: float | None = None) -> Assessment:
    """keypoints: (frames, n_kp, >=3)，skeleton 契约。返回评估结果（不落盘）。

    quality：调用方追加的质量统计项（同名键优先于内部统计）。
    """
    frame_view = classify_view_frames(keypoints)
    vr = classify_view(keypoints, frame_view=frame_view)
    series = M.compute_series(keypoints, frame_view=frame_view)
    q = dict(quality) if quality else {}
    for k, v in compute_quality(keypoints, frame_height).items():
        q.setdefault(k, v)
    report = build_report(label, vr, series, refs, level,
                          person=person, calibrated=calibrated, quality=q)
    return Assessment(report=report, series=series, frame_view=frame_view)
