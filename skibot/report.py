"""结构化报告生成（JSON）。

决策依据 docs/DECISIONS.md 第 9 条：报告是价值层，标注视频是信任层；不做评分。
每项指标给出：区间、达标率、一句话判断。参考值未校准期间强制带水印字段。
指标清单与视角门控消费 metrics.METRICS 注册表。
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import numpy as np

from .metrics import METRICS
from .reference import LEVELS, MetricRef, lookup, pct_in_range

UNCALIBRATED_NOTICE = "参考区间为未校准的占位初值，本报告仅供自查，不构成教学结论。"


def _verdict(series: np.ndarray, ref: MetricRef) -> str:
    valid = series[~np.isnan(series)]
    if valid.size == 0:
        return "有效数据不足，无法判断"
    median = float(np.median(valid))
    if median < ref.lo:
        return ref.verdict_low or "低于参考区间"
    if median > ref.hi:
        return ref.verdict_high or "高于参考区间"
    return ref.verdict_ok


def build_report(
    video: str,
    view_result,
    series: dict[str, np.ndarray],
    refs: dict[str, dict[str, MetricRef]],
    level: str,
    person: dict | None = None,
    calibrated: bool = False,
) -> dict:
    if level not in LEVELS:
        raise ValueError(f"未知水平档: {level}，可选 {LEVELS}")

    metrics_out = []
    for name, (cname, need_view) in METRICS.items():
        if view_result.view != need_view:
            continue
        s = series[name]
        ref = lookup(refs, name, level)
        p = pct_in_range(s, ref)
        valid = s[~np.isnan(s)]
        metrics_out.append(
            {
                "metric": name,
                "name": cname,
                "reference": [ref.lo, ref.hi],
                "median_deg": None if valid.size == 0 else round(float(np.median(valid)), 1),
                "pct_in_range": None if np.isnan(p) else round(p, 3),
                "n_valid_frames": int(valid.size),
                "verdict": _verdict(s, ref),
            }
        )

    report = {
        "schema": "skibot.report/0.1",
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "video": str(video),
        "level": level,
        "view": {"detected": view_result.view, "score": round(view_result.score, 2),
                 "n_valid_frames": view_result.n_valid_frames, **view_result.detail},
        "assumptions": [
            "纯 2D 图像平面角度，假设相机大致垂直于动作平面",
            "躯干前倾角以图像竖直方向为零基准，未做相机倾斜校正",
            "角度约定与 Sports2D 一致（见 reference_values.toml 头部）",
        ],
        "calibrated": calibrated,
        "notice": None if calibrated else UNCALIBRATED_NOTICE,
        "metrics": metrics_out,
    }
    if person is not None:
        report["person"] = person
    return report


def save_json(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def to_html(report: dict) -> str:
    raise NotImplementedError("HTML 报告随 Web 壳一起实现（二期，见 docs/DECISIONS.md）")
