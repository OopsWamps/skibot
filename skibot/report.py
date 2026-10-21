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

# 质量提示阈值（DECISIONS 5 的两个缓解项）：深蹲相位 2D 压缩放大膝弯误差；
# 人物过小（关键点 bbox < 画面高 20%）时像素噪声富集，EPFL 验证误差 1.8 倍。
DEEP_SQUAT_KNEE_DEG = 90.0
DEEP_SQUAT_RATIO_WARN = 0.2
SMALL_SUBJECT_RATIO = 0.2
LOW_CONF_FRAME_RATIO_WARN = 0.3


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


def _band_percentile(median: float | None, ref: MetricRef) -> float | None:
    """中位在参考带内的百分位位置：0=带下沿，100=带上沿，带外 <0 / >100。

    教练评审（docs/coach_review_simulated.md §4）：落带率受相位混合与帧间
    自相关影响难解读，并列中位数+百分位让用户知道自己在带里的位置。
    """
    if median is None or ref.hi <= ref.lo:
        return None
    return round(100.0 * (median - ref.lo) / (ref.hi - ref.lo), 1)


def build_report(
    video: str,
    view_result,
    series: dict[str, np.ndarray],
    refs: dict[str, dict[str, MetricRef]],
    level: str,
    person: dict | None = None,
    calibrated: bool = False,
    quality: dict | None = None,
) -> dict:
    if level not in LEVELS:
        raise ValueError(f"未知水平档: {level}，可选 {LEVELS}")

    metrics_out = []
    knee = series.get("knee_flexion")
    squat = np.nan if knee is None else float(np.mean(knee[~np.isnan(knee)] < DEEP_SQUAT_KNEE_DEG))
    quality = dict(quality) if quality else {}
    if not np.isnan(squat):
        quality["deep_squat_frame_ratio"] = round(squat, 3)
    notices = [] if calibrated else [UNCALIBRATED_NOTICE]

    for name, (cname, need_view) in METRICS.items():
        if view_result.view != need_view:
            continue
        s = series[name]
        ref = lookup(refs, name, level)
        p = pct_in_range(s, ref)
        valid = s[~np.isnan(s)]
        item = {
            "metric": name,
            "name": cname,
            "reference": [ref.lo, ref.hi],
            "median_deg": None if valid.size == 0 else round(float(np.median(valid)), 1),
            "pct_in_range": None if np.isnan(p) else round(p, 3),
            "band_percentile": _band_percentile(
                None if valid.size == 0 else float(np.median(valid)), ref),
            "n_valid_frames": int(valid.size),
            "verdict": _verdict(s, ref),
        }
        if name == "knee_flexion" and not np.isnan(squat) and squat >= DEEP_SQUAT_RATIO_WARN:
            item["caveat"] = (f"深蹲相位帧占 {squat*100:.0f}%，2D 投影在此相位放大膝弯误差，"
                              "数值仅供参考")
        metrics_out.append(item)

    report = {
        "schema": "skibot.report/0.1",
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "video": str(video),
        "level": level,
        "view": {"detected": view_result.view, "score": round(view_result.score, 2),
                 "n_valid_frames": view_result.n_valid_frames, **view_result.detail},
        "assumptions": [
            "纯 2D 图像平面角度，假设相机大致垂直于动作平面",
            "躯干前倾角以图像竖直方向为零基准，未做相机倾斜校正；"
            "坡面坡度与相机不水平会以表观前倾进入读数",
            "角度约定与 Sports2D 一致（见 reference_values.toml 头部）",
        ],
        "calibrated": calibrated,
        "notice": " ".join(notices) if notices else None,
        "metrics": metrics_out,
    }
    size_ratio = quality.get("person_height_ratio")
    if size_ratio is not None and size_ratio < SMALL_SUBJECT_RATIO:
        notices.append(f"人物在画面中偏小（约为画面高的 {size_ratio*100:.0f}%），"
                       "关键点误差增大，数值仅供参考")
    low_conf = quality.get("low_conf_frame_ratio")
    if low_conf is not None and low_conf >= LOW_CONF_FRAME_RATIO_WARN:
        notices.append(f"低置信度帧占 {low_conf*100:.0f}%，骨架质量受限，数值仅供参考")
    if quality:
        report["quality"] = quality
    if notices:  # 质量提示在 report 构造后追加，回写 notice
        report["notice"] = " ".join(notices)
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
