"""自绘标注视频：只画选中者，逐帧全量写出，叠加本项目五指标。

替代 Sports2D 的标注视频（实测两大问题：5 人全画、输出只剩 31/427 帧）。
骨架画在打码后的画面上（帧号与 trc 一致）；指标数值按参考区间绿/红着色。
序列计算与区间查询在模块内完成，调用方只给骨架、参考表与水平档。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import metrics as M
from .reference import MetricRef, lookup
from .skeleton import (
    L_ANKLE, L_HIP, L_KNEE, L_SHOULDER,
    R_ANKLE, R_HIP, R_KNEE, R_SHOULDER,
)

EDGES = [(L_SHOULDER, R_SHOULDER), (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),
         (L_HIP, R_HIP), (L_HIP, L_KNEE), (L_KNEE, L_ANKLE),
         (R_HIP, R_KNEE), (R_KNEE, R_ANKLE)]
JOINTS = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]

GREEN, RED, WHITE = (80, 220, 80), (60, 60, 230), (240, 240, 240)


def render_annotated(
    video_src: str | Path,
    kp: np.ndarray,                       # (frames,26,3) [x,y,conf]
    refs: dict[str, dict[str, MetricRef]],  # 全量参考表
    level: str,                           # 水平档，决定区间着色
    view: str,                            # 'sagittal' | 'frontal'，决定叠加哪套指标
    out_path: str | Path,
) -> Path:
    cap = cv2.VideoCapture(str(video_src))
    if not cap.isOpened():
        raise FileNotFoundError(f"打不开视频: {video_src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    series = M.compute_series(kp)
    active = [m for m, (_, need) in M.METRICS.items() if need == view]
    ref_by_metric = {m: lookup(refs, m, level) for m in active}

    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if f < kp.shape[0]:
            _draw_skeleton(frame, kp[f])
            _draw_metrics(frame, f, active, series, ref_by_metric)
        writer.write(frame)
        f += 1

    cap.release()
    writer.release()
    return out_path


def _draw_skeleton(frame, kp_f) -> None:
    for a, b in EDGES:
        if kp_f[a][2] > 0 and kp_f[b][2] > 0:
            cv2.line(frame, tuple(kp_f[a][:2].astype(int)), tuple(kp_f[b][:2].astype(int)),
                     GREEN, 3, cv2.LINE_AA)
    for j in JOINTS:
        if kp_f[j][2] > 0:
            cv2.circle(frame, tuple(kp_f[j][:2].astype(int)), 5, WHITE, -1, cv2.LINE_AA)


def _draw_metrics(frame, f, active, series, refs) -> None:
    y = 28
    cv2.putText(frame, "skibot", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2, cv2.LINE_AA)
    y += 30
    for m in active:
        s = series[m]
        ref = refs[m]
        v = s[f] if f < len(s) else np.nan
        label = m  # cv2 Hershey 字体不支持中文，叠加层用指标键名；中文名见 JSON 报告
        if np.isnan(v):
            line, color = f"{label}: -", WHITE
        else:
            in_range = ref.lo <= v <= ref.hi
            color = GREEN if in_range else RED
            line = f"{label}: {v:.0f} deg (ref {ref.lo:.0f}-{ref.hi:.0f})"
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        y += 26
