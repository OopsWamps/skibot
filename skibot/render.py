"""自绘标注视频：只画选中者，逐帧全量写出，叠加本项目五指标。

替代 Sports2D 的标注视频（实测两大问题：5 人全画、输出只剩 31/427 帧）。
骨架画在打码后的画面上（帧号与 trc 一致）；指标数值按参考区间绿/红着色。
序列与帧级视角出自 assess 的评估结果（Assessment）—— 报告与叠加层出自
同一次计算，本模块不重算。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from . import metrics as M
from .assess import Assessment
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


def probe_video(video_src: str | Path) -> tuple[int, int, float]:
    """视频元数据探测：（宽, 高, 帧率）。打不开时抛 FileNotFoundError。"""
    cap = cv2.VideoCapture(str(video_src))
    if not cap.isOpened():
        raise FileNotFoundError(f"打不开视频: {video_src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h, fps


def render_annotated(
    video_src: str | Path,
    kp: np.ndarray,                       # (frames,26,3) [x,y,conf]
    assessment: Assessment,               # assess 的评估结果：序列与帧级视角
    refs: dict[str, dict[str, MetricRef]],  # 全量参考表
    level: str,                           # 水平档，决定区间着色
    out_path: str | Path,
    view: str | None = None,              # 覆盖叠加哪套指标；默认按评估判定
) -> Path:
    cap = cv2.VideoCapture(str(video_src))
    if not cap.isOpened():
        raise FileNotFoundError(f"打不开视频: {video_src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))  # type: ignore[attr-defined]

    series = assessment.series
    detected = view or assessment.report["view"]["detected"]
    if detected == "mixed":
        # 跟拍混合视角：逐帧门控已保证每个值出自其可信平面，全部指标可画
        active = list(M.METRICS)
    else:
        active = [m for m, (_, need) in M.METRICS.items() if need == detected]
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
    return _transcode_h264(out_path)


def _transcode_h264(out_path: Path) -> Path:
    """mp4v 浏览器 <video> 不支持，转 H.264 + yuv420p + faststart。

    转码失败（ffmpeg 缺失等）保留原文件并降级——本地播放器仍可播 mp4v。
    """
    import subprocess

    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        print("[skibot] imageio_ffmpeg 不可用，标注视频保持 mp4v（浏览器或无法播放）",
              file=sys.stderr)
        return out_path

    tmp = out_path.with_suffix(".h264.mp4")
    r = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(out_path),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         str(tmp)],
        check=False, capture_output=True)
    if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
        tmp.replace(out_path)
    else:
        tmp.unlink(missing_ok=True)
        print(f"[skibot] 标注视频 H.264 转码失败，保留 mp4v：{r.stderr[-200:]!r}",
              file=sys.stderr)
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
