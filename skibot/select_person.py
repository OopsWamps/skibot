"""主轨迹选择：从 Sports2D 输出的多人物轨迹里锁定要分析的人。

判定条件（决策 6 的落地，2026-10-18 按真实视频教训修订）：
  - 尺寸：骨架包围盒主轴长度，画面里越大越可能是主角（雪场远景视频里
    背景滑雪者只有主角的几分之一大，是最稳的信号）。
  - 位移：髋部中心的累计路程。注意镜头跟摇会让背景人物位移虚高，只做辅因子。
  - 颜色：用户给 --color 时，按色相匹配打分，权重最高。

打分纯离线：select_person 只收轨迹与（可选的）已采样像素，不碰视频、
不打印。像素采样由调用方用 sample_pants_pixels 完成 —— 那一步需要真视频，
打分不需要，测试喂合成像素即可。

自动模式：0.70*尺寸 + 0.30*位移
颜色模式：0.65*颜色 + 0.25*尺寸 + 0.10*位移（颜色得分过低则回退自动，
由 color_fallback 标记，警告由调用方打印）
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .skeleton import L_ANKLE, L_HIP, L_KNEE, R_ANKLE, R_HIP, R_KNEE

MAX_SAMPLES = 30        # 每个人最多采多少帧的裤腿颜色
PATCH = 7               # 采样块边长（像素）
HUE_TOL_DEG = 18.0      # 色相容差
MIN_COLOR_SCORE = 0.12  # 颜色模式下，最佳匹配得分低于此值视为没匹配上

# 常用颜色名 → HSV 判定参数（hue 为 0~360 度；achromatic 用明度/饱和度区间）
NAMED_COLORS: dict[str, dict] = {
    "pink":    {"hue": 330, "sat": 0.25},
    "red":     {"hue": 0,   "sat": 0.35, "wrap": 15},
    "orange":  {"hue": 25,  "sat": 0.35},
    "yellow":  {"hue": 50,  "sat": 0.35},
    "green":   {"hue": 120, "sat": 0.20},
    "cyan":    {"hue": 185, "sat": 0.20},
    "blue":    {"hue": 220, "sat": 0.30},
    "purple":  {"hue": 280, "sat": 0.25},
    "black":   {"v_max": 90, "s_max": 0.45},
    "white":   {"v_min": 195, "s_max": 0.20},
    "gray":    {"s_max": 0.18, "v_min": 90, "v_max": 195},
    "grey":    {"s_max": 0.18, "v_min": 90, "v_max": 195},
}


@dataclass
class ColorSpec:
    name: str
    hue: float | None = None
    tol: float = HUE_TOL_DEG
    sat_min: float = 0.25
    v_min: float | None = None
    v_max: float | None = None
    s_max: float | None = None
    wrap: float = 0.0


@dataclass
class ScoreRow:
    """单个候选人物的得分行（cli 据此渲染打分表）。size_px/disp_px 为原始值。"""
    pid: int
    n_valid: int
    size_px: float
    disp_px: float
    size_n: float = 0.0
    disp_n: float = 0.0
    color: float | None = None
    score: float = 0.0


@dataclass
class Selection:
    """选人结果。color_fallback=True 表示颜色未匹配可信目标、已回退自动判定。"""
    pid: int
    reason: str
    rows: list[ScoreRow]
    color_fallback: bool = False
    color_best: float | None = None


def resolve_color(name_or_hex: str) -> ColorSpec:
    """'pink' 或 '#ff69b4' → 判定参数。"""
    s = name_or_hex.strip().lower()
    if s.startswith("#") and len(s) == 7:
        r, g, b = (int(s[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
        h, sv, v = colorsys.rgb_to_hsv(r, g, b)
        hue = h * 360.0
        if sv < 0.15:  # 低饱和十六进制按灰白黑处理
            return ColorSpec(name=s, v_min=90 if v > 0.35 else None,
                             v_max=90 if v <= 0.35 else None, s_max=0.20)
        return ColorSpec(name=s, hue=hue, sat_min=max(0.20, sv * 0.5))
    if s in NAMED_COLORS:
        c = NAMED_COLORS[s]
        return ColorSpec(name=s, hue=c.get("hue"), sat_min=c.get("sat", 0.25),
                         v_min=c.get("v_min"), v_max=c.get("v_max"),
                         s_max=c.get("s_max"), wrap=c.get("wrap", 0.0))
    raise ValueError(f"未知颜色: {name_or_hex}（可用 {', '.join(NAMED_COLORS)} 或 #rrggbb）")


def _match_mask(rgb: np.ndarray, spec: ColorSpec) -> np.ndarray:
    """rgb: (n,3) 0~255。返回逐像素是否匹配。"""
    r, g, b = rgb[:, 0] / 255.0, rgb[:, 1] / 255.0, rgb[:, 2] / 255.0
    mx, mn = np.maximum(np.maximum(r, g), b), np.minimum(np.minimum(r, g), b)
    v = mx
    s = np.where(mx > 1e-9, (mx - mn) / np.maximum(mx, 1e-9), 0.0)
    ok = np.ones(len(rgb), dtype=bool)
    if spec.hue is not None:
        ok &= s >= spec.sat_min
        h = np.zeros(len(rgb))
        d = np.maximum(mx - mn, 1e-9)
        m = (mx == r) & ok
        h[m] = (60 * ((g - b) / d) % 360)[m]
        m = (mx == g) & ok
        h[m] = (60 * ((b - r) / d) + 120)[m]
        m = (mx == b) & ok
        h[m] = (60 * ((r - g) / d) + 240)[m]
        dist = np.abs(h - spec.hue)
        if spec.wrap:  # 红色跨 0° 环绕
            dist = np.minimum(dist, 360 - dist)
        ok &= (dist <= spec.tol) | (360 - dist <= spec.tol)
    else:
        if spec.v_min is not None:
            ok &= v >= spec.v_min
        if spec.v_max is not None:
            ok &= v <= spec.v_max
        if spec.s_max is not None:
            ok &= s <= spec.s_max
    return ok


def sample_pants_pixels(video: str | Path, kp: np.ndarray) -> np.ndarray:
    """沿两条腿的髋-膝、膝-踝中点采 PATCH×PATCH 像素块。rgb: (n,3) 0~255。

    kp: (frames,26,3) [x,y,conf]。雪地/雾（高明度低饱和）会污染采样，
    对有色目标先滤掉；黑白灰目标交给 _match_mask 自行判定。
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"打不开视频: {video}")
    n_frames = kp.shape[0]
    frames_idx = [f for f in range(n_frames)
                  if kp[f][[L_HIP, L_KNEE, R_HIP, R_KNEE], 2].min() > 0]
    if not frames_idx:
        return np.zeros((0, 3))
    step = max(1, len(frames_idx) // MAX_SAMPLES)
    frames_idx = frames_idx[::step][:MAX_SAMPLES]

    pts_per_frame = [(L_HIP, L_KNEE, 0.5), (L_KNEE, L_ANKLE, 0.55),
                     (R_HIP, R_KNEE, 0.5), (R_KNEE, R_ANKLE, 0.55)]
    pixels = []
    half = PATCH // 2
    for f in frames_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for a, b, t in pts_per_frame:
            pa, pb = kp[f][a], kp[f][b]
            if pa[2] <= 0 or pb[2] <= 0:
                continue
            x = int((pa[0] + t * (pb[0] - pa[0])))
            y = int((pa[1] + t * (pb[1] - pa[1])))
            patch = rgb[max(0, y - half):y + half + 1, max(0, x - half):x + half + 1]
            if patch.size:
                med = np.median(patch.reshape(-1, 3), axis=0)
                pixels.append(med)
    cap.release()
    return np.array(pixels, dtype=float)


def color_score(rgb: np.ndarray, spec: ColorSpec) -> float:
    """匹配像素占比。无样本返回 0。"""
    if len(rgb) == 0:
        return 0.0
    if spec.hue is not None:  # 滤掉雪地/雾的高明度低饱和像素
        r, g, b = rgb[:, 0] / 255, rgb[:, 1] / 255, rgb[:, 2] / 255
        mx = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        s = np.where(mx > 1e-9, (mx - mn) / np.maximum(mx, 1e-9), 0)
        keep = ~((s < 0.15) & (mx > 0.7))
        rgb = rgb[keep]
        if len(rgb) == 0:
            return 0.0
    return float(_match_mask(rgb, spec).mean())


def size_score(kp: np.ndarray) -> float:
    """中位骨架包围盒主轴长度 px。仅统计关键点齐全的帧。"""
    lens = []
    for f in range(kp.shape[0]):
        pts = kp[f][kp[f][:, 2] > 0][:, :2]
        if len(pts) < 6:
            continue
        w = np.nanmax(pts[:, 0]) - np.nanmin(pts[:, 0])
        h = np.nanmax(pts[:, 1]) - np.nanmin(pts[:, 1])
        lens.append(max(w, h))
    return float(np.median(lens)) if lens else 0.0


def displacement_score(kp: np.ndarray) -> float:
    """髋部中心累计路程 px。"""
    hips = []
    for f in range(kp.shape[0]):
        l, r = kp[f][L_HIP], kp[f][R_HIP]
        if l[2] > 0 and r[2] > 0:
            hips.append((l[:2] + r[:2]) / 2)
    if len(hips) < 2:
        return 0.0
    d = np.diff(np.array(hips), axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def select_person(
    tracks: dict[int, np.ndarray],
    colors_by_pid: dict[int, np.ndarray] | None = None,
    color_name: str | None = None,
) -> Selection:
    """打分选人。colors_by_pid 由调用方用 sample_pants_pixels 预先采样。"""
    spec = resolve_color(color_name) if color_name else None
    if spec is not None and colors_by_pid is None:
        raise ValueError("给了 color_name 就必须给 colors_by_pid（先用 sample_pants_pixels 采样）")

    rows: list[ScoreRow] = []
    for pid, kp in tracks.items():
        row = ScoreRow(pid=pid,
                       n_valid=int((kp[:, :, 2] > 0).any(axis=1).sum()),
                       size_px=size_score(kp),
                       disp_px=displacement_score(kp))
        if spec is not None and colors_by_pid is not None:
            row.color = round(color_score(colors_by_pid[pid], spec), 3)
        rows.append(row)

    if not rows:
        raise ValueError("没有任何人物轨迹可选")

    size_max = max(r.size_px for r in rows) or 1.0
    disp_max = max(r.disp_px for r in rows) or 1.0
    for r in rows:
        r.size_n = round(r.size_px / size_max, 3)
        r.disp_n = round(r.disp_px / disp_max, 3)

    if spec is not None:
        for r in rows:
            r.score = round(0.65 * (r.color or 0) + 0.25 * r.size_n + 0.10 * r.disp_n, 3)
        best = max(rows, key=lambda r: r.score)
        if (best.color or 0) >= MIN_COLOR_SCORE:
            reason = (f"颜色 '{spec.name}' 匹配 {best.color:.0%}"
                      f"（尺寸 {best.size_n:.0%}，位移 {best.disp_n:.0%}）")
            return Selection(best.pid, reason, rows, color_best=best.color)
        # 颜色不可信，落到自动判定

    for r in rows:
        r.score = round(0.70 * r.size_n + 0.30 * r.disp_n, 3)
    best = max(rows, key=lambda r: r.score)
    reason = f"自动判定：尺寸 {best.size_n:.0%} + 位移 {best.disp_n:.0%}"
    return Selection(best.pid, reason, rows,
                     color_fallback=spec is not None,
                     color_best=best.color if spec is not None else None)
