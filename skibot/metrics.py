"""五项体态指标计算与指标注册表。

决策依据 docs/DECISIONS.md 第 7 条。角度约定与 Sports2D 对齐
（见 reference_values.toml 头部注释），关键点为 Halpe-26 索引（skibot.skeleton）。

METRICS 是唯一指标注册表（名字、中文名、适用视角），report 层按它渲染与
门控 —— 加指标 = 这里加一行 + reference_values.toml 加一节。

局限性（写进报告的假设）：
  - trunk_lean 以图像竖直方向为零基准，未做相机倾斜校正
    （Sports2D 有 floor_angle 估计，接入留待验证阶段）。
  - knee_valgus 是额状面 2D 近似：髋-膝-踝夹角偏离 180° 的量含真实的内扣，
    也含姿态投影误差，标注视频必须同步给出让用户自查。
"""

from __future__ import annotations

import numpy as np

from .skeleton import (
    CONF_MIN, L_ANKLE, L_HIP, L_KNEE, L_SHOULDER,
    R_ANKLE, R_HIP, R_KNEE, R_SHOULDER, SIDE_JOINTS,
)

# 唯一指标注册表：name → (中文名, 适用视角)
METRICS: dict[str, tuple[str, str]] = {
    "hip_flexion": ("屈髋角", "sagittal"),
    "knee_flexion": ("膝弯曲角", "sagittal"),
    "trunk_lean": ("躯干前倾角", "sagittal"),
    "knee_valgus": ("膝盖内扣角", "frontal"),
    "shoulder_hip_align": ("肩髋对齐", "frontal"),
}
SAGITTAL_METRICS = tuple(n for n, (_, plane) in METRICS.items() if plane == "sagittal")
FRONTAL_METRICS = tuple(n for n, (_, plane) in METRICS.items() if plane == "frontal")
ALL_METRICS = tuple(METRICS)


def angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """b 为顶点，a-b-c 三点夹角，0~180°。"""
    v1, v2 = a[:2] - b[:2], c[:2] - b[:2]
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-9:
        return np.nan
    cos = float(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def vertical_deviation(a: np.ndarray, b: np.ndarray) -> float:
    """节段 a→b 与图像竖直方向的偏角，0°=完全竖直。"""
    dx, dy = float(b[0] - a[0]), float(b[1] - a[1])
    if abs(dx) + abs(dy) < 1e-9:
        return np.nan
    return float(np.degrees(np.arctan2(abs(dx), abs(dy))))


def _ok(kp: np.ndarray, *idxs: int) -> bool:
    return all(kp[i, 2] >= CONF_MIN for i in idxs)


def select_side(kp: np.ndarray) -> str:
    """选置信度高的一侧肢体（侧面视频里近侧肢体更可信）。"""
    left = np.mean([kp[i, 2] for i in SIDE_JOINTS["left"]])
    right = np.mean([kp[i, 2] for i in SIDE_JOINTS["right"]])
    return "left" if left >= right else "right"


def compute_series(keypoints: np.ndarray) -> dict[str, np.ndarray]:
    """逐帧计算全部五项指标。无效帧为 NaN。keypoints: (frames, n_kp, >=3)。"""
    n = keypoints.shape[0]
    out = {m: np.full(n, np.nan) for m in ALL_METRICS}

    for f in range(n):
        kp = keypoints[f]
        side = select_side(kp)
        sh, hip, knee, ank = SIDE_JOINTS[side]

        if _ok(kp, sh, hip, knee):
            out["hip_flexion"][f] = angle_at(kp[sh], kp[hip], kp[knee])
        if _ok(kp, hip, knee, ank):
            out["knee_flexion"][f] = angle_at(kp[hip], kp[knee], kp[ank])

        if _ok(kp, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
            neck = (kp[L_SHOULDER][:2] + kp[R_SHOULDER][:2]) / 2.0
            pelvis = (kp[L_HIP][:2] + kp[R_HIP][:2]) / 2.0
            trunk = np.array([*neck, 1.0])
            out["trunk_lean"][f] = vertical_deviation(pelvis, trunk)

        if _ok(kp, L_HIP, L_KNEE, L_ANKLE) and _ok(kp, R_HIP, R_KNEE, R_ANKLE):
            left_valgus = angle_at(kp[L_HIP], kp[L_KNEE], kp[L_ANKLE])
            right_valgus = angle_at(kp[R_HIP], kp[R_KNEE], kp[R_ANKLE])
            out["knee_valgus"][f] = min(left_valgus, right_valgus)  # 取更扣的一侧

        if _ok(kp, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
            sh_line = kp[R_SHOULDER][:2] - kp[L_SHOULDER][:2]
            hip_line = kp[R_HIP][:2] - kp[L_HIP][:2]
            cross = abs(float(np.cross(sh_line, hip_line)))
            dot = float(np.dot(sh_line, hip_line))
            out["shoulder_hip_align"][f] = float(np.degrees(np.arctan2(cross, dot)))

    return out
