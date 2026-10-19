"""视角分类：侧面（矢状面）还是正面（额状面）。

决策依据 docs/DECISIONS.md 第 3 条：纯 2D 角度只在相机大致垂直于动作平面时可信，
侧面视频和正面视频要套不同的指标规则。

判定信号（对有效帧取中位数，抗单帧抖动）：
  1. 肩线水平度：正面时两肩在图像里接近水平连线。
  2. 鼻子偏移：正面时鼻子在两肩中点正上方；侧面时明显偏向一侧。
  3. 左右置信度不对称：侧面时被遮挡一侧的肢体置信度持续偏低。

阈值全部标记 TODO_VALIDATE，等第一批真实滑雪视频校准。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .skeleton import CONF_MIN, L_LIMB, NOSE, L_SHOULDER, R_LIMB, R_SHOULDER

# TODO_VALIDATE: 以下阈值均为经验占位值
SHOULDER_HORIZ_MAX_DEG = 25.0   # 肩线与水平线夹角小于此值 → 偏正面
NOSE_OFFSET_MAX_RATIO = 0.25    # 鼻子水平偏移/肩宽 小于此值 → 偏正面
CONF_ASYMMETRY_MIN = 0.25       # 左右肢体平均置信度差大于此值 → 偏侧面
MIN_VALID_FRAMES = 10           # 少于该数有效帧则判 ambiguous


@dataclass
class ViewResult:
    view: str          # 'sagittal' | 'frontal' | 'ambiguous'
    score: float       # >0 偏正面，<0 偏侧面，绝对值越大越确定
    n_valid_frames: int
    detail: dict


def _limb_conf_mean(kp: np.ndarray, idxs: list[int]) -> float:
    confs = [kp[i, 2] for i in idxs if kp[i, 2] > 0]
    return float(np.mean(confs)) if confs else 0.0


def classify_view(keypoints: np.ndarray) -> ViewResult:
    """keypoints: (frames, n_kp, >=3)，x/y/置信度。返回逐视频的视角判定。"""
    n_frames = keypoints.shape[0]
    frontal_votes, lateral_votes = [], []

    for f in range(n_frames):
        kp = keypoints[f]
        ls, rs = kp[L_SHOULDER], kp[R_SHOULDER]
        nose = kp[NOSE]
        if min(ls[2], rs[2], nose[2]) < CONF_MIN:
            continue
        shoulder_w = float(np.hypot(rs[0] - ls[0], rs[1] - ls[1]))
        if shoulder_w < 1e-6:
            continue

        # 信号 1：肩线水平度（0°=完全水平）
        shoulder_tilt = abs(np.degrees(np.arctan2(rs[1] - ls[1], rs[0] - ls[0])))
        shoulder_tilt = min(shoulder_tilt, 180.0 - shoulder_tilt)

        # 信号 2：鼻子相对肩中点的水平偏移（按肩宽归一）
        mid_x = (ls[0] + rs[0]) / 2.0
        nose_offset = abs(nose[0] - mid_x) / shoulder_w

        # 信号 3：左右肢体置信度不对称
        asym = abs(_limb_conf_mean(kp, L_LIMB) - _limb_conf_mean(kp, R_LIMB))

        votes = 0.0
        votes += 1.0 if shoulder_tilt < SHOULDER_HORIZ_MAX_DEG else -1.0
        votes += 1.0 if nose_offset < NOSE_OFFSET_MAX_RATIO else -1.0
        votes += -1.0 if asym > CONF_ASYMMETRY_MIN else 0.0
        (frontal_votes if votes > 0 else lateral_votes).append(votes)

    n_valid = len(frontal_votes) + len(lateral_votes)
    if n_valid < MIN_VALID_FRAMES:
        return ViewResult("ambiguous", 0.0, n_valid, {"reason": "有效帧不足"})

    frontal_ratio = len(frontal_votes) / n_valid
    score = (frontal_ratio - 0.5) * 2.0
    view = "frontal" if frontal_ratio > 0.6 else ("sagittal" if frontal_ratio < 0.4 else "ambiguous")
    return ViewResult(view, score, n_valid, {"frontal_ratio": frontal_ratio})
