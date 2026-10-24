"""三维骨架契约与二维→三维提升适配器接口。

完整坐标约定与指标定义见 docs/adr/0001-3d-metric-definitions.md §3；这里写
实现层必须满足的约束：

- joints (T, 26, 3)：Halpe-26，root-relative。构造时对髋点齐全的帧逐帧减去
  左右髋中点，保证原点在骨盆。
- 坐标：+x 解剖右、+y 重力反方向（上）、+z 解剖前，右手系。长度单位任意，
  角度指标尺度不变；角度输出为度。
- confidence (T, 26)：[0, 1]。0 = 缺失；>= CONF_MIN_3D 才可用于指标。
- gravity_up (3,) 或 (T,3)：单位向量，契约默认 (0, 1, 0)。适配器若只能
  输出相机坐标，必须给出逐帧重力向量；指标逐帧取这个向量，不把 +y 写死。
  构造后统一存成 (T,3)。
- 缺帧：关节不可得时坐标 NaN、confidence 0；整帧不可得时该帧全部如此。
  全零帧按缺帧处理（valid_mask 为 False），不当作"所有关节缩到原点"。
- source：数据来源标签（"mock"/"motionbert"/...），进报告的数据来源说明。
- gravity_source：重力来源标签，GRAVITY_SOURCES 之一或 "unknown"，构造时
  校验（ADR-0001 §5.1 的强制标注；unknown 时躯干前倾不出数）。

提升适配器是纯接缝：输入二维 Halpe-26 关键点，输出 Skeleton3D。换模型只换
适配器，评估流程与指标代码不动。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .skeleton import CORE_JOINTS, L_HIP, R_HIP

CONF_MIN_3D = 0.3          # 三维关节进入指标的最低置信度（暂定，issue 04 定稿）
MIN_CORE_JOINTS = 4        # 一帧至少要有几个核心关节可用
DEFAULT_GRAVITY_UP = (0.0, 1.0, 0.0)
# 重力来源（ADR-0001 §5.1）：报告必须标注；unknown 时躯干前倾不出数。
GRAVITY_SOURCES = ("calibrated", "adapter", "image_vertical")


@dataclass
class Skeleton3D:
    """一帧 26 关节的三维骨架序列 + 逐关节置信度。

    构造后 joints 已居中对齐到髋中点（髋点齐全的帧）。缺测关节保持 NaN。
    """

    joints: np.ndarray
    confidence: np.ndarray
    gravity_up: np.ndarray = field(
        default_factory=lambda: np.array(DEFAULT_GRAVITY_UP, dtype=float))
    source: str = "unknown"
    # 重力来源：calibrated / adapter / image_vertical 之一（GRAVITY_SOURCES），
    # unknown = 不可信，躯干前倾不出数。契约字段，构造时校验——此前住在
    # meta dict 的魔法键里，键名拼错只会静默降级。
    gravity_source: str = "unknown"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.joints = np.array(self.joints, dtype=float, copy=True)
        self.confidence = np.array(self.confidence, dtype=float, copy=True)
        self.gravity_up = np.array(self.gravity_up, dtype=float, copy=True)
        if self.gravity_source != "unknown" and self.gravity_source not in GRAVITY_SOURCES:
            raise ValueError(
                f"gravity_source 必须是 {GRAVITY_SOURCES} 之一或 'unknown'，"
                f"收到 {self.gravity_source!r}")

        if self.joints.ndim != 3 or self.joints.shape[1:] != (26, 3):
            raise ValueError(f"joints 形状必须是 (T, 26, 3)，收到 {self.joints.shape}")
        if self.confidence.shape != self.joints.shape[:2]:
            raise ValueError(
                f"confidence 形状必须是 {self.joints.shape[:2]}，收到 {self.confidence.shape}")
        if self.gravity_up.ndim == 1:
            if self.gravity_up.shape != (3,):
                raise ValueError(
                    f"gravity_up 必须是 (3,) 或 (T,3)，收到 {self.gravity_up.shape}")
            self.gravity_up = np.broadcast_to(
                self.gravity_up, (self.joints.shape[0], 3)).copy()
        elif self.gravity_up.ndim == 2:
            if self.gravity_up.shape != (self.joints.shape[0], 3):
                raise ValueError(
                    f"gravity_up 形状必须是 {(self.joints.shape[0], 3)}，"
                    f"收到 {self.gravity_up.shape}")
        else:
            raise ValueError(
                f"gravity_up 必须是 (3,) 或 (T,3)，收到 {self.gravity_up.shape}")
        if not np.isfinite(self.gravity_up).all():
            raise ValueError(f"gravity_up 不能含 NaN/Inf，收到 {self.gravity_up}")
        norms = np.linalg.norm(self.gravity_up, axis=1)
        if np.any(np.abs(norms - 1.0) > 1e-6):
            raise ValueError(f"gravity_up 每帧必须是单位向量，范数 {norms}")
        if not np.isfinite(self.confidence).all():
            raise ValueError("confidence 不能含 NaN/Inf；缺失记 0")
        if self.confidence.size and (
                self.confidence.min() < 0.0 or self.confidence.max() > 1.0):
            raise ValueError("confidence 必须在 [0, 1]")

        # 强制 root-relative。髋点不齐的帧没有可定义的原点，保持原样；
        # root_defined 会标出这些帧。
        mid = (self.joints[:, L_HIP] + self.joints[:, R_HIP]) / 2.0
        ok = np.isfinite(mid).all(axis=1)
        if ok.any():
            self.joints[ok] -= mid[ok, None, :]

    @property
    def n_frames(self) -> int:
        return int(self.joints.shape[0])

    @property
    def root_defined(self) -> np.ndarray:
        """逐帧是否定义了骨盆原点（左右髋都有限）。"""
        return (np.isfinite(self.joints[:, L_HIP]).all(axis=1)
                & np.isfinite(self.joints[:, R_HIP]).all(axis=1))

    @property
    def valid_mask(self) -> np.ndarray:
        """逐帧是否可用于指标/动画。

        条件：核心关节里至少 MIN_CORE_JOINTS 个坐标有限、置信度 >= CONF_MIN_3D，
        且不是全零帧。全零帧通常是适配器对缺失帧的填充，不是真实姿态。
        """
        finite = np.isfinite(self.joints).all(axis=2)
        # 只判“是否被适配器用全零填充”，不设绝对阈值：坐标单位任意缩放时
        # 全零帧仍是全零，真实骨架不会因为单位小而被判无效。
        nonzero = np.abs(np.nan_to_num(self.joints)).sum(axis=2) > 0.0
        good = finite & nonzero & (self.confidence >= CONF_MIN_3D)
        return good[:, CORE_JOINTS].sum(axis=1) >= MIN_CORE_JOINTS


class Lifter(Protocol):
    """二维关键点 → 三维骨架的适配器接口。

    输入 (T, 26, >=3) 的 [x, y, conf] 数组；输出 Skeleton3D。
    """

    name: str

    def lift(self, keypoints: np.ndarray) -> Skeleton3D: ...


class MockLifter:
    """契约测试用的假模型：二维骨架放到 z=0 平面，翻转图像 y，居中由契约做。

    这个适配器只证明流程与形状契约，不提供任何精度结论。真实模型（MotionBERT）
    在 issue 05 接入，Lifter 接口不变。
    """

    name = "mock"

    def lift(self, keypoints: np.ndarray) -> Skeleton3D:
        kp = np.asarray(keypoints, dtype=float)
        if kp.ndim != 3 or kp.shape[1] < 26 or kp.shape[2] < 3:
            raise ValueError(f"keypoints 形状必须是 (T, >=26, >=3)，收到 {kp.shape}")
        n = kp.shape[0]
        joints = np.full((n, 26, 3), np.nan)
        conf = np.clip(kp[:, :26, 2], 0.0, 1.0)
        present = conf > 0.0
        joints[:, :, 0] = kp[:, :26, 0]
        joints[:, :, 1] = -kp[:, :26, 1]   # 图像 y 向下 → 契约 y 向上
        joints[:, :, 2] = 0.0              # 假模型是平面骨架
        joints[~present] = np.nan
        # 平面骨架、图像竖直当重力：重力来源就必须如实标成 image_vertical，
        # 不能冒充 adapter 的规范化输出。
        return Skeleton3D(joints=joints, confidence=conf, source=self.name,
                          gravity_source="image_vertical",
                          meta={"adapter": type(self).__name__})
