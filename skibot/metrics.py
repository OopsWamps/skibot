"""体态指标计算与指标注册表。

二维五项是过渡期口径（一期 #7），三维六项是二期主口径（ADR-0001）。
角度约定见 reference_values.toml 头部注释，关键点为 Halpe-26（skibot.skeleton）。

注册表：
  METRICS         六项规范指标，带二维/三维基准标识（三维实现在 skibot.metrics3d）。
  LEGACY_METRICS  只存在于二维过渡期的遗留键：shoulder_hip_align。
  TWO_D_METRICS   二维管线实际产出的五个键（compute_series / report / render 用）。
  ALL_METRICS     六项规范指标（三维管线用）。

旧名兼容：knee_valgus 是 knee_abduction 的别名；reference.lookup 与报告读方
都按别名解析。二维报告 schema 0.1→0.2 的迁移见 docs/report_schema.md。

局限性（写进报告的假设）：
  - trunk_lean 以图像竖直方向为零基准，未做相机倾斜校正
    （Sports2D 有 floor_angle 估计，接入留待验证阶段）。
  - 二维 knee_abduction（旧 knee_valgus）是额状面投影：髋-膝-踝夹角偏离
    180° 的量含真实的内扣，也含姿态投影误差。三维 JCS 定义见 ADR-0001 §5.5，
    两者不可直接比较。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .skeleton import (
    CONF_MIN, L_ANKLE, L_HIP, L_KNEE, L_SHOULDER,
    R_ANKLE, R_HIP, R_KNEE, R_SHOULDER, SIDE_JOINTS, pick_side,
)


@dataclass(frozen=True)
class MetricDef:
    """一项指标的定义元数据。

    bases：哪些基准能算这项指标（"2d" / "3d"）。同一个键在不同基准下可能
    是不同定义（knee_abduction），definitions 写清各自口径。
    phase：是否参与相位聚合（ADR-0001 §2）。
    aliases：旧名；报告读方与参考表查询都按别名解析。
    replaced_by：遗留键被哪些规范指标取代。
    """
    name: str
    label: str
    plane: str
    bases: tuple[str, ...]
    phase: bool = False
    aliases: tuple[str, ...] = ()
    definitions: dict[str, str] = field(default_factory=dict)
    replaced_by: tuple[str, ...] = ()


# 六项规范指标（ADR-0001 §2）。knee_abduction 的 2D 基准是旧 knee_valgus
# 的投影口径，3D 基准是 JCS 外展；报告用 basis 字段区分。
METRICS: dict[str, MetricDef] = {
    "hip_flexion": MetricDef(
        "hip_flexion", "屈髋角", "sagittal", ("2d", "3d"), phase=True,
        definitions={
            "2d": "肩-髋-膝夹角，180°=伸直",
            "3d": "躯干轴-股骨，180°=伸直（ADR-0001 §5.2）",
        }),
    "knee_flexion": MetricDef(
        "knee_flexion", "膝弯曲角", "sagittal", ("2d", "3d"), phase=True,
        definitions={
            "2d": "髋-膝-踝夹角，180°=伸直",
            "3d": "屈曲量，0°=伸直（ADR-0001 §5.3）",
        }),
    "trunk_lean": MetricDef(
        "trunk_lean", "躯干前倾角", "sagittal", ("2d", "3d"), phase=True,
        definitions={
            "2d": "图像平面内与竖直方向的偏角",
            "3d": "重力参考的矢状/冠状分量（ADR-0001 §5.4）",
        }),
    "knee_abduction": MetricDef(
        "knee_abduction", "膝内外翻角", "frontal", ("2d", "3d"), phase=False,
        aliases=("knee_valgus",),
        definitions={
            "2d": "二维投影：髋-膝-踝夹角，180°=伸直（旧 knee_valgus）",
            "3d": "JCS 外展，0°=无内外翻，正=内扣方向（ADR-0001 §5.5）",
        }),
    "hip_shoulder_separation": MetricDef(
        "hip_shoulder_separation", "肩髋横断面分离", "transverse", ("3d",), phase=False,
        definitions={"3d": "横断面 H-Ssep，正=左肩向前（ADR-0001 §5.6）"}),
    "thoraco_pelvic_obliquity": MetricDef(
        "thoraco_pelvic_obliquity", "肩髋冠状面倾斜", "frontal", ("3d",), phase=False,
        definitions={"3d": "肩线与髋线在冠状面的夹角，正=左肩高于右肩（ADR-0001 §5.7）"}),
}

# 只存在于二维过渡期的遗留键。shoulder_hip_align 是旧合并口径，三维拆成
# H-Sep 与 obliquity 两项，没有单一替代。
LEGACY_METRICS: dict[str, MetricDef] = {
    "shoulder_hip_align": MetricDef(
        "shoulder_hip_align", "肩髋对齐（旧合并口径）", "frontal", ("2d",), phase=False,
        definitions={"2d": "图像平面肩线-髋线夹角"},
        replaced_by=("hip_shoulder_separation", "thoraco_pelvic_obliquity")),
}

# 二维管线实际产出的键，顺序即报告顺序。knee_abduction 沿用旧 knee_valgus 的
# 位置与数值；shoulder_hip_align 保留到三维报告上线。
TWO_D_METRICS: tuple[str, ...] = (
    "hip_flexion", "knee_flexion", "trunk_lean", "knee_abduction", "shoulder_hip_align",
)
ALL_METRICS: tuple[str, ...] = tuple(METRICS)  # 六项规范指标（三维管线）


def _def(name: str) -> MetricDef:
    if name in METRICS:
        return METRICS[name]
    return LEGACY_METRICS[name]


def resolve_metric(name: str) -> str:
    """旧名 → 规范名；未知名字原样返回，由调用方报错。"""
    if name in METRICS:
        return name
    for canonical, d in METRICS.items():
        if name in d.aliases:
            return canonical
    return name


def metric_label(name: str) -> str:
    """规范名或遗留键 → 中文名；未知名字原样返回。"""
    try:
        return _def(name).label
    except KeyError:
        return name


def metric_plane(name: str) -> str:
    """规范名或遗留键 → 适用视角；未知名字返回空串（不参与门控）。"""
    try:
        return _def(name).plane
    except KeyError:
        return ""


def metrics_for_basis(basis: str) -> tuple[str, ...]:
    """某基准下的指标键。2d 返回二维五键（含遗留键），3d 返回六项规范指标。"""
    if basis == "2d":
        return TWO_D_METRICS
    return tuple(m for m, d in METRICS.items() if basis in d.bases)


def active_for_view(view: str) -> list[str]:
    """某视角判定下放行的二维指标——report 与 render 共用的唯一门控规则。

    mixed 放行全部：逐帧门控（compute_series 的 frame_view）已保证每个值
    出自其可信平面；单一视角按注册表平面筛；ambiguous 全不放行。此前这条
    规则在 report 与 render 各写一遍，报告与叠加层的指标集可能漂移。
    """
    if view == "mixed":
        return list(TWO_D_METRICS)
    return [m for m in TWO_D_METRICS if metric_plane(m) == view]


SAGITTAL_METRICS = tuple(m for m in TWO_D_METRICS if metric_plane(m) == "sagittal")
FRONTAL_METRICS = tuple(m for m in TWO_D_METRICS if metric_plane(m) == "frontal")


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
    return pick_side(float(left), float(right))


def compute_series(keypoints: np.ndarray, frame_view: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """逐帧计算二维五项指标（过渡期口径）。无效帧为 NaN。keypoints: (frames, n_kp, >=3)。

    frame_view：view.classify_view_frames 的逐帧归属（'f'/'l'/''）。
    传入时 sagittal 指标只在侧面帧算、frontal 指标只在正面帧算，''（无
    信号）帧不归属任何平面、两项都不算——"侧面指标取自侧面帧"的声明
    对每一帧都成立；None 时全帧计算（合成测试的向后兼容口径）。

    返回的 dict 额外带 knee_valgus 别名（与 knee_abduction 同一数组），
    供尚未迁移的旧调用方过渡使用。
    """
    n = keypoints.shape[0]
    out = {m: np.full(n, np.nan) for m in TWO_D_METRICS}

    for f in range(n):
        kp = keypoints[f]
        if frame_view is not None:
            fv = frame_view[f]
            if fv not in ("f", "l"):
                continue  # 无信号帧：不归属任何平面，不贡献任何指标
            do_sag, do_frt = fv == "l", fv == "f"
        else:
            do_sag = do_frt = True
        side = select_side(kp)
        sh, hip, knee, ank = SIDE_JOINTS[side]

        if do_sag and _ok(kp, sh, hip, knee):
            out["hip_flexion"][f] = angle_at(kp[sh], kp[hip], kp[knee])
        if do_sag and _ok(kp, hip, knee, ank):
            out["knee_flexion"][f] = angle_at(kp[hip], kp[knee], kp[ank])

        if do_sag and _ok(kp, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
            neck = (kp[L_SHOULDER][:2] + kp[R_SHOULDER][:2]) / 2.0
            pelvis = (kp[L_HIP][:2] + kp[R_HIP][:2]) / 2.0
            trunk = np.array([*neck, 1.0])
            out["trunk_lean"][f] = vertical_deviation(pelvis, trunk)

        if do_frt and _ok(kp, L_HIP, L_KNEE, L_ANKLE) and _ok(kp, R_HIP, R_KNEE, R_ANKLE):
            left_valgus = angle_at(kp[L_HIP], kp[L_KNEE], kp[L_ANKLE])
            right_valgus = angle_at(kp[R_HIP], kp[R_KNEE], kp[R_ANKLE])
            out["knee_abduction"][f] = min(left_valgus, right_valgus)  # 取更扣的一侧

        if do_frt and _ok(kp, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
            sh_line = kp[R_SHOULDER][:2] - kp[L_SHOULDER][:2]
            hip_line = kp[R_HIP][:2] - kp[L_HIP][:2]
            cross = abs(float(np.cross(sh_line, hip_line)))
            dot = float(np.dot(sh_line, hip_line))
            out["shoulder_hip_align"][f] = float(np.degrees(np.arctan2(cross, dot)))

    out["knee_valgus"] = out["knee_abduction"]  # 旧名别名，迁移完成后删除
    return out
