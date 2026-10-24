"""三维六项 JCS 指标与三点夹角辅助量（ADR-0001 §5）。

输入是 `Skeleton3D`（root-relative、重力对齐、逐关节置信度），输出逐帧角度
序列。参考系是体固 / 重力 / 坡面，相机坐标不进入读数。缺测返回 NaN，不补 0、
不跨帧插值。视角分类不参与这里：三维指标对所有有效帧计算，视角只进质量统计。

序列键：

- `hip_flexion`：躯干轴-股骨，180° = 伸直，屈髋增加时变小。
- `knee_flexion`：屈曲量，0° = 伸直；二维回归列的髋-膝-踝夹角 = 180 - 本值。
- `trunk_lean_sagittal`：重力参考矢状分量，正 = 前倾。
- `trunk_lean_frontal`：重力参考冠状分量，正 = 向受试者左侧倾。
- `trunk_lean_total`：躯干轴与重力轴夹角。
- `trunk_lean_slope`：相对坡面法线的矢状分量；坡度 β 缺失时为 NaN。
- `knee_abduction_primary` / `_left` / `_right`：正 = 膝向中线（内扣方向）。
- `hip_shoulder_separation`：横断面 ψ，正 = 左肩向前。
- `thoraco_pelvic_obliquity`：冠状面 φ，正 = 左肩高于右肩。
- `hip_flexion_3pt`：同侧肩-髋-膝夹角，180° = 伸直（辅助量）。
- `knee_interior_3pt`：同侧髋-膝-踝夹角，180° = 伸直（辅助量）。

躯干前倾的重力来源必须是 calibrated / adapter / image_vertical 之一；
unknown 时四个分量全为 NaN（ADR-0001 §5.1，不拿缺省重力出数）。
"""

from __future__ import annotations

import numpy as np

from .skeleton import (
    L_ANKLE, L_HIP, L_KNEE, L_SHOULDER,
    R_ANKLE, R_HIP, R_KNEE, R_SHOULDER, SIDE_JOINTS, pick_side,
)
from .skeleton3d import CONF_MIN_3D, GRAVITY_SOURCES, Skeleton3D

_REL_EPS = 1e-9       # 长度向量的退化阈值，相对骨架尺度
_UNIT_EPS = 1e-9      # 单位向量之间（叉积范数）的退化阈值，无量纲
_DOT_EPS = 1e-12      # 单位向量点积的零判定

TRUNK_LEAN_SERIES = (
    "trunk_lean_sagittal", "trunk_lean_frontal",
    "trunk_lean_total", "trunk_lean_slope",
)
KNEE_ABDUCTION_SERIES = (
    "knee_abduction_primary", "knee_abduction_left", "knee_abduction_right",
)
HELPER_SERIES = ("hip_flexion_3pt", "knee_interior_3pt")

THREE_D_SERIES = (
    "hip_flexion", "knee_flexion",
    *TRUNK_LEAN_SERIES,
    *KNEE_ABDUCTION_SERIES,
    "hip_shoulder_separation", "thoraco_pelvic_obliquity",
    *HELPER_SERIES,
)


def _unit(v: np.ndarray, tol: float) -> np.ndarray | None:
    """单位化；零向量或非有限输入返回 None（调用方记 NaN）。

    tol 由骨架尺度乘 `_REL_EPS` 得到，不用绝对阈值——同一姿态换长度单位
    （米/厘米/归一化坐标）时退化判定不变。
    """
    if not np.isfinite(v).all():
        return None
    n = float(np.linalg.norm(v))
    if n <= tol:
        return None
    return v / n


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.degrees(np.arccos(float(np.clip(np.dot(u, v), -1.0, 1.0)))))


def select_side(sk: Skeleton3D, f: int) -> str:
    """按契约置信度选一侧；平票取左，规则唯一实现在 skeleton.pick_side。"""
    left = float(np.mean(sk.confidence[f, list(SIDE_JOINTS["left"])]))
    right = float(np.mean(sk.confidence[f, list(SIDE_JOINTS["right"])]))
    return pick_side(left, right)


def _frame_series(sk: Skeleton3D, f: int, *, side: str,
                  gravity_source: str,
                  slope_sag_deg: float | None) -> dict[str, float]:
    out = {key: np.nan for key in THREE_D_SERIES}

    j = sk.joints[f]
    conf = sk.confidence[f]

    # 骨架尺度：所有有限关节到其质心的最大距离。用它定退化阈值，坐标单位
    # 任意缩放时阈值同步缩放（ADR-0001 §3.2 的尺度不变要求）。
    finite = np.isfinite(j).all(axis=1)
    if not finite.any():
        return out
    pts = j[finite]
    scale = float(np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))
    if not np.isfinite(scale) or scale <= 0.0:
        return out
    tol = _REL_EPS * scale

    def ok(idx: int) -> bool:
        return bool(np.isfinite(j[idx]).all()) and float(conf[idx]) >= CONF_MIN_3D

    side_shoulder, side_hip, side_knee, side_ankle = SIDE_JOINTS[side]

    # 躯干轴、肩线、髋线。缺肩或缺髋时这些公共量不存在，相关指标记 NaN。
    u_t = None
    if ok(L_SHOULDER) and ok(R_SHOULDER) and ok(L_HIP) and ok(R_HIP):
        mid_sh = (j[L_SHOULDER] + j[R_SHOULDER]) / 2.0
        mid_hip = (j[L_HIP] + j[R_HIP]) / 2.0
        u_t = _unit(mid_sh - mid_hip, tol)

    s = j[L_SHOULDER] - j[R_SHOULDER] if ok(L_SHOULDER) and ok(R_SHOULDER) else None
    h = j[L_HIP] - j[R_HIP] if ok(L_HIP) and ok(R_HIP) else None

    s_body = a = None
    if u_t is not None and s is not None:
        s_body = _unit(s - float(np.dot(s, u_t)) * u_t, tol)
        if s_body is not None:
            # 叉积的两个输入都是单位向量，范数无量纲；不能用随坐标尺度
            # 增长的 tol，否则大尺度骨架会把有效前向轴判成退化。
            a = _unit(np.cross(u_t, s_body), _UNIT_EPS)

    # 屈髋：躯干轴-股骨。辅助量用同侧肩，口径与二维一致。
    if u_t is not None and ok(side_hip) and ok(side_knee):
        femur = _unit(j[side_knee] - j[side_hip], tol)
        if femur is not None:
            out["hip_flexion"] = _angle_deg(u_t, femur)
    if ok(side_shoulder) and ok(side_hip) and ok(side_knee):
        v1 = _unit(j[side_shoulder] - j[side_hip], tol)
        v2 = _unit(j[side_knee] - j[side_hip], tol)
        if v1 is not None and v2 is not None:
            out["hip_flexion_3pt"] = _angle_deg(v1, v2)

    # 膝弯：屈曲量 0° = 伸直；辅助量是教练惯用的膝夹角。
    if ok(side_hip) and ok(side_knee) and ok(side_ankle):
        v_t = _unit(j[side_hip] - j[side_knee], tol)
        v_s = _unit(j[side_ankle] - j[side_knee], tol)
        if v_t is not None and v_s is not None:
            interior = _angle_deg(v_t, v_s)
            out["knee_flexion"] = 180.0 - interior
            out["knee_interior_3pt"] = interior

    # 躯干前倾：重力参考。重力来源不可信时整项不出数。
    # total 只依赖躯干轴与重力，肩线退化时仍应出数（ADR-0001 §5.4）。
    if u_t is not None and gravity_source in GRAVITY_SOURCES:
        g = sk.gravity_up[f]
        out["trunk_lean_total"] = _angle_deg(u_t, g)
        if s_body is not None and a is not None:
            g_u = float(np.dot(g, u_t))
            g_a = float(np.dot(g, a))
            g_s = float(np.dot(g, s_body))
            if abs(g_u) > _DOT_EPS or abs(g_a) > _DOT_EPS:
                out["trunk_lean_sagittal"] = float(np.degrees(np.arctan2(-g_a, g_u)))
            if abs(g_u) > _DOT_EPS or abs(g_s) > _DOT_EPS:
                out["trunk_lean_frontal"] = float(np.degrees(np.arctan2(-g_s, g_u)))
            if slope_sag_deg is not None and not np.isnan(out["trunk_lean_sagittal"]):
                out["trunk_lean_slope"] = out["trunk_lean_sagittal"] - slope_sag_deg

    # 膝内外翻：左右分别算，主值取更偏内扣的一侧。
    if h is not None:
        for side_name, (hip_i, knee_i, ankle_i) in (
                ("left", (L_HIP, L_KNEE, L_ANKLE)),
                ("right", (R_HIP, R_KNEE, R_ANKLE))):
            if not (ok(hip_i) and ok(knee_i) and ok(ankle_i)):
                continue
            femur = _unit(j[knee_i] - j[hip_i], tol)
            shank = _unit(j[ankle_i] - j[knee_i], tol)
            if femur is None or shank is None:
                continue
            e1 = _unit(h - float(np.dot(h, femur)) * femur, tol)
            if e1 is None:
                continue
            beta = _angle_deg(e1, shank)
            if side_name == "right":
                out["knee_abduction_right"] = beta - 90.0
            else:
                out["knee_abduction_left"] = 90.0 - beta
        left = out["knee_abduction_left"]
        right = out["knee_abduction_right"]
        if not np.isnan(left) and not np.isnan(right):
            out["knee_abduction_primary"] = max(left, right)
        elif not np.isnan(left):
            out["knee_abduction_primary"] = left
        elif not np.isnan(right):
            out["knee_abduction_primary"] = right

    # H-Ssep：肩线/髋线在躯干横断面内的投影夹角。
    if u_t is not None and h is not None and s is not None:
        h_p = h - float(np.dot(h, u_t)) * u_t
        s_p = s - float(np.dot(s, u_t)) * u_t
        if np.linalg.norm(h_p) > tol and np.linalg.norm(s_p) > tol:
            out["hip_shoulder_separation"] = float(np.degrees(np.arctan2(
                float(np.dot(np.cross(h_p, s_p), u_t)),
                float(np.dot(h_p, s_p)))))

    # 肩髋冠状面倾斜：投影到以身体前向轴为法向的平面。
    if a is not None and s is not None and h is not None:
        s_f = s - float(np.dot(s, a)) * a
        h_f = h - float(np.dot(h, a)) * a
        if np.linalg.norm(s_f) > tol and np.linalg.norm(h_f) > tol:
            out["thoraco_pelvic_obliquity"] = float(np.degrees(np.arctan2(
                float(np.dot(np.cross(s_f, h_f), a)),
                float(np.dot(s_f, h_f)))))

    return out


def compute_series3d(skeleton: Skeleton3D, *,
                     slope_sag_deg: float | None = None,
                     gravity_source: str | None = None) -> dict[str, np.ndarray]:
    """逐帧计算三维六项指标与辅助量；无效帧、缺测指标为 NaN。

    slope_sag_deg：沿滚落线的坡度投影到受试者矢状面的带符号分量
    （ADR-0001 §5.1）；None 表示没有现场 IMU 读数，`trunk_lean_slope` 为 NaN。
    gravity_source：默认取契约字段 `skeleton.gravity_source`；不在
    calibrated / adapter / image_vertical 里时躯干前倾全为 NaN。
    """
    if gravity_source is None:
        gravity_source = skeleton.gravity_source

    out = {key: np.full(skeleton.n_frames, np.nan) for key in THREE_D_SERIES}
    valid = skeleton.valid_mask
    for f in range(skeleton.n_frames):
        if not valid[f]:
            continue
        side = select_side(skeleton, f)
        values = _frame_series(
            skeleton, f, side=side, gravity_source=gravity_source,
            slope_sag_deg=slope_sag_deg)
        for key, value in values.items():
            out[key][f] = value
    return out
