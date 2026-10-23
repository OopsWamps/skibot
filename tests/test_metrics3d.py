"""三维六项指标测试：python tests/test_metrics3d.py

用合成骨架覆盖 ADR-0001 §9 的验收用例：直立边界、屈曲单调、内外翻符号、
肩髋两项符号、绕重力轴旋转不变、重力倾斜只动躯干、正交投影与二维一致、
缺测与退化返回 NaN。最后经 assess + MockLifter 验证报告接入与视角门控取消。
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows GBK 控制台兜底

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skibot import metrics as M  # noqa: E402
from skibot.assess import assess  # noqa: E402
from skibot.metrics import METRICS  # noqa: E402
from skibot.metrics3d import THREE_D_SERIES, TRUNK_LEAN_SERIES, compute_series3d  # noqa: E402
from skibot.reference import load  # noqa: E402
from skibot.skeleton import (  # noqa: E402
    L_ANKLE, L_HIP, L_KNEE, L_SHOULDER, NOSE,
    R_ANKLE, R_HIP, R_KNEE, R_SHOULDER,
)
from skibot.skeleton3d import MockLifter, Skeleton3D  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def make_skeleton(pose: dict[int, tuple[float, float, float]], *,
                  gravity: tuple[float, float, float] = (0.0, 1.0, 0.0),
                  source: str = "adapter", conf: float = 0.9) -> Skeleton3D:
    joints = np.full((1, 26, 3), np.nan)
    confidence = np.zeros((1, 26))
    for idx, xyz in pose.items():
        joints[0, idx] = xyz
        confidence[0, idx] = conf
    return Skeleton3D(joints, confidence,
                      gravity_up=np.array(gravity, dtype=float),
                      source=source,
                      meta={"gravity_source": source})


def series(pose: dict[int, tuple[float, float, float]], *,
           slope_sag_deg: float | None = None,
           gravity: tuple[float, float, float] = (0.0, 1.0, 0.0),
           source: str = "adapter") -> dict[str, float]:
    sk = make_skeleton(pose, gravity=gravity, source=source)
    out = compute_series3d(sk, slope_sag_deg=slope_sag_deg)
    return {key: float(values[0]) for key, values in out.items()}


def upright_pose() -> dict[int, tuple[float, float, float]]:
    return {
        L_SHOULDER: (-0.1, 0.5, 0.0), R_SHOULDER: (0.1, 0.5, 0.0),
        L_HIP: (-0.1, 0.0, 0.0), R_HIP: (0.1, 0.0, 0.0),
        L_KNEE: (-0.1, -0.4, 0.0), R_KNEE: (0.1, -0.4, 0.0),
        L_ANKLE: (-0.1, -0.8, 0.0), R_ANKLE: (0.1, -0.8, 0.0),
    }


def test_upright_boundaries():
    s = series(upright_pose())
    assert abs(s["hip_flexion"] - 180.0) < 1e-6
    assert abs(s["knee_flexion"]) < 1e-6
    assert abs(s["trunk_lean_sagittal"]) < 1e-6
    assert abs(s["trunk_lean_frontal"]) < 1e-6
    assert abs(s["trunk_lean_total"]) < 1e-6
    assert abs(s["knee_abduction_left"]) < 1e-6
    assert abs(s["knee_abduction_right"]) < 1e-6
    assert abs(s["knee_abduction_primary"]) < 1e-6
    assert abs(s["hip_shoulder_separation"]) < 1e-6
    assert abs(s["thoraco_pelvic_obliquity"]) < 1e-6
    assert abs(s["hip_flexion_3pt"] - 180.0) < 1e-6
    assert abs(s["knee_interior_3pt"] - 180.0) < 1e-6
    assert np.isnan(s["trunk_lean_slope"])
    print("  直立：六项边界值 + 两点辅助量 ✓")


def test_knee_flexion_90():
    pose = upright_pose()
    pose[L_ANKLE] = (-0.1, -0.4, 0.4)   # 胫骨向前水平，膝屈曲 90°
    pose[R_ANKLE] = (0.1, -0.4, 0.4)
    s = series(pose)
    assert abs(s["knee_flexion"] - 90.0) < 1e-6
    assert abs(s["knee_interior_3pt"] - 90.0) < 1e-6
    assert abs(s["knee_abduction_left"]) < 1e-6
    assert abs(s["knee_abduction_right"]) < 1e-6
    print("  矢状面屈膝 90°：膝弯量 / 辅助量 / 外展 ✓")


def _hip_pose(phi_deg: float) -> dict[int, tuple[float, float, float]]:
    th = np.radians(phi_deg)
    hip = (0.0, 0.0, 0.0)
    knee = (0.0, -np.cos(th), np.sin(th))
    ankle = (0.0, -2.0 * np.cos(th), 2.0 * np.sin(th))
    return {
        L_SHOULDER: (0.0, 0.5, 0.0), R_SHOULDER: (0.0, 0.5, 0.0),
        L_HIP: hip, R_HIP: hip, L_KNEE: knee, R_KNEE: knee,
        L_ANKLE: ankle, R_ANKLE: ankle,
    }


def test_hip_flexion_monotonic():
    vals = [series(_hip_pose(p))["hip_flexion"] for p in (0, 15, 30, 45, 60)]
    assert all(a > b for a, b in zip(vals, vals[1:])), vals
    assert abs(vals[0] - 180.0) < 1e-6
    assert abs(vals[-1] - 120.0) < 1e-6
    print("  屈髋：随股骨前摆单调变小 ✓")


def _knee_pose(phi_deg: float) -> dict[int, tuple[float, float, float]]:
    th = np.radians(phi_deg)
    hip = (0.0, 0.0, 0.0)
    knee = (0.0, -1.0, 0.0)
    ankle = (0.0, -1.0 - np.cos(th), np.sin(th))
    return {
        L_SHOULDER: (0.0, 0.5, 0.0), R_SHOULDER: (0.0, 0.5, 0.0),
        L_HIP: hip, R_HIP: hip, L_KNEE: knee, R_KNEE: knee,
        L_ANKLE: ankle, R_ANKLE: ankle,
    }


def test_knee_flexion_monotonic():
    vals = [series(_knee_pose(p))["knee_flexion"] for p in (0, 15, 30, 45, 60, 90)]
    assert all(a < b for a, b in zip(vals, vals[1:])), vals
    assert abs(vals[0]) < 1e-6
    assert abs(vals[-1] - 90.0) < 1e-6
    print("  膝弯：随胫骨后摆单调变大 ✓")


def _valgus_pose(deg: float, side: str, sign: float = 1.0
                 ) -> dict[int, tuple[float, float, float]]:
    """side 腿的膝/踝向中线（sign=1）或向外（sign=-1）偏移 deg 度。"""
    th = np.radians(deg) * sign
    pose = upright_pose()
    if side == "left":
        kx = -0.1 + 0.4 * np.sin(th)
        ky = -0.4 * np.cos(th)
        pose[L_KNEE] = (kx, ky, 0.0)
        pose[L_ANKLE] = (kx, ky - 0.4, 0.0)
    else:
        kx = 0.1 - 0.4 * np.sin(th)
        ky = -0.4 * np.cos(th)
        pose[R_KNEE] = (kx, ky, 0.0)
        pose[R_ANKLE] = (kx, ky - 0.4, 0.0)
    return pose


def test_knee_abduction_sign_and_monotonic():
    for side in ("left", "right"):
        vals = [series(_valgus_pose(d, side))[f"knee_abduction_{side}"]
                for d in (0, 5, 10, 20)]
        assert all(a <= b + 1e-9 for a, b in zip(vals, vals[1:])), (side, vals)
        assert abs(vals[-1] - 20.0) < 1e-6
        varus = series(_valgus_pose(10, side, sign=-1.0))[f"knee_abduction_{side}"]
        assert abs(varus + 10.0) < 1e-6, (side, varus)
    print("  膝内外翻：向内为正、向外为负，随角度单调 ✓")


def _hsep_pose(d: float) -> dict[int, tuple[float, float, float]]:
    pose = upright_pose()
    pose[L_SHOULDER] = (-0.1, 0.5, d)   # 左肩向前、右肩向后，躯干轴仍竖直
    pose[R_SHOULDER] = (0.1, 0.5, -d)
    return pose


def _obliquity_pose(d: float) -> dict[int, tuple[float, float, float]]:
    pose = upright_pose()
    pose[L_SHOULDER] = (-0.1, 0.5 + d, 0.0)   # 左肩高于右肩
    pose[R_SHOULDER] = (0.1, 0.5 - d, 0.0)
    return pose


def test_separation_and_obliquity_signs():
    hsep = [series(_hsep_pose(d))["hip_shoulder_separation"]
            for d in (0, 0.05, 0.1, 0.2)]
    assert all(a < b for a, b in zip(hsep, hsep[1:])), hsep
    assert abs(hsep[2] - 45.0) < 1e-6
    obliq = [series(_obliquity_pose(d))["thoraco_pelvic_obliquity"]
             for d in (0, 0.05, 0.1, 0.2)]
    assert all(a < b for a, b in zip(obliq, obliq[1:])), obliq
    assert abs(obliq[2] - 45.0) < 1e-6
    print("  H-Ssep 左肩向前为正 / obliquity 左肩高为正 ✓")


def _rotate_pose(pose: dict[int, tuple[float, float, float]], deg: float
                 ) -> dict[int, tuple[float, float, float]]:
    th = np.radians(deg)
    rot = np.array([[np.cos(th), 0.0, np.sin(th)],
                    [0.0, 1.0, 0.0],
                    [-np.sin(th), 0.0, np.cos(th)]])
    return {idx: tuple(float(x) for x in rot @ np.array(xyz))
            for idx, xyz in pose.items()}


def test_rotation_about_gravity_invariance():
    pose = _valgus_pose(12, "left")
    pose[L_ANKLE] = (-0.1, -0.4, 0.3)
    base = series(pose)
    for deg in (30, 90, 180):
        rotated = series(_rotate_pose(pose, deg))
        for key in THREE_D_SERIES:
            a, b = base[key], rotated[key]
            if np.isnan(a):
                assert np.isnan(b), (key, deg)
            else:
                assert abs(a - b) < 1e-6, (key, deg, a, b)
    print("  骨架与重力一起绕重力轴旋转：六项不变 ✓")


def test_trunk_total_independent_of_shoulder_line():
    # 双肩重合：s_body/a 退化，但 total 只依赖躯干轴与重力，仍应为 0
    pose = upright_pose()
    pose[L_SHOULDER] = (0.0, 0.5, 0.0)
    pose[R_SHOULDER] = (0.0, 0.5, 0.0)
    s = series(pose)
    assert abs(s["trunk_lean_total"]) < 1e-6
    assert np.isnan(s["trunk_lean_sagittal"])

    # 肩线与躯干轴平行：同样只有 total 有值
    pose2 = upright_pose()
    pose2[L_SHOULDER] = (0.0, 0.6, 0.0)
    pose2[R_SHOULDER] = (0.0, 0.4, 0.0)
    s2 = series(pose2)
    assert abs(s2["trunk_lean_total"]) < 1e-6
    assert np.isnan(s2["trunk_lean_sagittal"])

    # 躯干水平前向：total = 90°
    pose3 = {
        L_SHOULDER: (0.0, 0.0, 0.6), R_SHOULDER: (0.0, 0.0, 0.2),
        L_HIP: (0.0, 0.0, 0.0), R_HIP: (0.0, 0.0, 0.0),
        L_KNEE: (0.0, -0.4, 0.0), R_KNEE: (0.0, -0.4, 0.0),
        L_ANKLE: (0.0, -0.8, 0.0), R_ANKLE: (0.0, -0.8, 0.0),
    }
    s3 = series(pose3)
    assert abs(s3["trunk_lean_total"] - 90.0) < 1e-6
    assert np.isnan(s3["trunk_lean_sagittal"])
    print("  trunk_lean_total 不依赖肩线退化门控 ✓")


def test_per_frame_gravity():
    pose = upright_pose()
    joints = np.full((2, 26, 3), np.nan)
    confidence = np.zeros((2, 26))
    for idx, xyz in pose.items():
        joints[:, idx] = xyz
        confidence[:, idx] = 0.9
    gravity = np.array([[0.0, 1.0, 0.0],
                        [0.0, np.cos(np.radians(10)), np.sin(np.radians(10))]])
    sk = Skeleton3D(joints, confidence, gravity_up=gravity,
                    meta={"gravity_source": "adapter"})
    assert sk.gravity_up.shape == (2, 3)
    out = compute_series3d(sk)
    assert abs(out["trunk_lean_sagittal"][0]) < 1e-6
    assert abs(out["trunk_lean_sagittal"][1] + 10.0) < 1e-6
    assert abs(out["hip_flexion"][0] - out["hip_flexion"][1]) < 1e-6
    print("  逐帧重力向量：躯干前倾逐帧生效，体固指标不变 ✓")


def test_scale_invariance():
    pose = _valgus_pose(12.0, "left")
    pose[L_ANKLE] = (-0.1, -0.4, 0.3)
    base = series(pose)
    for factor in (1e-8, 1e-3, 1.0, 1e3, 1e8, 1e10, 1e12):
        scaled = {idx: tuple(c * factor for c in xyz)
                  for idx, xyz in pose.items()}
        vals = series(scaled)
        for key in THREE_D_SERIES:
            a, b = base[key], vals[key]
            if np.isnan(a):
                assert np.isnan(b), (key, factor)
            else:
                assert abs(a - b) < 1e-6, (key, factor, a, b)
    print("  坐标单位缩放 1e-8~1e12：六项结果不变 ✓")


def test_gravity_tilt_changes_only_trunk():
    base = series(upright_pose())
    g = (0.0, np.cos(np.radians(10)), np.sin(np.radians(10)))
    tilted = series(upright_pose(), gravity=g)
    for key in ("hip_flexion", "knee_flexion", "knee_abduction_primary",
                "knee_abduction_left", "knee_abduction_right",
                "hip_shoulder_separation", "thoraco_pelvic_obliquity",
                "hip_flexion_3pt", "knee_interior_3pt"):
        assert abs(tilted[key] - base[key]) < 1e-6, key
    assert abs(tilted["trunk_lean_sagittal"] + 10.0) < 1e-6
    assert abs(tilted["trunk_lean_frontal"]) < 1e-6
    assert abs(tilted["trunk_lean_total"] - 10.0) < 1e-6
    print("  重力倾斜：只动躯干前倾，体固指标不变 ✓")


def _bent_sagittal_pose(hip_deg: float, knee_deg: float
                        ) -> dict[int, tuple[float, float, float]]:
    ah, ak = np.radians(hip_deg), np.radians(knee_deg)
    hip = (0.0, 0.0, 0.0)
    knee = (0.0, -np.cos(ah), np.sin(ah))
    ankle = (0.0, knee[1] - np.cos(ak), knee[2] + np.sin(ak))
    return {
        L_SHOULDER: (0.0, 0.5, 0.0), R_SHOULDER: (0.0, 0.5, 0.0),
        L_HIP: hip, R_HIP: hip, L_KNEE: knee, R_KNEE: knee,
        L_ANKLE: ankle, R_ANKLE: ankle,
    }


def _project_to_image(pose: dict[int, tuple[float, float, float]]) -> np.ndarray:
    """相机沿 +x 看：3D 的 (y, z) 变成图像的 (y, x_2d)。"""
    kp = np.zeros((26, 3))
    for idx, (_, y, z) in pose.items():
        kp[idx] = [z, y, 0.9]
    return kp[None]


def test_projection_equivalence():
    pose = _bent_sagittal_pose(30.0, 60.0)
    s3 = series(pose)
    s2 = M.compute_series(_project_to_image(pose))
    assert abs(s3["hip_flexion"] - s2["hip_flexion"][0]) < 0.1
    assert abs(s3["knee_flexion"] - (180.0 - s2["knee_flexion"][0])) < 0.1
    assert abs(s3["hip_flexion_3pt"] - s2["hip_flexion"][0]) < 0.1
    assert abs(s3["knee_interior_3pt"] - s2["knee_flexion"][0]) < 0.1
    print("  正交投影：三维髋/膝与二维口径误差 < 0.1° ✓")


def test_missing_and_degenerate():
    pose = upright_pose()
    del pose[L_KNEE]
    del pose[R_KNEE]
    s = series(pose)
    assert np.isnan(s["hip_flexion"])
    assert np.isnan(s["knee_flexion"])
    assert np.isnan(s["knee_abduction_primary"])
    assert not np.isnan(s["hip_shoulder_separation"])
    assert not np.isnan(s["thoraco_pelvic_obliquity"])
    assert not np.isnan(s["trunk_lean_sagittal"])

    pose2 = upright_pose()
    pose2[L_ANKLE] = pose2[L_KNEE]      # 踝膝重合：胫骨向量退化
    pose2[R_ANKLE] = pose2[R_KNEE]
    s2 = series(pose2)
    assert np.isnan(s2["knee_flexion"])
    assert np.isnan(s2["knee_abduction_primary"])
    assert not np.isnan(s2["hip_flexion"])

    pose3 = upright_pose()
    pose3[L_HIP] = (0.0, 0.0, 0.0)      # 髋线退化
    pose3[R_HIP] = (0.0, 0.0, 0.0)
    s3 = series(pose3)
    assert np.isnan(s3["knee_abduction_primary"])
    assert np.isnan(s3["hip_shoulder_separation"])
    assert np.isnan(s3["thoraco_pelvic_obliquity"])
    assert not np.isnan(s3["hip_flexion"])

    zero = Skeleton3D(np.zeros((2, 26, 3)), np.full((2, 26), 0.9),
                      meta={"gravity_source": "adapter"})
    out = compute_series3d(zero)
    for key in THREE_D_SERIES:
        assert np.isnan(out[key]).all(), key

    missing = Skeleton3D(np.full((1, 26, 3), np.nan), np.zeros((1, 26)),
                         meta={"gravity_source": "adapter"})
    out = compute_series3d(missing)
    for key in THREE_D_SERIES:
        assert np.isnan(out[key]).all(), key
    print("  缺关节 / 向量退化 / 全零帧 / 全缺帧：NaN，不抛异常 ✓")


def test_unknown_gravity_and_slope():
    s = series(upright_pose(), source="unknown")
    for key in TRUNK_LEAN_SERIES:
        assert np.isnan(s[key]), key
    assert not np.isnan(s["hip_flexion"])

    slope = series(upright_pose(), slope_sag_deg=5.0)
    assert abs(slope["trunk_lean_sagittal"]) < 1e-6
    assert abs(slope["trunk_lean_slope"] + 5.0) < 1e-6
    g = (0.0, np.cos(np.radians(10)), np.sin(np.radians(10)))
    tilted = series(upright_pose(), gravity=g, slope_sag_deg=5.0)
    assert abs(tilted["trunk_lean_slope"] + 15.0) < 1e-6
    print("  重力来源可信度 / 坡面法向版本 ✓")


def side_frames_2d(n: int = 20) -> np.ndarray:
    frames = []
    for _ in range(n):
        kp = np.zeros((26, 3))
        kp[L_SHOULDER] = [100, 100, 0.9]
        kp[R_SHOULDER] = [105, 100, 0.9]
        kp[L_HIP] = [110, 200, 0.9]
        kp[R_HIP] = [112, 200, 0.9]
        kp[L_KNEE] = [110, 300, 0.9]
        kp[R_KNEE] = [111, 300, 0.9]
        kp[L_ANKLE] = [110, 400, 0.9]
        kp[R_ANKLE] = [111, 400, 0.9]
        kp[NOSE] = [130, 60, 0.9]
        frames.append(kp)
    return np.stack(frames)


def frontal_frame_2d() -> np.ndarray:
    kp = np.zeros((26, 3))
    kp[L_SHOULDER] = [90, 100, 0.9]
    kp[R_SHOULDER] = [120, 100, 0.9]
    kp[L_HIP] = [98, 200, 0.9]
    kp[R_HIP] = [112, 200, 0.9]
    kp[L_KNEE] = [96, 300, 0.9]
    kp[R_KNEE] = [114, 300, 0.9]
    kp[L_ANKLE] = [95, 400, 0.9]
    kp[R_ANKLE] = [115, 400, 0.9]
    kp[NOSE] = [105, 60, 0.9]
    return kp


def test_assess_report_and_2d_regression():
    refs = load(ROOT / "reference_values.toml")
    kp = side_frames_2d()
    res2 = assess(kp, level="novice", refs=refs, label="side.mp4")
    res3 = assess(kp, level="novice", refs=refs, label="side.mp4",
                  lifter=MockLifter(), slope_sag_deg=5.0)

    metrics2d = [m for m in res3.report["metrics"] if m["basis"] == "2d"]
    metrics3d = [m for m in res3.report["metrics"] if m["basis"] == "3d"]
    assert metrics2d == res2.report["metrics"]   # 二维逐值不变
    assert res3.report["data_source"] == "mock"
    assert "data_source" not in res2.report
    by2 = {m["metric"]: m for m in metrics2d}
    assert by2["hip_flexion"]["definition"]
    assert by2["knee_flexion"]["definition"]
    assert by2["trunk_lean"]["definition"]
    assert [m["metric"] for m in metrics3d] == list(METRICS)
    assert all(m["basis"] == "3d" for m in metrics3d)
    assert all(m["reference"] is None for m in metrics3d)
    assert all(m["verdict"] is None for m in metrics3d)
    assert all(m["n_valid_frames"] == kp.shape[0] for m in metrics3d)

    by = {m["metric"]: m for m in metrics3d}
    assert by["hip_flexion"]["helper"]["hip_flexion_3pt_deg"] is not None
    assert by["knee_flexion"]["helper"]["knee_interior_3pt_deg"] is not None
    assert by["trunk_lean"]["gravity_source"] == "image_vertical"
    assert by["trunk_lean"]["slope_source"] == "imu"
    assert by["trunk_lean"]["slope_normal_deg"] is not None
    assert by["knee_abduction"]["left_deg"] is not None
    assert by["knee_abduction"]["right_deg"] is not None
    assert by["knee_abduction"]["side"] in ("left", "right")
    assert by["hip_shoulder_separation"]["abs_median_deg"] is not None
    assert by["thoraco_pelvic_obliquity"]["abs_median_deg"] is not None
    assert set(res3.series3d) == set(THREE_D_SERIES)
    assert "view_frame_ratio" in res3.report["quality"]
    assert "view_frame_ratio" not in res2.report["quality"]  # 二维报告逐位不变
    print("  assess：二维回归不变 + 三维六项带基准/辅助量 ✓")


def test_missing_frame_nulls_all_3d():
    refs = load(ROOT / "reference_values.toml")
    kp = side_frames_2d()
    kp[0] = 0.0
    res = assess(kp, level="novice", refs=refs, label="side.mp4",
                 lifter=MockLifter())
    for key in THREE_D_SERIES:
        assert np.isnan(res.series3d[key][0]), key
    by = {m["metric"]: m for m in res.report["metrics"] if m["basis"] == "3d"}
    assert all(m["n_valid_frames"] == kp.shape[0] - 1 for m in by.values())
    assert all(m["median_deg"] is not None for m in by.values())
    print("  缺数据帧：六项为 null，不产生 0 ✓")


def test_3d_ignores_view_gating():
    refs = load(ROOT / "reference_values.toml")
    side = side_frames_2d(10)
    front = np.stack([frontal_frame_2d() for _ in range(10)])
    kp = np.concatenate([side, front])
    res = assess(kp, level="novice", refs=refs, label="mixed.mp4",
                 lifter=MockLifter())
    assert res.report["view"]["detected"] == "mixed"
    assert (res.frame_view[:10] == "l").all()
    assert (res.frame_view[10:] == "f").all()
    # 二维侧面指标在正面帧被门控；三维不分视角，所有有效帧都有值
    assert np.isnan(res.series["hip_flexion"][10:]).all()
    assert np.isfinite(res.series3d["hip_flexion"][10:]).all()
    by = {m["metric"]: m for m in res.report["metrics"] if m["basis"] == "3d"}
    assert by["hip_flexion"]["n_valid_frames"] == kp.shape[0]
    ratio = res.report["quality"]["view_frame_ratio"]
    assert ratio["sagittal"] > 0 and ratio["frontal"] > 0
    print("  视角门控取消：三维全帧计算，视角只进质量统计 ✓")


if __name__ == "__main__":
    test_upright_boundaries()
    test_knee_flexion_90()
    test_hip_flexion_monotonic()
    test_knee_flexion_monotonic()
    test_knee_abduction_sign_and_monotonic()
    test_separation_and_obliquity_signs()
    test_rotation_about_gravity_invariance()
    test_trunk_total_independent_of_shoulder_line()
    test_per_frame_gravity()
    test_scale_invariance()
    test_gravity_tilt_changes_only_trunk()
    test_projection_equivalence()
    test_missing_and_degenerate()
    test_unknown_gravity_and_slope()
    test_assess_report_and_2d_regression()
    test_missing_frame_nulls_all_3d()
    test_3d_ignores_view_gating()
    print("全部通过")
