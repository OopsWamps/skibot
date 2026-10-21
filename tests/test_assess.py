"""评估流程测试：python tests/test_assess.py
assess 是主路径与抢救路径共用的 seam：合成侧面骨架进，报告 dict 出。
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows GBK 控制台兜底

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skibot.assess import assess  # noqa: E402
from skibot.metrics import SAGITTAL_METRICS  # noqa: E402
from skibot.reference import load  # noqa: E402
from skibot.skeleton import (  # noqa: E402
    L_ANKLE, L_ELBOW, L_HIP, L_KNEE, L_SHOULDER, R_ANKLE, R_ELBOW, R_HIP,
    R_KNEE, R_SHOULDER,
)

ROOT = Path(__file__).resolve().parent.parent


def side_view_frames(n: int = 20) -> np.ndarray:
    """n 帧直立侧面骨架：肩线近重合、鼻子偏出，分类为 sagittal。

    左右侧肢体置信度不对称（远侧低）——喂 view 的信号 3。
    """
    frames = []
    for _ in range(n):
        kp = np.zeros((26, 3))
        kp[L_SHOULDER] = [100, 100, 0.9]
        kp[R_SHOULDER] = [105, 100, 0.9]
        kp[L_HIP] = [110, 200, 0.9]
        kp[R_HIP] = [112, 200, 0.9]
        kp[L_KNEE] = [110, 300, 0.9]
        kp[R_KNEE] = [111, 300, 0.9]
        # 左近右远：右侧肢体置信度低 → 左右不对称（>0.25）→ 侧面票
        kp[L_ELBOW] = [108, 150, 0.8]
        kp[R_ELBOW] = [106, 150, 0.2]
        kp[L_ANKLE] = [110, 400, 0.9]
        kp[R_ANKLE] = [111, 400, 0.2]
        kp[0] = [130, 60, 0.9]  # NOSE 偏出肩中点
        frames.append(kp)
    return np.stack(frames)


def test_assess_side_view_report():
    refs = load(ROOT / "reference_values.toml")
    report = assess(side_view_frames(), level="novice", refs=refs, label="t.mp4",
                    person={"id": 0, "selection": "测试"})
    assert report["view"]["detected"] == "sagittal", report["view"]
    assert [m["metric"] for m in report["metrics"]] == list(SAGITTAL_METRICS)
    assert report["person"] == {"id": 0, "selection": "测试"}
    assert report["calibrated"] is False and report["notice"]
    print("  assess：侧面骨架 → 侧面指标 + person 透传 ✓")


def test_assess_without_person():
    refs = load(ROOT / "reference_values.toml")
    report = assess(side_view_frames(), level="novice", refs=refs, label="t.trc")
    assert "person" not in report  # 抢救路径没有选人环节
    print("  assess：无 person 字段时不写入报告 ✓")


def test_assess_quality_hints():
    """质量提示：小人物 + 低置信度进 notice，深蹲占比进 knee_flexion caveat。"""
    refs = load(ROOT / "reference_values.toml")
    frames = side_view_frames()
    frames[:, :, 2] = 0.35  # 全帧低置信度（<0.5）
    frames[:, [L_ANKLE, R_ANKLE], 2] = 0.0  # 踝不参与（坐标 (0,0) 会算出假深蹲）
    report = assess(frames, level="novice", refs=refs, label="t.mp4",
                    quality={"person_height_ratio": 0.12,
                             "low_conf_frame_ratio": 0.9})
    assert "quality" in report
    assert report["quality"]["person_height_ratio"] == 0.12
    assert "偏小" in report["notice"] and "低置信度" in report["notice"], report["notice"]
    # side_view_frames 的骨架直立（膝弯≈180°），不触发深蹲 caveat
    knee = next(m for m in report["metrics"] if m["metric"] == "knee_flexion")
    assert "caveat" not in knee
    print("  assess：小人物+低置信度提示写入 notice ✓")

    # 深蹲帧过半 → knee_flexion 带 caveat
    squat = side_view_frames()
    squat[:, [L_ANKLE, R_ANKLE]] = [[180, 290, 0.9], [181, 290, 0.9]]  # 膝弯 ≈82° <90°
    report2 = assess(squat, level="novice", refs=refs, label="t.mp4")
    knee2 = next(m for m in report2["metrics"] if m["metric"] == "knee_flexion")
    assert "caveat" in knee2, knee2
    assert report2["quality"]["deep_squat_frame_ratio"] >= 0.2
    print("  assess：深蹲相位 caveat 写入膝弯项 ✓")


def frontal_frame() -> np.ndarray:
    """单帧正面骨架：肩线水平、鼻子居中、两侧置信度对称 → frontal 票。"""
    kp = np.zeros((26, 3))
    kp[L_SHOULDER] = [90, 100, 0.9]
    kp[R_SHOULDER] = [120, 100, 0.9]
    kp[L_HIP] = [98, 200, 0.9]
    kp[R_HIP] = [112, 200, 0.9]
    kp[L_KNEE] = [96, 300, 0.9]
    kp[R_KNEE] = [114, 300, 0.9]
    kp[L_ANKLE] = [95, 400, 0.9]
    kp[R_ANKLE] = [115, 400, 0.9]
    kp[L_ELBOW] = [80, 150, 0.85]
    kp[R_ELBOW] = [130, 150, 0.85]
    kp[0] = [105, 60, 0.9]  # NOSE 居中
    return kp


def test_assess_mixed_view():
    """跟拍混合视角：半侧面半正面 → mixed，五指标按帧级门控各自产出。"""
    refs = load(ROOT / "reference_values.toml")
    side = side_view_frames(10)
    front = np.stack([frontal_frame() for _ in range(10)])
    report = assess(np.concatenate([side, front]), level="novice", refs=refs,
                    label="t.mp4")
    assert report["view"]["detected"] == "mixed", report["view"]
    names = [m["metric"] for m in report["metrics"]]
    assert set(names) == {"hip_flexion", "knee_flexion", "trunk_lean",
                          "knee_valgus", "shoulder_hip_align"}, names
    by = {m["metric"]: m for m in report["metrics"]}
    # sagittal 三项只取侧面帧（10），frontal 两项只取正面帧（10）
    assert by["knee_flexion"]["n_valid_frames"] <= 10
    assert by["shoulder_hip_align"]["n_valid_frames"] <= 10
    assert "混合视角" in (report["notice"] or "")
    print("  assess：mixed 视角逐帧门控，五指标各自产出 ✓")


if __name__ == "__main__":
    test_assess_side_view_report()
    test_assess_without_person()
    test_assess_quality_hints()
    test_assess_mixed_view()
    print("全部通过")
