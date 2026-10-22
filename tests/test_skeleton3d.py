"""三维骨架契约与提升适配器测试：python tests/test_skeleton3d.py

覆盖：形状/置信度/重力向量校验、假模型的翻转与居中、缺帧/全零帧/
低置信帧的 valid_mask，以及 assess 接入 lifter 后二维输出逐位不变。
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows GBK 控制台兜底

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skibot.assess import assess  # noqa: E402
from skibot.reference import load  # noqa: E402
from skibot.skeleton import (  # noqa: E402
    L_ANKLE, L_HIP, L_KNEE, L_SHOULDER, NOSE, R_ANKLE, R_HIP, R_KNEE, R_SHOULDER,
)
from skibot.skeleton3d import CONF_MIN_3D, MockLifter, Skeleton3D  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def side_frames(n: int = 20) -> np.ndarray:
    """侧面骨架：两肩近重合、鼻子偏出；核心八点置信度齐全。"""
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


def _expect_value_error(fn) -> None:
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_contract_validation():
    joints = np.zeros((1, 26, 3))
    conf = np.ones((1, 26))
    _expect_value_error(lambda: Skeleton3D(joints[:, :25], conf[:, :25]))
    _expect_value_error(lambda: Skeleton3D(joints, conf[:, :25]))
    _expect_value_error(lambda: Skeleton3D(joints, np.full((1, 26), 1.5)))
    _expect_value_error(lambda: Skeleton3D(joints, np.full((1, 26), np.nan)))
    _expect_value_error(lambda: Skeleton3D(
        joints, conf, gravity_up=np.array([0.0, 2.0, 0.0])))
    print("  契约校验：形状 / 置信度 / 重力向量 ✓")


def test_mock_lifter():
    kp = side_frames(2)
    sk = MockLifter().lift(kp)
    assert sk.joints.shape == (2, 26, 3)
    assert sk.confidence.shape == (2, 26)
    assert sk.source == "mock" and sk.meta["adapter"] == "MockLifter"
    assert sk.n_frames == 2
    assert sk.valid_mask.all() and sk.root_defined.all()

    # root-relative：髋中点在原点
    mid = (sk.joints[:, L_HIP] + sk.joints[:, R_HIP]) / 2.0
    assert np.allclose(mid, 0.0)
    # 图像 y 向下翻转：肩在髋上方（图像 y 更小）→ 三维 y 更大
    assert sk.joints[0, L_SHOULDER, 1] > sk.joints[0, L_HIP, 1]

    # 缺失关节：坐标 NaN、置信度 0；核心点仍够，帧仍有效
    kp2 = kp.copy()
    kp2[:, L_ANKLE] = [0.0, 0.0, 0.0]
    sk2 = MockLifter().lift(kp2)
    assert np.isnan(sk2.joints[:, L_ANKLE]).all()
    assert (sk2.confidence[:, L_ANKLE] == 0).all()
    assert sk2.valid_mask.all()
    print("  假模型：形状 / 翻转 / 居中 / 缺测 ✓")


def test_missing_and_zero_frames():
    kp = side_frames(3)
    kp[1] = 0.0                 # 全零帧
    kp[2, :, 2] = 0.1           # 低置信帧
    sk = MockLifter().lift(kp)
    assert sk.valid_mask[0]
    assert not sk.valid_mask[1]  # 全零帧按缺帧处理
    assert not sk.valid_mask[2]  # 置信度低于阈值
    assert not sk.root_defined[1]
    assert np.isfinite(sk.joints[2, L_HIP]).all()  # 坐标还在
    assert sk.confidence[2, L_HIP] < CONF_MIN_3D
    print("  缺帧 / 全零帧 / 低置信帧：valid_mask 行为 ✓")


def test_assess_integration():
    refs = load(ROOT / "reference_values.toml")
    kp = side_frames(20)
    res2d = assess(kp, level="novice", refs=refs, label="t.mp4")
    res3d = assess(kp, level="novice", refs=refs, label="t.mp4", lifter=MockLifter())

    assert res2d.skeleton3d is None
    assert res3d.skeleton3d is not None
    assert res3d.skeleton3d.n_frames == kp.shape[0]
    # 二维报告与序列逐位不变
    assert res3d.report["metrics"] == res2d.report["metrics"]
    assert res3d.report["schema"] == res2d.report["schema"]
    for key in res2d.series:
        assert np.array_equal(res2d.series[key], res3d.series[key],
                              equal_nan=True), key
    assert np.array_equal(res2d.frame_view, res3d.frame_view)

    class BadLifter:
        name = "bad"

        def lift(self, keypoints):
            return MockLifter().lift(keypoints[:1])

    _expect_value_error(lambda: assess(
        kp, level="novice", refs=refs, label="t.mp4", lifter=BadLifter()))
    print("  assess：可选 lifter 端到端，二维输出不变 ✓")


if __name__ == "__main__":
    test_contract_validation()
    test_mock_lifter()
    test_missing_and_zero_frames()
    test_assess_integration()
    print("全部通过")
