"""零依赖冒烟测试：python tests/test_metrics.py
用合成骨架验证角度数学与视角分类，不依赖 sports2d/opencv。
TODO: 引入真实滑雪视频后，补关键点抽查与角度精度验证（docs/DECISIONS.md 第 10 条）。
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows GBK 控制台兜底

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skibot import metrics as M  # noqa: E402
from skibot.reference import load, lookup, pct_in_range  # noqa: E402
from skibot.report import build_report  # noqa: E402
from skibot.view import ViewResult, classify_view  # noqa: E402


def kp(pts: dict[int, tuple[float, float]], conf: float = 0.9) -> np.ndarray:
    """(26,4) 单帧骨架：{Halpe索引: (x,y)}，未给的点 conf=0。"""
    arr = np.zeros((26, 4))
    for idx, (x, y) in pts.items():
        arr[idx] = [x, y, conf, 1.0]
    return arr


def test_angle_math():
    f = kp({5: (110, 100), 11: (110, 200), 13: (110, 300), 15: (110, 400),
            6: (120, 100), 12: (120, 200), 14: (120, 300), 16: (120, 400)})
    s = M.compute_series(f[None])
    assert abs(s["hip_flexion"][0] - 180.0) < 1e-6, s["hip_flexion"]
    assert abs(s["knee_flexion"][0] - 180.0) < 1e-6
    assert abs(s["trunk_lean"][0] - 0.0) < 1e-6
    assert abs(s["knee_valgus"][0] - 180.0) < 1e-6
    assert abs(s["shoulder_hip_align"][0] - 0.0) < 1e-6

    # 膝盖向内推 20px → valgus < 180；关键点置信度过低 → NaN
    f2 = kp({5: (110, 100), 11: (110, 200), 13: (130, 300), 15: (110, 400),
             6: (120, 100), 12: (120, 200), 14: (120, 300), 16: (120, 400)})
    s2 = M.compute_series(f2[None])
    assert s2["knee_valgus"][0] < 175.0
    # 只有左侧骨架且左髋置信度过低 → 换边无可换，hip_flexion 应为 NaN
    f3 = kp({5: (110, 100), 11: (110, 200), 13: (110, 300), 15: (110, 400)})
    f3[M.L_HIP, 2] = 0.1
    s3 = M.compute_series(f3[None])
    assert np.isnan(s3["hip_flexion"][0])
    print("  角度数学 ✓")


def test_view_classification():
    frames = []
    for _ in range(20):  # 正面：肩线水平、鼻子居中、左右置信度对称
        frames.append(kp({0: (150, 60), 5: (100, 100), 6: (200, 100),
                          7: (80, 160), 8: (220, 160),
                          11: (110, 200), 12: (190, 200),
                          13: (105, 300), 14: (195, 300),
                          15: (100, 400), 16: (200, 400)}))
    vr = classify_view(np.stack(frames))
    assert vr.view == "frontal", vr

    frames = []
    for _ in range(20):  # 侧面：两肩几乎重合、鼻子偏出、右侧置信度低
        f = kp({0: (120, 60), 5: (100, 100), 6: (105, 100),
                7: (95, 160), 8: (100, 160),
                11: (110, 200), 12: (112, 200),
                13: (108, 300), 14: (110, 300),
                15: (105, 400), 16: (108, 400)})
        f[6, 2] = f[8, 2] = f[12, 2] = f[14, 2] = f[16, 2] = 0.35
        frames.append(f)
    vr = classify_view(np.stack(frames))
    assert vr.view == "sagittal", vr
    print("  视角分类 ✓")


def test_reference_and_report():
    refs = load(Path(__file__).resolve().parent.parent / "reference_values.toml")
    ref = lookup(refs, "hip_flexion", "advanced")
    s = np.full(50, 150.0)
    assert pct_in_range(s, ref) == 1.0
    assert np.isnan(pct_in_range(np.full(10, np.nan), ref))

    vr = ViewResult("sagittal", 0.8, 50, {})
    report = build_report("test.mp4", vr,
                          {"hip_flexion": s, "knee_flexion": s, "trunk_lean": np.full(50, 12.0),
                           "knee_valgus": s, "shoulder_hip_align": s},
                          refs, "novice")
    assert report["calibrated"] is False and report["notice"]
    names = [m["metric"] for m in report["metrics"]]
    assert names == ["hip_flexion", "knee_flexion", "trunk_lean"], names  # 侧面不出正面指标
    assert report["metrics"][0]["pct_in_range"] == 1.0
    print("  参考区间与报告 ✓")


if __name__ == "__main__":
    test_angle_math()
    test_view_classification()
    test_reference_and_report()
    print("全部通过")
