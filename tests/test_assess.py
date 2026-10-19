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
from skibot.skeleton import L_HIP, L_KNEE, L_SHOULDER, R_HIP, R_KNEE, R_SHOULDER  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def side_view_frames(n: int = 20) -> np.ndarray:
    """n 帧直立侧面骨架：肩线近重合、鼻子偏出，分类为 sagittal。"""
    frames = []
    for _ in range(n):
        kp = np.zeros((26, 3))
        kp[L_SHOULDER] = [100, 100, 0.9]
        kp[R_SHOULDER] = [105, 100, 0.9]
        kp[L_HIP] = [110, 200, 0.9]
        kp[R_HIP] = [112, 200, 0.9]
        kp[L_KNEE] = [110, 300, 0.9]
        kp[R_KNEE] = [111, 300, 0.9]
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


if __name__ == "__main__":
    test_assess_side_view_report()
    test_assess_without_person()
    print("全部通过")
