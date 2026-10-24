"""标注视频渲染测试：python tests/test_render.py
cv2 写 6 帧合成 mp4 → assess → render_annotated，断言产物存在；
probe_video 的元数据探测一并对账。
"""

import sys
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows GBK 控制台兜底

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skibot.assess import assess  # noqa: E402
from skibot.reference import load  # noqa: E402
from skibot.render import probe_video, render_annotated  # noqa: E402
from skibot.skeleton import (  # noqa: E402
    L_ANKLE, L_ELBOW, L_HIP, L_KNEE, L_SHOULDER, NOSE, R_ANKLE, R_ELBOW, R_HIP,
    R_KNEE, R_SHOULDER,
)

ROOT = Path(__file__).resolve().parent.parent


def tiny_side_kp(n: int = 12) -> np.ndarray:
    """坐标缩进 64x48 画面的侧面骨架（视角 'l'，核心点 conf 齐）。"""
    frames = []
    for _ in range(n):
        kp = np.zeros((26, 3))
        kp[L_SHOULDER] = [10, 10, 0.9]
        kp[R_SHOULDER] = [11, 10, 0.9]
        kp[L_HIP] = [11, 20, 0.9]
        kp[R_HIP] = [12, 20, 0.9]
        kp[L_KNEE] = [11, 30, 0.9]
        kp[R_KNEE] = [12, 30, 0.9]
        kp[L_ANKLE] = [11, 40, 0.9]
        kp[R_ANKLE] = [12, 40, 0.9]
        kp[NOSE] = [14, 6, 0.9]
        frames.append(kp)
    return np.stack(frames)


def frontal_frame() -> np.ndarray:
    """单帧正面骨架（同 test_assess 的构造）：肩线水平、鼻子居中、两侧对称。"""
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
    kp[NOSE] = [105, 60, 0.9]
    return kp


def _write_tiny_video(path: Path, n: int = 6, w: int = 64, h: int = 48) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    assert writer.isOpened()
    for _ in range(n):
        writer.write(np.full((h, w, 3), 40, dtype=np.uint8))
    writer.release()


def test_probe_and_render():
    tmp = Path(tempfile.mkdtemp(prefix="skibot_test_render_"))
    video = tmp / "tiny.mp4"
    w, h = 64, 48
    _write_tiny_video(video, n=12, w=w, h=h)

    assert probe_video(video) == (w, h, 30.0)

    kp = tiny_side_kp()
    refs = load(ROOT / "reference_values.toml")
    res = assess(kp, level="novice", refs=refs, label="tiny.mp4")
    assert res.report["view"]["detected"] == "sagittal"

    out = tmp / "anno.mp4"
    render_annotated(video, kp, res, out)
    assert out.exists() and out.stat().st_size > 0
    print("  probe_video / render_annotated 产物 ✓")


def test_render_mixed():
    """mixed 评估结果走渲染：默认按判定与显式 view="mixed" 都能出全指标叠加。"""
    tmp = Path(tempfile.mkdtemp(prefix="skibot_test_render_mixed_"))
    video = tmp / "tiny.mp4"
    _write_tiny_video(video)

    kp = np.concatenate([tiny_side_kp(10), np.stack([frontal_frame() for _ in range(10)])])
    refs = load(ROOT / "reference_values.toml")
    res = assess(kp, level="novice", refs=refs, label="mixed.mp4")
    assert res.report["view"]["detected"] == "mixed"

    out_default = tmp / "anno_default.mp4"
    render_annotated(video, kp, res, out_default)
    assert out_default.exists() and out_default.stat().st_size > 0

    out_explicit = tmp / "anno_explicit.mp4"
    render_annotated(video, kp, res, out_explicit, view="mixed")
    assert out_explicit.exists() and out_explicit.stat().st_size > 0
    print("  render mixed 叠加（默认判定 + 显式指定）产物 ✓")


if __name__ == "__main__":
    test_probe_and_render()
    test_render_mixed()
    print("全部通过")
