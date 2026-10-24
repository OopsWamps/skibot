"""评估收尾测试：python tests/test_finalize.py

assess_and_publish 是报告与标注视频的唯一发布点：成对原子替换、渲染失败
保持旧产物、只落报告的抢救模式。用合成侧视骨架 + 小视频，不跑 Sports2D。
"""

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows GBK 控制台兜底

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skibot.finalize import assess_and_publish  # noqa: E402
from skibot.reference import load  # noqa: E402
from skibot.skeleton import (  # noqa: E402
    L_HIP, L_KNEE, L_SHOULDER, NOSE, R_HIP, R_KNEE, R_SHOULDER,
)

ROOT = Path(__file__).resolve().parent.parent
REFS = load(ROOT / "reference_values.toml")


def side_kp(n: int = 12) -> np.ndarray:
    """n 帧直立侧面骨架（视角 sagittal）。"""
    frames = []
    for _ in range(n):
        kp = np.zeros((26, 3))
        kp[L_SHOULDER] = [10, 10, 0.9]
        kp[R_SHOULDER] = [11, 10, 0.9]
        kp[L_HIP] = [11, 20, 0.9]
        kp[R_HIP] = [12, 20, 0.9]
        kp[L_KNEE] = [11, 30, 0.9]
        kp[R_KNEE] = [12, 30, 0.9]
        kp[NOSE] = [14, 6, 0.9]
        frames.append(kp)
    return np.stack(frames)


def tiny_video(path: Path, n: int = 12) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    assert w.isOpened()
    for _ in range(n):
        w.write(np.full((48, 64, 3), 40, np.uint8))
    w.release()
    return path


def test_publish_report_and_video(tmp: Path):
    """发布后报告与标注视频就位，临时文件清理干净。"""
    tmp = Path(tempfile.mkdtemp(prefix="skibot_test_finalize_"))
    src = tiny_video(tmp / "src.mp4")
    rp, vp = tmp / "report.json", tmp / "v_annotated.mp4"

    res = assess_and_publish(side_kp(), src, refs=REFS, level="novice",
                             label="t.mp4", report_path=rp, video_path=vp)
    assert rp.exists() and rp.stat().st_size > 0
    assert vp.exists() and vp.stat().st_size > 0
    assert res.report["view"]["detected"] == "sagittal"
    assert not list(tmp.glob("*.tmp")) and not list(tmp.glob("*.bak"))
    print("  finalize：报告 + 标注视频成对发布 ✓")


def test_report_only_mode(tmp: Path):
    """video_path=None（抢救路径）：只落报告，不渲染。"""
    tmp = Path(tempfile.mkdtemp(prefix="skibot_test_finalize_ro_"))
    rp = tmp / "report.json"
    assess_and_publish(side_kp(), None, refs=REFS, level="intermediate",
                       label="t.trc", report_path=rp)
    assert rp.exists()
    assert not any(tmp.glob("*.mp4"))
    print("  finalize：只落报告的抢救模式 ✓")


def test_render_failure_keeps_old_pair(tmp: Path):
    """渲染失败：旧报告与旧视频原样保留，临时文件清干净。"""
    import skibot.finalize as F

    tmp = Path(tempfile.mkdtemp(prefix="skibot_test_finalize_fail_"))
    src = tiny_video(tmp / "src.mp4")
    rp, vp = tmp / "report.json", tmp / "v_annotated.mp4"
    assess_and_publish(side_kp(), src, refs=REFS, level="novice",
                       label="t.mp4", report_path=rp, video_path=vp)
    old_report, old_video = rp.read_bytes(), vp.read_bytes()

    def boom(*a, **k):
        raise RuntimeError("渲染挂了")

    orig = F.render_annotated
    F.render_annotated = boom
    try:
        try:
            assess_and_publish(side_kp(), src, refs=REFS, level="novice",
                               label="t.mp4", report_path=rp, video_path=vp)
            raise AssertionError("渲染失败应抛出")
        except RuntimeError:
            pass
    finally:
        F.render_annotated = orig

    assert rp.read_bytes() == old_report and vp.read_bytes() == old_video
    assert not list(tmp.glob("*.tmp")) and not list(tmp.glob("*.bak"))
    print("  finalize：渲染失败保持旧产物对 ✓")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        test_publish_report_and_video(t / "publish")
        test_report_only_mode(t / "reportonly")
        test_render_failure_keeps_old_pair(t / "fail")
    print("全部通过")
