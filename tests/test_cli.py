"""CLI 管线层测试：python tests/test_cli.py

复用同一结果目录重跑时，analyze_one 只认本次运行产出的轨迹——旧 trc 在
Sports2D 失败后留在原地，不能被 discover_tracks 捡来当本次结果（产出基于
旧关键点的"成功"报告），也不允许被销毁（旧轨迹还是旧报告改选与预览的
数据源）。run_sports2d 用替身替换，不真跑 Sports2D。
"""

import os
import sys
import tempfile
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows GBK 控制台兜底

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import skibot.cli as cli_mod  # noqa: E402
from skibot.cli import AnalyzeOptions, analyze_one  # noqa: E402


def test_analyze_one_rejects_stale_trc(tmp: Path):
    """重跑同一目录且本次没有新轨迹：拒绝采纳旧 trc，同时不销毁它。"""
    video = tmp / "clip.mp4"
    video.write_bytes(b"fake")  # 内容不重要：run_sports2d 是替身
    opts = AnalyzeOptions(blur=False, result_dir=tmp / "results")

    work = tmp / "results" / "clip"
    work.mkdir(parents=True)
    stale = work / "clip_px_person00.trc"
    stale.write_text("stale", encoding="utf-8")
    yesterday = time.time() - 3600
    os.utime(stale, (yesterday, yesterday))  # 旧 mtime：不是本次产物

    called = {}

    def fake_sports2d(src, result_dir, track_order):
        called["ran"] = True

    orig = cli_mod.run_sports2d
    cli_mod.run_sports2d = fake_sports2d
    try:
        try:
            analyze_one(video, opts)
            raise AssertionError("没有新轨迹应失败")
        except FileNotFoundError as e:
            assert "没有产出新轨迹" in str(e)
    finally:
        cli_mod.run_sports2d = orig

    assert called.get("ran")
    assert stale.exists()  # 旧轨迹保留：旧报告的改选与预览仍有数据源
    print("  analyze_one：拒绝陈旧轨迹且不销毁 ✓")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        test_analyze_one_rejects_stale_trc(Path(td))
    print("全部通过")
