"""上传预检测试：python tests/test_precheck.py

合成 mp4（cv2.VideoWriter）+ 注入人物检测替身，不依赖真视频与真检测；
人物占比的拒绝/放行/无法判定三分支都有用例钉住。
"""

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skibot.precheck import PrecheckConfig, run_precheck  # noqa: E402


def make_video(path: Path, size=(1280, 720), fps=25, n=88, value=128,
               ramp=False) -> Path:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert w.isOpened(), f"VideoWriter 打不开: {path}"
    for i in range(n):
        v = min(i * 2, 255) if ramp else value
        w.write(np.full((size[1], size[0], 3), v, np.uint8))
    w.release()
    return path


class FakeHog:
    """替身 HOG 检测器：记录收到的帧尺寸，返回预置矩形。"""

    def __init__(self, rects):
        self.rects = rects
        self.shapes: list[tuple[int, int]] = []

    def detectMultiScale(self, frame, **kw):
        self.shapes.append(frame.shape[:2])
        return self.rects, None


class _Patched:
    """临时替换模块属性，退出时还原（测试脚本不依赖 pytest fixture）。"""

    def __init__(self, **attrs):
        self.attrs = attrs
        self.saved = {}

    def __enter__(self):
        import skibot.precheck as P
        for k, v in self.attrs.items():
            self.saved[k] = getattr(P, k)
            setattr(P, k, v)
        return P

    def __exit__(self, *exc):
        import skibot.precheck as P
        for k, v in self.saved.items():
            setattr(P, k, v)
        return False


def test_default_probe_scaling():
    """内置 HOG 路径本身：720p 原生检测，只有超 1920 才缩放。"""
    import skibot.precheck as P

    fake = FakeHog([(0, 0, 50, 200)])
    with _Patched(_hog=lambda: fake):
        r = P._default_probe(np.zeros((720, 1280, 3), np.uint8))
        assert fake.shapes[-1] == (720, 1280), fake.shapes[-1]
        assert abs(r[0] - 200 / 720) < 1e-9, r

        # 4K 竖拍：长边 4000 → 缩到 1920，比例对缩放不变
        r = P._default_probe(np.zeros((4000, 3000, 3), np.uint8))
        assert fake.shapes[-1] == (1920, 1440), fake.shapes[-1]
        assert abs(r[0] - 200 / 1920) < 1e-9, r

    print("  预检：内置 HOG 原生检测 / 超长边缩放 ✓")


def test_default_probe_end_to_end(tmp: Path):
    """不注入替身，只替换 _hog：小人物拒绝、大人物放行。"""
    video = make_video(tmp / "e2e.mp4")
    with _Patched(_hog=lambda: FakeHog([(0, 0, 50, 40)])):  # 40/720 ≈ 5.6%
        r = run_precheck(video)
    assert not r.passed and r.issues[0].check == "person", _problems(r)

    with _Patched(_hog=lambda: FakeHog([(0, 0, 50, 200)])):  # 200/720 ≈ 28%
        r = run_precheck(video)
    assert r.passed, _problems(r)

    print("  预检：内置 HOG 端到端（小人物拒绝 / 大人物放行）✓")


def _problems(r):
    return [i.problem for i in r.issues]


def test_container_and_metadata(tmp: Path):
    # 合格素材：720p、3.5s、灰帧。HOG 检不出人物 → 无法判定放行
    r = run_precheck(make_video(tmp / "good.mp4"))
    assert r.passed, _problems(r)
    assert r.person_ratio is None and "未检出人物" in r.notes[0]

    # 容器/编码：垃圾字节直接拒绝，不进抽帧
    bad = tmp / "bad.mp4"
    bad.write_bytes(b"\x00\x01garbage-not-a-video")
    r = run_precheck(bad)
    assert not r.passed and len(r.issues) == 1
    assert r.issues[0].check == "container" and "无法解码" in r.issues[0].problem
    assert "H.264" in r.issues[0].advice

    # 文件不存在：基础设施问题，不给"转码"建议
    r = run_precheck(tmp / "missing.mp4")
    assert not r.passed and r.issues[0].check == "file"
    assert "不存在" in r.issues[0].problem

    print("  预检：容器/元数据（合格放行 / 垃圾字节拒绝 / 文件缺失）✓")


def test_duration_and_resolution(tmp: Path):
    # 时长过短：720p 1s < 3s，分辨率本身合格 → 只有 duration 一条
    called = []
    r = run_precheck(make_video(tmp / "short.mp4", n=25),
                     person_probe=lambda f: called.append(1) or [0.5])
    assert not r.passed and [i.check for i in r.issues] == ["duration"]
    assert "太短" in r.issues[0].problem and "换刃" in r.issues[0].advice
    assert not called  # 元数据已判失败就不抽帧（节省上传路径耗时）

    # 多重不合格：320x240、1s 同时命中分辨率与时长，两条原因都给出
    r = run_precheck(make_video(tmp / "multi.mp4", size=(320, 240), n=25))
    assert {i.check for i in r.issues} == {"resolution", "duration"}
    assert all(i.advice for i in r.issues)

    # 时长过长：用判据上限压到 2s，3.5s 素材超限；建议里给降采样命令
    r = run_precheck(make_video(tmp / "long.mp4"),
                     PrecheckConfig(min_duration_s=1.0, max_duration_s=2.0))
    assert not r.passed and [i.check for i in r.issues] == ["duration"]
    assert "超过" in r.issues[0].problem and "fps=15" in r.issues[0].advice

    # 分辨率过低：320x240 但时长 3.6s 合格 → 只有 resolution 一条
    r = run_precheck(make_video(tmp / "small.mp4", size=(320, 240), n=90))
    assert not r.passed and [i.check for i in r.issues] == ["resolution"]
    assert "320x240" in r.issues[0].problem and "720p" in r.issues[0].advice

    # 判据可配置：min_duration_s=0.5 时 1s 素材通过
    r = run_precheck(make_video(tmp / "ok1s.mp4", n=25),
                     PrecheckConfig(min_duration_s=0.5))
    assert r.passed, _problems(r)

    # 分辨率判据也可配置：320x240 在更低门槛下放行
    r = run_precheck(make_video(tmp / "small_ok.mp4", size=(320, 240), n=90),
                     PrecheckConfig(min_short_side=200, min_long_side=300))
    assert r.passed, _problems(r)

    # 帧数上限独立于时长：3.5s@50fps=175 帧在 max_frames=100 下被拒
    r = run_precheck(make_video(tmp / "many.mp4", fps=50, n=175),
                     PrecheckConfig(max_frames=100))
    assert not r.passed and [i.check for i in r.issues] == ["duration"]
    assert "帧" in r.issues[0].problem

    # n_probe_frames 生效：=1 时抽帧检测恰好调用一次
    calls = []
    r = run_precheck(make_video(tmp / "one_probe.mp4"),
                     PrecheckConfig(n_probe_frames=1),
                     person_probe=lambda f: calls.append(1) or [0.5])
    assert r.passed and len(calls) == 1, (r, len(calls))

    print("  预检：时长上下限 / 分辨率下限 / 判据可配置 / 抽帧次数 ✓")


def test_probe_failures(tmp: Path):
    """不可解码拒绝；只能顺序读、不能 seek 的容器按无法判定放行。"""
    video = make_video(tmp / "probe.mp4")

    class FakeCap:
        """VideoCapture 替身。can_read 决定 read 是否成功；meta 覆盖无元数据。"""

        def __init__(self, can_read, meta=True, seek_only=False):
            self.can_read = can_read
            self.meta = meta
            self.seek_only = seek_only
            self.pos = 0
            self.reads = 0

        def isOpened(self):
            return True

        def get(self, prop):
            # meta=False 模拟 VFR/索引缺失：宽高可读，帧率与帧数为 0
            if not self.meta and prop in (cv2.CAP_PROP_FPS,
                                          cv2.CAP_PROP_FRAME_COUNT):
                return 0.0
            return {cv2.CAP_PROP_FRAME_WIDTH: 1280.0,
                    cv2.CAP_PROP_FRAME_HEIGHT: 720.0,
                    cv2.CAP_PROP_FPS: 25.0,
                    cv2.CAP_PROP_FRAME_COUNT: 88.0}.get(prop, 0.0)

        def set(self, prop, value):
            self.pos = value
            return True

        def read(self):
            # 一帧都解不出来：文件损坏/编码不支持
            if not self.can_read:
                return False, None
            # 只能从第 0 帧顺序读（容器不支持按帧 seek）
            if self.seek_only and self.pos != 0:
                return False, None
            self.pos += 1
            self.reads += 1
            return True, np.zeros((720, 1280, 3), np.uint8)

        def release(self):
            pass

    import skibot.precheck as P

    saved = P.cv2.VideoCapture
    try:
        P.cv2.VideoCapture = lambda *a, **k: FakeCap(False)  # type: ignore[assignment]
        r = run_precheck(video)
        assert not r.passed and r.issues[0].check == "container", r
        assert "无法解码" in r.issues[0].problem and r.issues[0].advice

        P.cv2.VideoCapture = lambda *a, **k: FakeCap(True, seek_only=True)  # type: ignore[assignment]
        r = run_precheck(video, person_probe=lambda f: [0.3])
        assert r.passed and r.person_ratio is None, r
        assert r.notes and r.notes[0].startswith("抽帧定位失败"), r.notes

        # 帧率/帧数读不出（VFR/索引缺失）：跳过时长与人物占比判据放行；
        # 不抽帧——无法确定采样位置，从片头抽样会把长素材误判成人物太小
        P.cv2.VideoCapture = lambda *a, **k: FakeCap(True, meta=False)  # type: ignore[assignment]
        seen = []
        r = run_precheck(video, person_probe=lambda f: seen.append(1) or [0.3])
        assert r.passed and r.person_ratio is None, r
        assert any("无法读取帧率" in n for n in r.notes), r.notes
        assert any("无法确定视频帧数" in n for n in r.notes), r.notes
        assert not seen

        # 能读帧但不能按帧定位（set 返回 False）：不能拿片头 5 帧当样本，
        # 按“无法判定”放行，探针一次都不该被调用
        class NoSeekCap(FakeCap):
            def set(self, prop, value):
                return False

        P.cv2.VideoCapture = lambda *a, **k: NoSeekCap(True)  # type: ignore[assignment]
        seen = []
        r = run_precheck(video, person_probe=lambda f: seen.append(1) or [0.05])
        assert r.passed and r.person_ratio is None, r
        assert r.notes and r.notes[0].startswith("抽帧定位失败"), r.notes
        assert not seen

        # 帧率读不出但帧数可读：帧数上限仍然生效
        class VfrCap(FakeCap):
            def get(self, prop):
                if prop == cv2.CAP_PROP_FPS:
                    return 0.0
                if prop == cv2.CAP_PROP_FRAME_COUNT:
                    return 20000.0
                return super().get(prop)

        P.cv2.VideoCapture = lambda *a, **k: VfrCap(True)  # type: ignore[assignment]
        r = run_precheck(video, person_probe=lambda f: [0.3])
        assert not r.passed and "20000" in r.issues[0].problem, r
        assert "帧" in r.issues[0].problem
    finally:
        P.cv2.VideoCapture = saved  # type: ignore[assignment]

    # 采样位置：亮度递增视频，注入记录帧均值的探针
    ramp = make_video(tmp / "ramp.mp4", n=88, ramp=True)
    means = []

    def mean_probe(frame):
        means.append(float(frame.mean()))
        return [0.5]

    r = run_precheck(ramp, person_probe=mean_probe)
    assert r.passed and len(means) == 5, len(means)
    assert means == sorted(means), means          # 采样点按时间递增
    assert means[0] > 5 and means[-1] < 200, means  # 落在 10%~90%，不碰首尾

    print("  预检：不可解码拒绝 / 不能 seek 按无法判定放行 / 采样点 10%~90% ✓")


def test_person_probe(tmp: Path):
    video = make_video(tmp / "person.mp4")

    # 确凿检出且全部过小 → 拒绝 + 重拍建议
    r = run_precheck(video, person_probe=lambda f: [0.05, 0.06])
    assert not r.passed and [i.check for i in r.issues] == ["person"]
    assert "6%" in r.issues[0].problem and "1/3" in r.issues[0].advice
    assert abs(r.person_ratio - 0.06) < 1e-9  # 取最大检出

    # 检出够大 → 放行，不写放行说明
    r = run_precheck(video, person_probe=lambda f: [0.3])
    assert r.passed and r.person_ratio == 0.3 and not r.notes

    # 检测器抛异常 → 无法判定放行，不误杀
    def boom(frame):
        raise RuntimeError("detector down")

    r = run_precheck(video, person_probe=boom)
    assert r.passed and r.person_ratio is None
    assert "检测器异常" in r.notes[0]  # 和"漏检"分开写，坏掉的检测器不伪装

    # 阈值可配置：0.05 的检出在 min_person_ratio=0.04 下放行
    r = run_precheck(video, PrecheckConfig(min_person_ratio=0.04),
                     person_probe=lambda f: [0.05])
    assert r.passed, _problems(r)

    # 默认阈值边界：0.17 拒绝、0.19 放行
    assert not run_precheck(video, person_probe=lambda f: [0.17]).passed
    assert run_precheck(video, person_probe=lambda f: [0.19]).passed

    print("  预检：人物占比（过小拒绝 / 够大放行 / 检测失败不误杀 / 阈值边界）✓")


def test_invalid_config():
    """判据配错直接报错，不静默关掉检查。"""
    for kw in ({"min_person_ratio": -0.1},
               {"min_person_ratio": float("nan")},
               {"n_probe_frames": 0},
               {"min_duration_s": 0},
               {"max_frames": 0},
               {"min_short_side": -1}):
        try:
            PrecheckConfig(**kw)
            raise AssertionError(f"应拒绝 {kw}")
        except ValueError:
            pass
    print("  预检：非法判据报错 ✓")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        test_container_and_metadata(Path(td))
        test_duration_and_resolution(Path(td))
        test_person_probe(Path(td))
        test_invalid_config()
        test_default_probe_scaling()
        with tempfile.TemporaryDirectory() as td2:
            test_default_probe_end_to_end(Path(td2))
        with tempfile.TemporaryDirectory() as td3:
            test_probe_failures(Path(td3))
    print("全部通过")
