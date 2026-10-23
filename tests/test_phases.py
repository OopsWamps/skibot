"""三维相位切分测试：python tests/test_phases.py

合成侧倾序列覆盖：弯数与边界、50% 相位划分、幅值/时长丢弃、首尾不完整、
低置信弯不进聚合、无合格弯回退、逐帧标签，以及 assess 的 fps 接入。
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows GBK 控制台兜底

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skibot.assess import assess  # noqa: E402
from skibot.phases import (  # noqa: E402
    PHASE_METRICS, aggregate_phases, phase_labels, segment_phases,
)
from skibot.reference import load  # noqa: E402
from skibot.skeleton import (  # noqa: E402
    L_ANKLE, L_HIP, L_KNEE, L_SHOULDER, NOSE,
    R_ANKLE, R_HIP, R_KNEE, R_SHOULDER,
)
from skibot.skeleton3d import MockLifter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _triangular(signal: np.ndarray, start: int, end: int, amplitude: float) -> None:
    peak = (start + end) // 2
    half = peak - start
    for i in range(start, end):
        signal[i] = amplitude * (1.0 - abs(i - peak) / half)


def wave_signal() -> np.ndarray:
    """50 帧 @10fps：头段负 5 帧，两个 2s 的三角弯，尾部零 5 帧。"""
    lean = np.zeros(50)
    lean[:5] = -1.0
    _triangular(lean, 5, 25, 10.0)
    _triangular(lean, 25, 45, -10.0)
    return lean


def knee_signal() -> np.ndarray:
    """膝弯周期：两个正峰对齐侧倾极值，极小对齐换刃。"""
    knee = np.zeros(50)
    _triangular(knee, 5, 25, 20.0)
    _triangular(knee, 25, 45, 20.0)
    knee[46:] = np.linspace(1.0, 4.0, 4)
    return knee


def test_two_turns_and_phases():
    lean = wave_signal()
    seg = segment_phases(lean, fps=10.0, knee=knee_signal(), lateral=lean)
    assert len(seg.turns) == 2, [d.to_dict() for d in seg.discarded]
    t0, t1 = seg.turns
    assert abs(t0.start_frame - 5) <= 1 and abs(t0.end_frame - 25) <= 1
    assert abs(t1.start_frame - 25) <= 1 and abs(t1.end_frame - 45) <= 1
    assert abs(t0.peak_frame - 15) <= 1 and abs(t1.peak_frame - 35) <= 1
    assert abs(t0.peak_lean_deg - 10.0) < 1e-6
    assert abs(t1.peak_lean_deg + 10.0) < 1e-6
    assert abs(t0.duration_s - 2.0) < 1e-6
    assert abs(t0.amplitude_deg - 10.0) < 1e-6
    assert t0.confidence is not None and t0.confidence >= 0.8
    assert (t0.entry.start_frame, t0.entry.end_frame) == (5, 10)
    assert (t0.control.start_frame, t0.control.end_frame) == (10, 21)
    assert (t0.exit.start_frame, t0.exit.end_frame) == (21, 25)
    assert (t1.entry.start_frame, t1.entry.end_frame) == (25, 30)
    assert (t1.control.start_frame, t1.control.end_frame) == (30, 41)
    assert (t1.exit.start_frame, t1.exit.end_frame) == (41, 45)

    reasons = {(d.start_frame, d.end_frame): d.reason for d in seg.discarded}
    assert reasons[(0, 5)] == "incomplete"
    assert reasons[(45, 50)] == "incomplete"

    labels = phase_labels(seg, len(lean))
    assert labels[0] == "" and labels[4] == ""
    assert labels[5] == "entry" and labels[10] == "control"
    assert labels[20] == "control" and labels[21] == "exit"
    assert labels[25] == "entry" and labels[41] == "exit"
    assert labels[45] == ""

    data = seg.to_dict()
    assert data["fallback"] is None and len(data["turns"]) == 2
    assert data["turns"][0]["entry"]["start_frame"] == 5
    assert data["discarded"][0]["reason"] == "incomplete"
    print("  两个弯：边界 / 50% 相位 / 首尾丢弃 / 逐帧标签 / JSON ✓")


def test_amplitude_and_duration_discards():
    low = np.zeros(50)
    low[:5] = -1.0
    _triangular(low, 5, 25, 4.0)
    seg_low = segment_phases(low, fps=10.0)
    assert not seg_low.turns and seg_low.fallback == "no_valid_turns"
    assert any(d.reason == "low_amplitude" for d in seg_low.discarded)

    short = np.zeros(50)
    short[:5] = -1.0
    short[5], short[6], short[7] = 0.0, 10.0, 5.0
    short[8:] = 0.0
    seg_short = segment_phases(short, fps=10.0)
    assert not seg_short.turns and seg_short.fallback == "no_valid_turns"
    assert any(d.reason == "too_short" for d in seg_short.discarded)
    print("  半周期幅值 <6° / 时长 <0.5s：丢弃并给出原因 ✓")


def test_zero_platforms_and_half_frame_boundaries():
    # 首尾都是零平台：两个波瓣都应是完整弯
    lean = np.array([0, 0, 2, 10, 2, 0, 0, -2, -10, -2, 0, 0], dtype=float)
    seg = segment_phases(lean, fps=5.0)
    assert len(seg.turns) == 2, [d.to_dict() for d in seg.discarded]
    assert abs(seg.turns[0].peak_lean_deg - 10.0) < 1e-6
    assert abs(seg.turns[1].peak_lean_deg + 10.0) < 1e-6

    # 一次符号变化只产生一个过零，不应出现 start==end 的 degenerate
    single = segment_phases(np.array([1, 1, 0, -1, -1], dtype=float), fps=10.0)
    assert len(single.discarded) == 1 and single.discarded[0].reason == "no_crossings"
    dup = segment_phases(np.array(
        [8, 9, 10, 0, 0, 0, -7, -10, -8, 0, 0, 8, 10, 9, 8, 7, 6, 0, 0, 0],
        dtype=float), fps=10.0)
    assert len(dup.turns) == 2, [d.to_dict() for d in dup.discarded]
    assert all(d.start_frame != d.end_frame for d in dup.discarded), dup.discarded

    # 换刃恰在半帧：时长按 0.5s 算，不能因 banker's rounding 变成 0.4s
    half = np.array([-5, -5, -5, -2, 2, 8, 8, 3, 2, -2, -5, -5, -5, -5],
                    dtype=float)
    seg_half = segment_phases(half, fps=10.0)
    assert len(seg_half.turns) == 1, [d.to_dict() for d in seg_half.discarded]
    assert abs(seg_half.turns[0].duration_s - 0.5) < 1e-6
    assert seg_half.turns[0].amplitude_deg >= 6.0

    # 极值贴边不再额外判 degenerate
    edge = np.array([-3, -3, -3, -3, -3, 1, 2, 4, 7, 9, -1, -1, -1, -1],
                    dtype=float)
    seg_edge = segment_phases(edge, fps=10.0)
    assert len(seg_edge.turns) == 1, [d.to_dict() for d in seg_edge.discarded]
    assert abs(seg_edge.turns[0].amplitude_deg - 9.0) < 1e-6

    # 换刃 3.4 帧：ceil 应取 4，round 会取 3
    frac = np.array([-5, -5, -5, -2, 3, 8, 7, 5, 2, -2, -5, -6, -5], dtype=float)
    seg_frac = segment_phases(frac, fps=10.0)
    assert len(seg_frac.turns) == 1, [d.to_dict() for d in seg_frac.discarded]
    assert seg_frac.turns[0].start_frame == 4
    assert seg_frac.turns[0].end_frame == 9

    # 三帧前导零平台：起始换刃取最后一个零（2），不是平台中心（1）
    lead = np.array([0, 0, 0, 2, 10, 2, 0, 0, -2, -10, -2, 0, 0], dtype=float)
    seg_lead = segment_phases(lead, fps=5.0)
    assert len(seg_lead.turns) == 2, [d.to_dict() for d in seg_lead.discarded]
    assert seg_lead.turns[0].start_frame == 2

    # 换刃 3.4/8.3 帧：精确时长 0.49s < 0.5；改回帧差会算成 0.5s
    dur = np.array([-5, -5, -5, -2, 3, 8, 7, 5, 3, -7, -5, -5], dtype=float)
    seg_dur = segment_phases(dur, fps=10.0)
    assert not seg_dur.turns
    assert any(d.reason == "too_short" for d in seg_dur.discarded), seg_dur.discarded
    print("  零平台对称 / 重复过零 / 半帧时长 / 极值贴边 / ceil / 精确时长 ✓")


def test_nan_gap_and_detrend():
    # 换刃跨 NaN 缺口：幅值用峰值绝对值，不能按区间 max-min 低估成 5°
    nan_lean = np.array(
        [8, 10, 9, 7, 5, np.nan, np.nan, np.nan, np.nan, -5, -8, -10, -8, -5,
         np.nan, np.nan, np.nan, 5, 8, 10, 8, 5], dtype=float)
    seg_nan = segment_phases(nan_lean, fps=10.0)
    assert len(seg_nan.turns) == 1, [d.to_dict() for d in seg_nan.discarded]
    assert abs(seg_nan.turns[0].amplitude_deg - 10.0) < 1e-6
    assert seg_nan.turns[0].duration_s >= 0.5

    # 3s 长弯 + 横向线性漂移：只在换刃近零帧拟合漂移，不能把弯形减掉
    n = 70
    lean = np.zeros(n)
    lean[:5] = -1.0
    _triangular(lean, 5, 35, 10.0)
    _triangular(lean, 35, 65, -10.0)
    lateral = lean + np.linspace(0.0, 8.0, n)
    seg = segment_phases(lean, fps=10.0, lateral=lateral, knee=np.zeros(n))
    assert len(seg.turns) == 2, [d.to_dict() for d in seg.discarded]
    assert all(t.confidence is not None and t.confidence >= 0.5 for t in seg.turns)
    print("  跨 NaN 幅值 / 3s 长弯横向去趋势 ✓")


def test_confidence_noise_gate():
    n = 70
    lean = np.zeros(n)
    lean[:5] = -1.0
    _triangular(lean, 5, 35, 10.0)
    _triangular(lean, 35, 65, -10.0)
    drift = 320.0 + 0.01 * np.arange(n)

    # 纯线性横向漂移：没有真实振荡，置信度应为 None，弯仍保留
    seg = segment_phases(lean, fps=10.0, lateral=drift, knee=np.zeros(n))
    assert len(seg.turns) == 2, [d.to_dict() for d in seg.discarded]
    assert all(t.confidence is None for t in seg.turns)

    # 平坦信号 + 1e-6 噪声：不能被当成交叉证据
    rng = np.random.default_rng(0)
    noisy = drift + rng.normal(0.0, 1e-6, n)
    seg_noise = segment_phases(lean, fps=10.0, lateral=noisy, knee=np.zeros(n))
    assert len(seg_noise.turns) == 2
    assert all(t.confidence is None for t in seg_noise.turns)

    seg_knee = segment_phases(lean, fps=10.0, knee=rng.normal(0.0, 1e-6, n))
    assert len(seg_knee.turns) == 2
    assert all(t.confidence is None for t in seg_knee.turns)
    print("  交叉信号噪声门槛：纯漂移 / 微噪声 → confidence=None ✓")


def test_long_sequence_detrend():
    n = 310
    lean = np.zeros(n)
    lean[:5] = -1.0
    for k in range(10):
        _triangular(lean, 5 + 30 * k, 35 + 30 * k, 10.0 if k % 2 == 0 else -10.0)
    seg = segment_phases(lean, fps=10.0, lateral=lean, knee=np.zeros(n))
    assert len(seg.turns) == 10, [d.to_dict() for d in seg.discarded]
    assert all(t.confidence is not None and t.confidence >= 0.5 for t in seg.turns)
    print("  10 个 3s 弯：横向去趋势不随弯序号丢弯 ✓")


def test_low_confidence_not_aggregated():
    lean = wave_signal()
    knee = knee_signal()
    good = segment_phases(lean, fps=10.0, knee=knee, lateral=lean)
    assert len(good.turns) == 2

    bad = segment_phases(lean, fps=10.0, knee=np.roll(knee, 5),
                         lateral=np.roll(lean, 10))
    assert not bad.turns
    assert any(d.reason == "low_confidence" for d in bad.discarded)

    series = {m: np.full(50, 100.0) for m in PHASE_METRICS}
    assert aggregate_phases(series, bad) is None
    agg = aggregate_phases(series, good)
    assert agg is not None
    assert agg["hip_flexion"]["entry"]["median_deg"] == 100.0
    assert agg["hip_flexion"]["entry"]["n_frames"] == 10
    assert agg["hip_flexion"]["control"]["n_frames"] == 22
    assert agg["hip_flexion"]["exit"]["n_frames"] == 8
    print("  低置信弯：丢弃原因 + 不进聚合；合格弯按相位聚合成 10/22/8 帧 ✓")


def test_no_crossings_and_validation():
    seg = segment_phases(np.zeros(20), fps=10.0)
    assert not seg.turns and seg.fallback == "no_valid_turns"
    assert seg.discarded[0].reason == "no_crossings"
    assert aggregate_phases({m: np.zeros(20) for m in PHASE_METRICS}, seg) is None

    for bad_call in (
        lambda: segment_phases(np.zeros(10), fps=0.0),
        lambda: segment_phases(np.zeros((10, 1)), fps=10.0),
        lambda: segment_phases(np.zeros(10), fps=10.0, knee=np.zeros(9)),
        lambda: segment_phases(np.zeros(10), fps=10.0, lateral=np.zeros(9)),
    ):
        try:
            bad_call()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")
    print("  无过零回退 / 非法输入报错 ✓")


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


def test_assess_fps_integration():
    refs = load(ROOT / "reference_values.toml")
    kp = side_frames_2d()
    res = assess(kp, level="novice", refs=refs, label="side.mp4",
                 lifter=MockLifter(), fps=10.0)
    assert res.phases is not None
    assert res.phases.fps == 10.0
    assert res.phases.fallback == "no_valid_turns"  # 侧倾恒定，没有过零

    no_fps = assess(kp, level="novice", refs=refs, label="side.mp4",
                    lifter=MockLifter())
    assert no_fps.phases is None
    print("  assess：有 lifter+fps 才切相位，缺 fps 时为 None ✓")


if __name__ == "__main__":
    test_two_turns_and_phases()
    test_amplitude_and_duration_discards()
    test_zero_platforms_and_half_frame_boundaries()
    test_nan_gap_and_detrend()
    test_confidence_noise_gate()
    test_long_sequence_detrend()
    test_low_confidence_not_aggregated()
    test_no_crossings_and_validation()
    test_assess_fps_integration()
    print("全部通过")
