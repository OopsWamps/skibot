"""MotionBERT 提升适配器测试：python tests/test_motionbert.py

映射表按官方 dataset_wild.halpe2h36m 逐项核对；归一化复刻 crop_scale
ratio=1；lift 用注入的假推理函数覆盖切窗与契约（不加载 torch）；
真实模型集成用例在权重与 torch 都在位时才跑（端到端冒烟）。
"""

import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows GBK 控制台兜底

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skibot.assess import assess  # noqa: E402
from skibot.motionbert import (  # noqa: E402
    _HALPE_FROM_H36M, _normalize_bbox, halpe_to_h36m)
from skibot.reference import load as load_refs  # noqa: E402
from skibot.skeleton import (  # noqa: E402
    HEAD_TOP, HIP_CENTER, L_ANKLE, L_HIP, L_KNEE, L_SHOULDER, NECK, NOSE,
    R_ANKLE, R_HIP, R_KNEE, R_SHOULDER)

ROOT = Path(__file__).resolve().parent.parent
REFS = load_refs(ROOT / "reference_values.toml")

# Halpe-26 里 20-25 没有命名常量：左右大脚趾/小脚趾/脚跟
UNMAPPED = (1, 2, 3, 4, 20, 21, 22, 23, 24, 25)  # 眼耳与脚趾脚跟：H36M-17 无对应


def make_kp2d(t: int = 300, conf: float = 0.9) -> np.ndarray:
    """站立人物的 Halpe-26 像素坐标序列，轻微侧向摆动。"""
    w, h = 1280.0, 720.0
    cx = w / 2
    pose = {
        NOSE: (cx, h * 0.12), HEAD_TOP: (cx, h * 0.08), NECK: (cx, h * 0.26),
        L_SHOULDER: (cx - 45, h * 0.28), R_SHOULDER: (cx + 45, h * 0.28),
        7: (cx - 70, h * 0.42), 8: (cx + 70, h * 0.42),      # 肘
        9: (cx - 85, h * 0.55), 10: (cx + 85, h * 0.55),     # 腕
        L_HIP: (cx - 30, h * 0.60), R_HIP: (cx + 30, h * 0.60),
        HIP_CENTER: (cx, h * 0.60),
        L_KNEE: (cx - 32, h * 0.78), R_KNEE: (cx + 32, h * 0.78),
        L_ANKLE: (cx - 34, h * 0.94), R_ANKLE: (cx + 34, h * 0.94),
        20: (cx - 26, h * 0.97), 21: (cx + 26, h * 0.97),    # 大脚趾
        22: (cx - 40, h * 0.96), 23: (cx + 40, h * 0.96),    # 小脚趾
        24: (cx - 44, h * 0.94), 25: (cx + 44, h * 0.94),    # 脚跟
    }
    kp = np.zeros((t, 26, 3))
    for idx, (x, y) in pose.items():
        kp[:, idx, 0] = x
        kp[:, idx, 1] = y
    kp[:, :, 2] = conf
    kp[:, :, 0] += 10.0 * np.sin(np.arange(t) * 2 * np.pi / 60)[:, None]
    return kp


class Recording:
    """假推理函数：记下每窗形状，输出确定值。"""

    def __init__(self):
        self.shapes: list[tuple[int, ...]] = []

    def __call__(self, motion: np.ndarray) -> np.ndarray:
        self.shapes.append(motion.shape)
        f = motion.shape[0]
        return (np.arange(f * 17 * 3, dtype=np.float32) / 1000.0).reshape(f, 17, 3)


def test_halpe_to_h36m_mapping():
    kp = np.stack([np.arange(26 * 3, dtype=float).reshape(26, 3)])  # (1,26,3) 逐点不同
    out = halpe_to_h36m(kp)
    expect = {0: 19, 1: 12, 2: 14, 3: 16, 4: 11, 5: 13, 6: 15,
              8: 18, 9: 0, 10: 17, 11: 5, 12: 7, 13: 9,
              14: 6, 15: 8, 16: 10}
    for j36, src in expect.items():
        assert np.array_equal(out[0, j36], kp[0, src]), f"h36m {j36} 应来自 halpe {src}"
    # 脊点 = 颈(18) 与髋中(19) 的中点，坐标与置信通道同公式
    assert np.allclose(out[0, 7], (kp[0, 18] + kp[0, 19]) / 2.0)
    print("  halpe2h36m 映射表逐项一致 ✓")


def test_normalize_bbox():
    seq = np.zeros((1, 17, 3))
    seq[0, 0] = [10, 300, 1]   # y 最低点
    seq[0, 1] = [30, 100, 1]   # y 最高点
    seq[0, 2] = [20, 200, 1]
    seq[0, 3] = [20, 260, 1]   # 框内第 4 点（官方实现要求 ≥4 个有效点）
    out = _normalize_bbox(seq)
    # conf=0 的点坐标同样被官方实现变换（可能出界），断言只看有效点
    m = seq[0, :, 2] != 0
    assert np.isclose(out[0, m, 1].max(), 1.0) and np.isclose(out[0, m, 1].min(), -1.0)
    assert out[0, m, 0].max() <= 1.0 and out[0, m, 0].min() >= -1.0
    # 有效点不足 4：官方实现返回全零（下游 valid_mask 判无效帧）
    sparse = np.zeros((1, 17, 3))
    sparse[0, 0] = [10, 10, 1]
    assert _normalize_bbox(sparse).sum() == 0
    # 零跨度（所有点重合）：同样全零
    same = np.zeros((2, 17, 3))
    same[..., :2] = 5.0
    same[..., 2] = 1.0
    assert _normalize_bbox(same).sum() == 0
    print("  bbox 归一化与退化分支 ✓")


def test_lift_windows_and_contract():
    from skibot.motionbert import MotionBERTLifter

    rec = Recording()
    lifter = MotionBERTLifter(infer_fn=rec)
    kp = make_kp2d(t=500)
    sk3 = lifter.lift(kp)

    # 500 帧 = 243 + 243 + 14，尾窗可短（DSTformer temp_embed 按帧切片）
    assert rec.shapes == [(243, 17, 3), (243, 17, 3), (14, 17, 3)]
    assert sk3.joints.shape == (500, 26, 3)
    assert sk3.confidence.shape == (500, 26)
    assert sk3.source == "motionbert"
    assert sk3.gravity_source == "adapter"
    assert sk3.meta["clip_len"] == 243
    # H36M-17 无对应的 Halpe 点保持缺测：NaN + conf 0
    assert np.isnan(sk3.joints[:, list(UNMAPPED)]).all()
    assert (sk3.confidence[:, list(UNMAPPED)] == 0).all()
    mapped = [j for j in range(26) if j not in UNMAPPED]
    assert np.isfinite(sk3.joints[:, mapped]).all()
    assert np.allclose(sk3.confidence[:, mapped], 0.9)
    # 契约强制 root-relative：髋中点为原点
    mid = (sk3.joints[:, L_HIP] + sk3.joints[:, R_HIP]) / 2.0
    assert np.allclose(mid, 0.0, atol=1e-9)
    assert sk3.valid_mask.all()
    print("  切窗与 Skeleton3D 契约 ✓")


def test_lift_all_missing():
    from skibot.motionbert import MotionBERTLifter

    lifter = MotionBERTLifter(infer_fn=Recording())
    kp = make_kp2d(t=10)
    kp[:, :, 2] = 0.0  # 全帧无检测：归一化返回全零，模型仍会输出，置信度全 0
    sk3 = lifter.lift(kp)
    assert (sk3.confidence == 0).all()
    assert not sk3.valid_mask.any()
    print("  全缺测帧判无效 ✓")


def test_nan_missing_joints_sanitized():
    """trc 缺测关节坐标是 NaN（不是 0）：进网络的窗口必须全是有限值。"""
    from skibot.motionbert import MotionBERTLifter

    seen: list[np.ndarray] = []

    def infer(motion: np.ndarray) -> np.ndarray:
        seen.append(motion.copy())
        return np.zeros((motion.shape[0], 17, 3), dtype=np.float32)

    kp = make_kp2d(t=30)
    kp[:, NOSE] = [np.nan, np.nan, 0.0]  # 缺测记法与 trc 一致
    sk3 = MotionBERTLifter(infer_fn=infer).lift(kp)
    assert seen
    for m in seen:
        assert np.isfinite(m).all(), "NaN 进了网络输入"
    # 输出侧：无对应的点仍按缺测（NaN + conf 0）；鼻子有对应但置信度随输入为 0
    assert np.isnan(sk3.joints[:, UNMAPPED]).all()
    assert (sk3.confidence[:, NOSE] == 0).all()
    print("  trc 缺测 NaN 不进网络 ✓")


def test_output_frame_rotation_and_hip_center():
    """模型输出 y 朝下：适配器绕 x 轴转 180°；髋中点由左右髋重建。"""
    from skibot.motionbert import MotionBERTLifter

    def infer(motion: np.ndarray) -> np.ndarray:
        out = np.zeros((motion.shape[0], 17, 3), dtype=np.float32)
        out[:, 9] = [0.0, 0.2, 0.3]   # h36m 鼻点在模型帧里 +y（朝下）+z
        out[:, 1] = [0.1, 0.0, 0.0]   # h36m 右髋
        out[:, 4] = [-0.1, 0.0, 0.0]  # h36m 左髋
        return out

    kp = make_kp2d(t=10)
    sk3 = MotionBERTLifter(infer_fn=infer).lift(kp)
    assert np.allclose(sk3.gravity_up, np.tile([0.0, 1.0, 0.0], (10, 1)))
    assert np.allclose(sk3.joints[:, NOSE, 1], -0.2)   # y 取反到朝上
    assert np.allclose(sk3.joints[:, NOSE, 2], -0.3)   # z 同取反（180° 旋转，保持手性）
    # 髋中点 = 左右髋中点（不用 h36m 0 号 pelvis 通道）
    assert np.allclose(sk3.joints[:, HIP_CENTER, 0], 0.0, atol=1e-12)
    assert np.allclose(sk3.confidence[:, HIP_CENTER], 0.9)
    print("  输出帧旋转与髋中点重建 ✓")


def test_assess_labels_and_2d_regression():
    from skibot.motionbert import MotionBERTLifter

    kp = make_kp2d(t=120)
    base = assess(kp, level="intermediate", refs=REFS, label="t")
    lifted = assess(kp, level="intermediate", refs=REFS, label="t",
                    lifter=MotionBERTLifter(infer_fn=Recording()))
    # 三维旁路不得改变二维输出（spec：两者并行产出，回归对照）
    for k, v in base.series.items():
        assert np.array_equal(v, lifted.series[k], equal_nan=True), \
            f"二维序列 {k} 被三维改动"
    assert np.array_equal(base.frame_view, lifted.frame_view)
    assert lifted.report["data_source"] == "motionbert"
    assert lifted.series3d is not None
    assert lifted.skeleton3d.gravity_source == "adapter"
    by3d = {m["metric"]: m for m in lifted.report["metrics"] if m["basis"] == "3d"}
    assert by3d["trunk_lean"]["gravity_source"] == "adapter"
    assert "note" not in by3d["trunk_lean"] or "不可信" not in by3d["trunk_lean"]["note"]
    print("  assess 标注 motionbert/adapter，二维逐位不变 ✓")


def test_real_model_integration():
    """真实权重端到端：torch 或权重缺席时跳过（可选项，不算失败）。"""
    ckpt = ROOT / "models" / "motionbert" / "FT_MB_release_MB_ft_h36m" / "best_epoch.bin"
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  跳过真实模型集成（未安装 torch）")
        return
    if not ckpt.exists():
        print("  跳过真实模型集成（权重缺失）")
        return
    from skibot.motionbert import MotionBERTLifter

    kp = make_kp2d(t=300)
    t0 = time.perf_counter()
    lifter = MotionBERTLifter()
    sk3 = lifter.lift(kp)
    dt = time.perf_counter() - t0
    device = lifter._device
    assert sk3.joints.shape == (300, 26, 3)
    frac = float(sk3.valid_mask.mean())
    assert frac > 0.5, f"有效帧占比过低 {frac}"
    peak = ""
    if device == "cuda":
        peak = f"，CUDA 峰值 {torch.cuda.max_memory_allocated() / 1e6:.0f} MB"
    print(f"  真实模型：{device} 上 300 帧提升 {dt:.2f}s，"
          f"有效帧 {frac * 100:.0f}%{peak} ✓")


if __name__ == "__main__":
    test_halpe_to_h36m_mapping()
    test_normalize_bbox()
    test_lift_windows_and_contract()
    test_lift_all_missing()
    test_nan_missing_joints_sanitized()
    test_output_frame_rotation_and_hip_center()
    test_assess_labels_and_2d_regression()
    test_real_model_integration()
    print("全部通过")
