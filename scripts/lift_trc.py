"""从已有 px trc 复现三维骨架（issue 05 验收 1）：不重跑姿态检测。

用法：
    python scripts/lift_trc.py <trc> [--out <npz>]

打印耗时、FPS 与显存峰值；--out 时保存逐帧 26 关节（root-relative）、
置信度与重力向量（npz）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows GBK 控制台兜底

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from skibot.motionbert import MotionBERTLifter  # noqa: E402
from skibot.trc_io import load_kp  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="px trc → MotionBERT 三维骨架")
    ap.add_argument("trc", type=Path, help="Sports2D 产出的 px trc 文件")
    ap.add_argument("--out", type=Path, default=None,
                    help="保存 npz 的路径（缺省只打印统计）")
    args = ap.parse_args(argv)

    kp = load_kp(args.trc)
    print(f"[skibot] 二维关键点：{args.trc.name}，{kp.shape[0]} 帧")
    lifter = MotionBERTLifter()
    t0 = time.perf_counter()
    sk3 = lifter.lift(kp)
    dt = time.perf_counter() - t0
    peak = ""
    if lifter._device == "cuda":
        import torch
        peak = f"，CUDA 峰值 {torch.cuda.max_memory_allocated() / 1e6:.0f} MB"
    print(f"[skibot] 三维提升：{sk3.source}（{lifter._device}），{dt:.2f}s"
          f"（{kp.shape[0] / dt:.0f} FPS）{peak}")
    print(f"[skibot] 有效帧 {int(sk3.valid_mask.sum())}/{sk3.n_frames}，"
          f"gravity_source={sk3.gravity_source}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, joints=sk3.joints,
                            confidence=sk3.confidence,
                            gravity_up=sk3.gravity_up,
                            source=sk3.source,
                            gravity_source=sk3.gravity_source)
        print(f"[skibot] 已保存：{args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
