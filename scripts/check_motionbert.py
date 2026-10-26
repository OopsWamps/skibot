"""MotionBERT 环境自检：torch（CUDA）、官方权重、DSTformer 前向各过一遍。

不依赖联网，任一步失败以非零码退出。用法：
    python scripts/check_motionbert.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows GBK 控制台兜底

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT / "third_party" / "MotionBERT"
CKPT = ROOT / "models" / "motionbert" / "FT_MB_release_MB_ft_h36m" / "best_epoch.bin"
CONFIG = REPO / "configs" / "pose3d" / "MB_ft_h36m.yaml"


def main() -> int:
    import torch

    print(f"[skibot] torch {torch.__version__}，CUDA 可用：{torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[skibot] GPU：{torch.cuda.get_device_name(0)}")

    sys.path.insert(0, str(REPO))
    from lib.utils.learning import load_backbone
    from lib.utils.tools import get_config

    args = get_config(str(CONFIG))
    if not hasattr(args, "num_joints"):
        args.num_joints = 17
    model = load_backbone(args)
    print(f"[skibot] 权重：{CKPT.name}（{CKPT.stat().st_size / 1e6:.0f} MB）")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["model_pos"]
    if any(k.startswith("module.") for k in sd):  # DataParallel 存档带前缀
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()

    x = torch.randn(1, 243, 17, 3)
    with torch.no_grad():
        y = model(x)
    if not torch.isfinite(y).all():
        print("[skibot] CPU 前向输出含非有限值", file=sys.stderr)
        return 1
    print(f"[skibot] CPU 前向：输入 {tuple(x.shape)} → 输出 {tuple(y.shape)}")

    if torch.cuda.is_available():
        model = model.cuda()
        with torch.no_grad():
            y2 = model(x.cuda())
        diff = float((y - y2.cpu()).abs().max())
        print(f"[skibot] CUDA 前向：输出 {tuple(y2.shape)}，与 CPU 最大差 {diff:.2e}")

    print("[skibot] MotionBERT 自检通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
