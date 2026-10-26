"""MotionBERT lift 主干（DSTformer）的 vendor 副本与构造工厂。

来源 Walter0807/MotionBERT（Apache-2.0，license 在本目录）；模型定义只把
DropPath 导入改成包内相对。超参取官方 configs/pose3d/MB_ft_h36m.yaml 的
Model 段——release lift 权重（FT_MB_release_MB_ft_h36m）与该配置配对；
att_fuse 走 DSTformer 默认 True，与 checkpoint 里的 ts_attn 键一致。
torch 是可选依赖（三维提升开关关着时不该拖累安装与导入）。
"""

from __future__ import annotations

# release lift 配置（configs/pose3d/MB_ft_h36m.yaml 的 Model 段）
MODEL_PARAMS = dict(dim_feat=512, mlp_ratio=2, depth=5, dim_rep=512,
                    num_heads=8)
MAXLEN = 243
NUM_JOINTS = 17


def build_lift_backbone():
    """构造 release lift 模型。torch 缺失时给出安装提示再抛 ImportError。"""
    try:
        import torch.nn as nn
        from torch.nn import LayerNorm
    except ImportError as e:
        raise ImportError(
            "三维提升需要 torch：pip install torch（Windows CPU 构建直接装，"
            "CUDA 构建加 --index-url https://download.pytorch.org/whl）") from e
    from functools import partial

    from .dstformer import DSTformer

    return DSTformer(dim_in=3, dim_out=3, num_joints=NUM_JOINTS, maxlen=MAXLEN,
                     norm_layer=partial(LayerNorm, eps=1e-6), **MODEL_PARAMS)
