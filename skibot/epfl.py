"""EPFL Ski2DPose 数据集 adapter：目录布局、关节映射、GT 骨架构造。

验证进展 #1-5 对账工具（docs/DECISIONS.md「验证进展」节）的数据集接口，
收拢原先散在五份脚本里的同一份知识：
  scripts/reconcile_epfl.py       → 验证进展 #1（图像级对账）
  scripts/attribute_tail.py       → 验证进展 #2（膝弯胖尾巴归因）
  scripts/ref_from_epfl.py        → 验证进展 #4（参考值分布统计）
  scripts/reconcile_epfl_video.py → 验证进展 #5（视频级对账）
  scripts/inspect_epfl.py         → 标签勘察（#1-5 的前置）
数据集本体不入库，目录按仓库根 ROOT 的相对路径落位；解包结构或关节
映射变动只改这里，脚本不感知。

ref_from_epfl 的输出 schema（gt_angle_stats.json）是验证进展 #4 的承重墙，
本模块的映射与 gt_keypoints 构造行为不得改变其输出。

GT 可见度是二值标注契约（vis==1 记 conf 1.0，否则 0），与模型置信度
门槛 skeleton.CONF_MIN 是两套语义：比较 GT 可见度用 >= 0.5，不要换成
CONF_MIN；模型预测侧才用 CONF_MIN。
"""

from __future__ import annotations

import numpy as np

# 数据集目录布局（相对仓库根 ROOT；脚本里用 ROOT / 常量拼绝对路径）
DATASET_DIR = "datasets/epfl_ski2dpose"
IMAGES_DIR = f"{DATASET_DIR}/images"
LABELS_JSON = f"{DATASET_DIR}/ski2dpose_labels.json"
VIDEOS_DIR = f"{DATASET_DIR}/videos/Videos"

# EPFL 24 关节顺序 → Halpe-26 索引（只对账我们有映射的点）
EPFL_JOINTS = [
    "head", "neck",
    "shoulder_right", "elbow_right", "hand_right", "pole_basket_right",
    "shoulder_left", "elbow_left", "hand_left", "pole_basket_left",
    "hip_right", "knee_right", "ankle_right",
    "hip_left", "knee_left", "ankle_left",
    "ski_tip_right", "toes_right", "heel_right", "ski_tail_right",
    "ski_tip_left", "toes_left", "heel_left", "ski_tail_left",
]
EPFL_TO_HALPE = {
    "head": 17, "neck": 18,
    "shoulder_right": 6, "elbow_right": 8, "hand_right": 10,
    "shoulder_left": 5, "elbow_left": 7, "hand_left": 9,
    "hip_right": 12, "knee_right": 14, "ankle_right": 16,
    "hip_left": 11, "knee_left": 13, "ankle_left": 15,
}

# 视角分类用的映射：epfl 'head' 同时喂给 halpe 的 NOSE(0) 和 HEAD(17)：
# view 分类要鼻子信号，Ski2DPose 的 head 点是头部中心，当鼻子用是可接受
# 的近似。指标对账仍用 EPFL_TO_HALPE（head 只当 HEAD）。
EPFL_TO_HALPE_VIEW = {**EPFL_TO_HALPE, "head": 0}


def gt_keypoints(items, width, height, n_frames=None) -> np.ndarray:
    """Ski2DPose 标注记录 → (T, 26, 3) Halpe-26 骨架 [x, y, conf]。

    items 是按 rec["frame_idx"] 排好序的 (iid, rec) 序列。n_frames 给出时
    （视频级对账）每条记录写进第 rec["frame_idx"] 行，未标注帧留全零，
    T = n_frames；缺省时（分布统计）按 items 顺序稠密堆叠，T = len(items)。
    归一化坐标乘以视频宽高；vis==1 的点记 conf 1.0、未检出保持 0
    （GT 可见度是二值标注契约，见模块 docstring）。
    """
    kp = np.zeros((n_frames if n_frames is not None else len(items), 26, 3))
    for i, (_iid, rec) in enumerate(items):
        f = rec["frame_idx"] if n_frames is not None else i
        for jname, hidx in EPFL_TO_HALPE_VIEW.items():
            x, y, vis = rec["annotation"][EPFL_JOINTS.index(jname)]
            if vis == 1:
                kp[f, hidx] = (x * width, y * height, 1.0)
    return kp
