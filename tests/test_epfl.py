"""EPFL 数据集 adapter 测试：python tests/test_epfl.py
合成标注钉住 gt_keypoints 两种模式与映射契约——gt_angle_stats.json 是
验证进展 #4 的承重墙，映射翻车就是参考值翻车，必须有测试。
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows GBK 控制台兜底

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skibot.epfl import EPFL_JOINTS, EPFL_TO_HALPE, EPFL_TO_HALPE_VIEW, gt_keypoints  # noqa: E402
from skibot.skeleton import NOSE  # noqa: E402


def blank_rec(frame_idx: int) -> dict:
    return {"frame_idx": frame_idx, "annotation": [(0.0, 0.0, 0)] * len(EPFL_JOINTS)}


def test_mapping_contract():
    assert len(EPFL_JOINTS) == 24 and len(set(EPFL_JOINTS)) == 24
    assert set(EPFL_TO_HALPE) <= set(EPFL_JOINTS)  # index() 不许炸
    # head 只在 VIEW 映射里改喂 NOSE(0)，其余条目与对账映射逐项一致
    assert EPFL_TO_HALPE["head"] == 17 and EPFL_TO_HALPE_VIEW["head"] == 0
    non_head_view = {k: v for k, v in EPFL_TO_HALPE_VIEW.items() if k != "head"}
    non_head_ref = {k: v for k, v in EPFL_TO_HALPE.items() if k != "head"}
    assert non_head_view == non_head_ref
    print("  映射契约：24 关节 + head 双喂 NOSE/HEAD ✓")


def test_gt_keypoints_dense():
    rec = blank_rec(0)
    rec["annotation"][EPFL_JOINTS.index("shoulder_left")] = (0.1, 0.2, 1)
    rec["annotation"][EPFL_JOINTS.index("head")] = (0.3, 0.4, 1)
    kp = gt_keypoints([(0, rec)], width=100, height=200)
    assert kp.shape == (1, 26, 3)
    assert kp[0, EPFL_TO_HALPE["shoulder_left"]].tolist() == [10.0, 40.0, 1.0]
    assert kp[0, NOSE].tolist() == [30.0, 80.0, 1.0]      # head 在 VIEW 构造里喂 NOSE
    assert kp[0, 17].tolist() == [0.0, 0.0, 0.0]          # HEAD 槽位由对账映射（EPFL_TO_HALPE）负责
    assert kp[0, EPFL_TO_HALPE["hip_right"]].tolist() == [0.0, 0.0, 0.0]  # vis==0 全零
    print("  gt_keypoints 稠密模式：缩放/槽位/vis ✓")


def test_gt_keypoints_sparse():
    recs = []
    for f in (0, 5):
        r = blank_rec(f)
        r["annotation"][EPFL_JOINTS.index("knee_left")] = (0.5, 0.5, 1)
        recs.append((f, r))
    kp = gt_keypoints(recs, width=10, height=10, n_frames=8)
    assert kp.shape == (8, 26, 3)                          # T = n_frames
    assert kp[0, EPFL_TO_HALPE["knee_left"]].tolist() == [5.0, 5.0, 1.0]
    assert kp[5, EPFL_TO_HALPE["knee_left"]].tolist() == [5.0, 5.0, 1.0]
    assert (kp[1] == 0).all() and (kp[7] == 0).all()       # 未标注帧全零
    print("  gt_keypoints 稀疏模式：frame_idx 落位 ✓")


if __name__ == "__main__":
    test_mapping_contract()
    test_gt_keypoints_dense()
    test_gt_keypoints_sparse()
    print("全部通过")
