"""TRC 解析与 Halpe 映射测试：python tests/test_trc.py
用合成 trc 文本钉住 trc_io.parse_trc / trc_io.keypoints_halpe 的行为，
不依赖 sports2d/opencv，也不解压 datasets。
"""

import sys
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows GBK 控制台兜底

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skibot.trc_io import keypoints_halpe, parse_trc  # noqa: E402


def write_trc(text: str) -> Path:
    """把 trc 文本写进临时目录，返回路径。"""
    tmp = Path(tempfile.mkdtemp(prefix="skibot_test_trc_"))
    p = tmp / "fake_px_person00.trc"
    p.write_text(text, encoding="utf-8")
    return p


# 3 marker × 2 帧的最小 trc：数值手工可算
BASIC_TRC = """PathFileType\t4\t(X/Y/Z)\tD:/fake/px_person00.trc
DataRate\tCameraFrameRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames
30.00\t30.00\t2\t3\tm\t30.00\t1\t2
Frame\tTime\tNose\tLHip\tRKnee

1\t0.000000\t10.0\t20.0\t30.0\t40.0\t50.0\t60.0\t70.0\t80.0\t90.0
2\t0.033333\t11.0\t21.0\t31.0\t41.0\t51.0\t61.0\t71.0\t81.0\t91.0
"""


def test_parse_trc_basic():
    p = write_trc(BASIC_TRC)
    names, xyz = parse_trc(p)
    assert names == ["Nose", "LHip", "RKnee"], names
    assert xyz.shape == (2, 3, 3), xyz.shape
    # 第 1 帧：Frame/Time 两列之后按 marker 顺序每 3 列一个 xyz
    assert xyz[0].tolist() == [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]]
    assert xyz[1].tolist() == [[11.0, 21.0, 31.0], [41.0, 51.0, 61.0], [71.0, 81.0, 91.0]]
    print("  parse_trc 基本解析 ✓")


# 第 2 帧 LHip 的 x 为空（遮挡丢点）：LHip 三列 = 空, 51, 61
GAP_TRC = """PathFileType\t4\t(X/Y/Z)\tD:/fake/px_person00.trc
DataRate\tCameraFrameRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames
30.00\t30.00\t2\t3\tm\t30.00\t1\t2
Frame\tTime\tNose\tLHip\tRKnee

1\t0.000000\t10.0\t20.0\t30.0\t40.0\t50.0\t60.0\t70.0\t80.0\t90.0
2\t0.033333\t11.0\t21.0\t31.0\t\t51.0\t61.0\t71.0\t81.0\t91.0
"""


def test_parse_trc_empty_cell_is_nan():
    p = write_trc(GAP_TRC)
    names, xyz = parse_trc(p)
    assert xyz.shape == (2, 3, 3), xyz.shape  # 行没被丢掉
    assert np.isnan(xyz[1, 1, 0]), xyz[1]      # LHip.x 空 → NaN
    assert xyz[1, 1, 1] == 51.0 and xyz[1, 1, 2] == 61.0
    assert xyz[0, 1].tolist() == [40.0, 50.0, 60.0]  # 第 1 帧不受影响
    print("  parse_trc 空单元格 → NaN ✓")


# 只有表头、没有数据行：Sports2D 收尾挂死时常留下的残缺文件
HEADER_ONLY_TRC = """PathFileType\t4\t(X/Y/Z)\tD:/fake/px_person00.trc
DataRate\tCameraFrameRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames
30.00\t30.00\t2\t3\tm\t30.00\t1\t2
Frame\tTime\tNose\tLHip\tRKnee

"""


def test_parse_trc_no_data_rows_raises():
    p = write_trc(HEADER_ONLY_TRC)
    try:
        parse_trc(p)
    except ValueError as e:
        assert "px_person00" in str(e), e  # 报错里要能定位到文件
    else:
        raise AssertionError("残缺 trc 没有抛 ValueError")
    print("  parse_trc 无数据行 → ValueError ✓")


def test_keypoints_halpe_aliases_and_slots():
    # trc marker 名带下划线、大小写混用（Sports2D 各版本写法不一）
    names = ["Nose", "L_Hip", "R_Knee"]
    # z 值故意给成 3/6/9，断言 conf=1 即证明 z 列被丢弃
    xyz = np.array([
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        [[11.0, 12.0, 13.0], [14.0, 15.0, 16.0], [17.0, 18.0, 19.0]],
    ])
    out = keypoints_halpe(names, xyz)
    assert out.shape == (2, 26, 3), out.shape
    assert out[0, 0].tolist() == [1.0, 2.0, 1.0]      # Nose：xy 保留，z→conf=1
    assert out[1, 11].tolist() == [14.0, 15.0, 1.0]   # L_Hip → 槽位 11
    assert out[1, 14].tolist() == [17.0, 18.0, 1.0]   # R_Knee → 槽位 14
    assert np.isnan(out[0, 15, 0]) and out[0, 15, 2] == 0.0  # LAnkle 没给：NaN + conf=0
    print("  keypoints_halpe 别名匹配与槽位 ✓")


def test_keypoints_halpe_per_frame_conf():
    # L_Hip 第 2 帧丢点：conf 应逐帧给 1/0，而不是整列一个值
    names = ["Nose", "L_Hip"]
    xyz = np.array([
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[11.0, 12.0, 13.0], [np.nan, np.nan, np.nan]],
    ])
    out = keypoints_halpe(names, xyz)
    assert out[0, 11].tolist() == [4.0, 5.0, 1.0]                    # 有数据帧 conf=1
    assert np.isnan(out[1, 11, 0]) and np.isnan(out[1, 11, 1])       # xy 保持 NaN
    assert out[1, 11, 2] == 0.0                                      # 丢点帧 conf=0
    assert out[1, 0, 2] == 1.0                                       # 同帧 Nose 不受影响
    print("  keypoints_halpe 逐帧置信度 ✓")


if __name__ == "__main__":
    test_parse_trc_basic()
    test_parse_trc_empty_cell_is_nan()
    test_parse_trc_no_data_rows_raises()
    test_keypoints_halpe_aliases_and_slots()
    test_keypoints_halpe_per_frame_conf()
    print("全部通过")
