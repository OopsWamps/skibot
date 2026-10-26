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

from skibot.trc_io import (  # noqa: E402
    keypoints_halpe, load_kp, parse_trc, parse_trc_full, stitch,
)
from skibot.trc_io import write_trc as write_trc_file  # noqa: E402  避开本文件的 write_trc fixture


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


def test_keypoints_halpe_full_body22():
    # Sports2D body_with_feet 实际输出的 22 marker（2026-10-26 按
    # results_channel 真实 trc 表头核对）：肘腕头颈髋中与十个脚点此前
    # 缺别名全部落空，MotionBERT 有 4 个输入通道恒为零
    names = ["Hip", "RHip", "RKnee", "RAnkle", "RBigToe", "RSmallToe",
             "RHeel", "LHip", "LKnee", "LAnkle", "LBigToe", "LSmallToe",
             "LHeel", "Neck", "Head", "Nose", "RShoulder", "RElbow",
             "RWrist", "LShoulder", "LElbow", "LWrist"]
    xyz = np.zeros((1, 22, 3))
    xyz[:, :, 0] = np.arange(22)[None, :]  # 每列唯一坐标用于核对槽位
    out = keypoints_halpe(names, xyz)
    expect = {"Nose": 0, "LShoulder": 5, "RShoulder": 6, "LElbow": 7,
              "RElbow": 8, "LWrist": 9, "RWrist": 10, "LHip": 11, "RHip": 12,
              "LKnee": 13, "RKnee": 14, "LAnkle": 15, "RAnkle": 16,
              "Head": 17, "Neck": 18, "Hip": 19, "LBigToe": 20, "RBigToe": 21,
              "LSmallToe": 22, "RSmallToe": 23, "LHeel": 24, "RHeel": 25}
    for marker, slot in expect.items():
        col = names.index(marker)
        assert out[0, slot, 0] == col, f"{marker} 应映射到槽位 {slot}"
        assert out[0, slot, 2] == 1.0
    # 眼耳不在该 marker 集：保持缺测
    for slot in (1, 2, 3, 4):
        assert np.isnan(out[0, slot, 0]) and out[0, slot, 2] == 0.0
    print("  22 marker 全量映射（含肘腕头颈髋中脚点）✓")


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


def test_load_kp_and_write_roundtrip():
    """load_kp 组合入口 + write_trc 保形往返。

    夹具用 26 marker 全名表（表头与数据一致，write_trc 的真实使用形态：
    Sports2D 的 src trc 本就带全量 marker 名），有值槽位 xy 与 conf 不变。
    """
    alias = {0: "Nose", 3: "LEar", 4: "REar", 5: "LShoulder", 6: "RShoulder",
             11: "LHip", 12: "RHip", 13: "LKnee", 14: "RKnee", 15: "LAnkle", 16: "RAnkle"}
    names = [alias.get(i, f"J{i}") for i in range(26)]
    xyz = (np.arange(2 * 26 * 3).reshape(2, 26, 3) % 97) * 1.5
    xyz[1, 5] = np.nan  # 第 2 帧 LShoulder 丢点 → conf 逐帧
    lines = ["PathFileType\t4\t(X/Y/Z)\tfake.trc",
             "DataRate\tCameraFrameRate\tNumFrames\tNumMarkers\tUnits",
             "30.00\t30.00\t2\t26\tm",
             "Frame\tTime\t" + "\t".join(names), ""]
    for f in range(2):
        cells = [str(f + 1), f"{f / 30:.6f}"]
        for j in range(26):
            for c in range(3):
                v = xyz[f, j, c]
                cells.append("" if np.isnan(v) else f"{v:.6f}")
        lines.append("\t".join(cells))
    tmp = Path(tempfile.mkdtemp(prefix="skibot_test_trc_"))
    p = tmp / "full.trc"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    kp = load_kp(p)
    assert kp.shape == (2, 26, 3)
    assert set(np.argwhere(kp[0][:, 2] > 0).ravel().tolist()) == set(alias)

    out = tmp / "roundtrip.trc"
    write_trc_file(kp, p, out)
    kp2 = load_kp(out)
    assert (kp2[:, :, 2] == kp[:, :, 2]).all()
    m = kp[:, :, 2] > 0
    assert np.allclose(kp2[:, :, :2][m], kp[:, :, :2][m])
    print("  load_kp / write_trc 保形往返 ✓")


def test_write_trc_row_count_guard():
    p = write_trc(BASIC_TRC)
    kp = load_kp(p)
    try:
        write_trc_file(kp[:-1], p, p.parent / "bad.trc")
    except ValueError as e:
        assert "!= kp 帧" in str(e), e
    else:
        raise AssertionError("行数不符未抛 ValueError")
    print("  write_trc 行数守卫 ✓")


def _write_full_trc(directory: Path, name: str, encoding: str = "utf-8") -> Path:
    """26 marker 全量 trc（write_trc 的真实使用形态：Sports2D 的 src trc
    本就带全量 marker 名）。返回写入的路径。"""
    alias = {0: "Nose", 3: "LEar", 4: "REar", 5: "LShoulder", 6: "RShoulder",
             11: "LHip", 12: "RHip", 13: "LKnee", 14: "RKnee", 15: "LAnkle", 16: "RAnkle"}
    names = [alias.get(i, f"J{i}") for i in range(26)]
    xyz = (np.arange(2 * 26 * 3).reshape(2, 26, 3) % 97) * 1.5
    xyz[1, 5] = np.nan  # 第 2 帧 LShoulder 丢点 → conf 逐帧
    lines = ["PathFileType\t4\t(X/Y/Z)\tfake.trc",
             "DataRate\tCameraFrameRate\tNumFrames\tNumMarkers\tUnits",
             "30.00\t30.00\t2\t26\tm",
             "Frame\tTime\t" + "\t".join(names), ""]
    for f in range(2):
        cells = [str(f + 1), f"{f / 30:.6f}"]
        for j in range(26):
            for c in range(3):
                v = xyz[f, j, c]
                cells.append("" if np.isnan(v) else f"{v:.6f}")
        lines.append("\t".join(cells))
    p = directory / name
    p.write_text("\n".join(lines) + "\n", encoding=encoding)
    return p


def test_write_trc_tolerates_gbk_template():
    """GBK 头的 src trc（5748d6d 修的那类）能被解析，也要能当 write_trc 的模板。

    之前 write_trc 严格按 UTF-8 读模板，缝合抢救路径恰好死在触发编码
    修复的那类文件上；读写两条路径必须共用同一份编码知识。
    """
    tmp = Path(tempfile.mkdtemp(prefix="skibot_test_trc_"))
    p = _write_full_trc(tmp, "gbk_px_person00.trc", encoding="gbk")

    kp = load_kp(p)  # 宽容读：GBK 头能解析
    out = tmp / "stitched.trc"
    write_trc_file(kp, p, out)  # 模板读走同一条容错路径，不再崩
    kp2 = load_kp(out)
    assert (kp2[:, :, 2] == kp[:, :, 2]).all()
    assert np.allclose(kp2[:, :, :2], kp[:, :, :2], equal_nan=True)
    print("  write_trc GBK 模板 ✓")


def test_parse_trc_full_time_column():
    p = write_trc(BASIC_TRC)
    names, t, xyz = parse_trc_full(p)
    assert names == ["Nose", "LHip", "RKnee"], names
    assert np.allclose(t, [0.0, 0.033333]), t
    assert xyz.shape == (2, 3, 3)
    print("  parse_trc_full Time 列 ✓")


def test_stitch_blend_and_guard():
    """缝合：前段取供体、后段取主轨、中间逐关键点线性补插。"""
    n = 5
    kp_main = np.zeros((n, 26, 3))
    kp_donor = np.zeros((n, 26, 3))
    for f in range(n):
        kp_main[f, 11] = [100 + f, 200, 1.0]   # L_HIP 主轨
        kp_donor[f, 11] = [0 + f, 200, 1.0]    # L_HIP 供体
    merged = stitch(kp_main, kp_donor, donor_last=1, main_first=3)
    assert (merged[0, 11, 0] == 0.0) and (merged[1, 11, 0] == 1.0)   # 供体段
    assert merged[2, 11, 0] == 1.0 + 0.5 * (103.0 - 1.0)              # 线性补插：t=0.5
    assert merged[2, 11, 2] == 1.0                                   # 补插帧 conf=1
    assert merged[3, 11, 0] == 103.0 and merged[4, 11, 0] == 104.0   # 主轨段
    # 一侧端点缺点的补插帧：该点不参与混合，保留主轨在该帧的原值
    kp_donor2 = kp_donor.copy()
    kp_donor2[1, 11, 2] = 0.0
    merged2 = stitch(kp_main, kp_donor2, donor_last=1, main_first=3)
    assert merged2[2, 11, 0] == 102.0 and merged2[2, 11, 2] == 1.0
    print("  stitch 供体/补插/主轨三段 ✓")


if __name__ == "__main__":
    test_parse_trc_basic()
    test_parse_trc_empty_cell_is_nan()
    test_parse_trc_no_data_rows_raises()
    test_keypoints_halpe_aliases_and_slots()
    test_keypoints_halpe_full_body22()
    test_keypoints_halpe_per_frame_conf()
    test_load_kp_and_write_roundtrip()
    test_write_trc_row_count_guard()
    test_write_trc_tolerates_gbk_template()
    test_parse_trc_full_time_column()
    test_stitch_blend_and_guard()
    print("全部通过")
