"""Sports2D trc 格式 adapter：解析、Halpe-26 重排、读写两端、轨迹缝合。

布局契约（索引常量、marker 别名）在 skeleton.py；
产物定位（按文件名找 px trc）属于后端边界，在 backend.py。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .skeleton import NAME_ALIASES


def _read_text_tolerant(path: str | Path) -> str:
    """读 trc 文本：UTF-8 优先，失败退回 GBK。

    Sports2D 在 Windows 中文系统上把头部路径按系统默认编码（GBK）写盘，
    数据行是纯 ASCII 不受影响；之前只用 ASCII 文件名的素材一直没暴露。
    """
    raw = Path(path).read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("gbk", errors="replace")


def parse_trc_full(path: str | Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """解析 OpenSim .trc 全量：返回 (marker 名, Time 列 (T,), (frames, markers, 3))。"""
    lines = _read_text_tolerant(path).splitlines()
    data_rows, times, names = [], [], []
    for i, ln in enumerate(lines):
        cells = ln.split("\t")
        try:
            int(cells[0])
        except (ValueError, IndexError):
            if i == 3:  # 表头第 4 行是 marker 名
                names = [c.strip() for c in cells[2:] if c.strip()]
            continue
        data_rows.append([float(c) if c.strip() else np.nan for c in cells])
        times.append(float(cells[1]) if len(cells) > 1 and cells[1].strip() else np.nan)
    if not data_rows:
        raise ValueError(f"trc 中没有数据行: {path}")
    arr = np.array(data_rows)
    n_markers = (arr.shape[1] - 2) // 3
    xyz = arr[:, 2:2 + n_markers * 3].reshape(arr.shape[0], n_markers, 3)
    return names, np.array(times), xyz


def parse_trc(path: str | Path) -> tuple[list[str], np.ndarray]:
    """解析 OpenSim .trc。返回 (marker 名列表, (frames, markers, 3) 数组)。"""
    names, _, xyz = parse_trc_full(path)
    return names, xyz


def load_kp(path: str | Path) -> np.ndarray:
    """px trc 路径 → Halpe 契约数组 (frames, 26, 3)。主路径与全部脚本的共用入口。"""
    return keypoints_halpe(*parse_trc(path))


def keypoints_halpe(names: list[str], xyz: np.ndarray) -> np.ndarray:
    """把 trc 列按 Halpe-26 索引重排，输出 (frames, 26, 3)：[x, y, conf]。

    trc 的 Z 列丢弃；conf 为 0/1（trc 不含置信度，关键点存在即为 1），
    契约语义见 skeleton.py。缺的点整列 NaN、conf=0。
    """
    norm = [n.lower().replace("_", "").replace(" ", "") for n in names]
    out = np.full((xyz.shape[0], 26, 3), np.nan)
    out[:, :, 2] = 0.0  # 缺的点 conf=0 而非 NaN（NaN 会毒化 metrics.select_side 的置信度均值）
    for halpe_idx, aliases in NAME_ALIASES.items():
        for a in aliases:
            key = a.lower().replace("_", "").replace(" ", "")
            if key in norm:
                col = norm.index(key)
                valid = ~np.isnan(xyz[:, col, 0])
                out[:, halpe_idx, 0] = xyz[:, col, 0]
                out[:, halpe_idx, 1] = xyz[:, col, 1]
                out[:, halpe_idx, 2] = valid.astype(float)
                break
    return out


def write_trc(kp: np.ndarray, src_trc: str | Path, out_trc: str | Path) -> Path:
    """以 src_trc 的表头与 Frame#/Time 列为模板，只替换坐标区。丢点写空单元格。

    src 数据行数与 kp 帧数不符时抛 ValueError（写前校验，残缺 src 不许静默截断）。
    """
    n_markers = kp.shape[1]
    lines = Path(src_trc).read_text(encoding="utf-8").splitlines()

    def _is_data_row(ln: str) -> bool:
        try:
            int(ln.split("\t")[0])
        except (ValueError, IndexError):
            return False
        return True

    n_data = sum(1 for ln in lines if _is_data_row(ln))
    if n_data != kp.shape[0]:
        raise ValueError(f"src trc 数据行 {n_data} != kp 帧 {kp.shape[0]}")

    out, f = [], 0
    for ln in lines:
        cells = ln.split("\t")
        if not _is_data_row(ln):
            out.append(ln)  # 表头行原样保留
            continue
        vals = []
        for j in range(n_markers):
            x, y, c = kp[f, j]
            vals += [f"{x:.6f}" if c > 0 else "", f"{y:.6f}" if c > 0 else "", ""]
        f += 1
        out.append("\t".join(cells[:2] + vals))
    Path(out_trc).write_text("\n".join(out) + "\n", encoding="utf-8")
    return Path(out_trc)


def stitch(kp_main: np.ndarray, kp_donor: np.ndarray,
           donor_last: int, main_first: int) -> np.ndarray:
    """缝合断裂主轨迹：0–donor_last 取供体，main_first 起取主轨，
    中间各帧按两端逐关键点线性补插（两端都有的点才补，conf 置 1）。

    跟踪器把主角开头串到背景人物时的抢救手段（段级串轨是 select_person
    的全视频统一选人救不了的，见 docs/DECISIONS.md 验证进展 #7 的段级清洗）。
    """
    merged = kp_main.copy()
    merged[: donor_last + 1] = kp_donor[: donor_last + 1]
    a, b = kp_donor[donor_last], kp_main[main_first]
    both = (a[:, 2] > 0) & (b[:, 2] > 0)
    for f in range(donor_last + 1, main_first):
        t = (f - donor_last) / (main_first - donor_last)
        merged[f, both, :2] = a[both, :2] + t * (b[both, :2] - a[both, :2])
        merged[f, both, 2] = 1.0
    return merged
