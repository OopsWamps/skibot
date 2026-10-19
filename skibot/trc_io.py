"""Sports2D trc 格式 adapter：解析、Halpe-26 重排。

布局契约（索引常量、marker 别名）在 skeleton.py；
产物定位（按文件名找 px trc）属于后端边界，在 backend.py。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .skeleton import NAME_ALIASES


def parse_trc(path: str | Path) -> tuple[list[str], np.ndarray]:
    """解析 OpenSim .trc。返回 (marker 名列表, (frames, markers, 3) 数组)。"""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    data_rows, names = [], []
    for i, ln in enumerate(lines):
        cells = ln.split("\t")
        try:
            int(cells[0])
        except (ValueError, IndexError):
            if i == 3:  # 表头第 4 行是 marker 名
                names = [c.strip() for c in cells[2:] if c.strip()]
            continue
        data_rows.append([float(c) if c.strip() else np.nan for c in cells])
    if not data_rows:
        raise ValueError(f"trc 中没有数据行: {path}")
    arr = np.array(data_rows)
    n_markers = (arr.shape[1] - 2) // 3
    xyz = arr[:, 2:2 + n_markers * 3].reshape(arr.shape[0], n_markers, 3)
    return names, xyz


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
