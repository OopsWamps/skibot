"""反查已完成分析素材中的连续滑行段：选中人物 trc 的连续有效区间 → 原视频时间段。

判定：trc 行有效 = 多数关键点似然 > 0.5 且坐标非 NaN。连续 ≥ MIN_S 秒的区间即候选滑行段。
用法：.venv/Scripts/python.exe scripts/find_continuous_runs.py [result_dir ...]
产物：datasets/recreational_ski/continuous_runs.txt（每条素材的滑行段列表，原视频帧号）
"""

import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skibot import artifacts  # noqa: E402
from skibot.trc_io import parse_trc_full  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "datasets/recreational_ski/results"
MIN_S = 10.0          # 连续滑行段最短秒数
LIK_THR = 0.5         # 本工具口径备忘：原设计按似然筛，trc 本体无似然列，实际判据只看坐标
KP_FRAC = 0.6         # 有效关键点占比阈值


def read_trc(path: Path) -> tuple[float, np.ndarray]:
    """读 Sports2D px trc：解析走 trc_io.parse_trc_full，这里只算帧率与有效掩码。

    有效行 = x/y 均非 NaN 的关键点占比 ≥ KP_FRAC（trc 丢点写空单元格 → NaN，
    与原先的 isfinite 判据同义）。
    """
    _, time, xyz = parse_trc_full(path)
    ok_kp = np.isfinite(xyz[:, :, 0]) & np.isfinite(xyz[:, :, 1])
    valid = ok_kp.mean(axis=1) >= KP_FRAC
    dt = np.diff(time)
    pos = dt[dt > 0]
    fps = 1.0 / np.median(pos) if len(pos) else 30.0
    return fps, valid


def continuous_runs(valid: np.ndarray, fps: float) -> list[tuple[int, int, float]]:
    """连续 True 区间 → [(起帧, 止帧, 时长s), ...]，按帧号（0 起）。"""
    runs = []
    start = None
    for i, v in enumerate(valid):
        if v and start is None:
            start = i
        elif not v and start is not None:
            dur = (i - start) / fps
            if dur >= MIN_S:
                runs.append((start, i - 1, dur))
            start = None
    if start is not None:
        dur = (len(valid) - start) / fps
        if dur >= MIN_S:
            runs.append((start, len(valid) - 1, dur))
    return runs


def person_from_report(d: Path) -> int | None:
    rep = artifacts.report_path(d)
    if not rep.exists():
        return None
    import json
    return json.loads(rep.read_text(encoding="utf-8")).get("person", {}).get("id")


def main() -> int:
    dirs = [Path(a) if Path(a).is_absolute() else ROOT / a for a in sys.argv[1:]]
    dirs = [d for d in dirs if d.is_dir()] or sorted(p for p in RES.iterdir() if p.is_dir())
    out = []
    for d in dirs:
        bv = d.name
        pid = person_from_report(d)
        if pid is None:
            out.append(f"{bv}: no report")
            continue
        hits = sorted((d / f"{bv}_Sports2D").glob(f"*px_person{pid:02d}.trc"))
        if not hits:
            out.append(f"{bv}: no trc for person{pid:02d}")
            continue
        fps, valid = read_trc(hits[0])
        runs = continuous_runs(valid, fps)
        cov = valid.mean() if len(valid) else 0
        out.append(f"{bv} (person{pid:02d}, {fps:.0f}fps, 有效 {cov*100:.0f}%): "
                   + (", ".join(f"[{a}-{b}] {dur:.0f}s" for a, b, dur in runs[:8]) or "无 ≥10s 连续段"))
    outp = ROOT / "datasets/recreational_ski/continuous_runs.txt"
    outp.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"written {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
