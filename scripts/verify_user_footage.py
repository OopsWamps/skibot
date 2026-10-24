"""批量核对大众素材分析产物：标注视频抽帧拼图 + 角度曲线跳变检查 + 指标汇总。

放行标准①在大众素材上的落实：人工看拼图核对骨架贴合，脚本查跳变和指标分布。
用法：.venv/Scripts/python.exe scripts/verify_user_footage.py
产物：datasets/recreational_ski/verify_preview/*.jpg、verify_summary.txt
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _footage import RESULTS, VERIFY_PREVIEW, strip_frames  # noqa: E402
from skibot import artifacts  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# 相邻帧角速度阈值（度/秒）：滑雪动态动作通常 <300；持续 >600 视为跳变/误检
JUMP_RATE_DEG_S = 600.0


def read_angle_mot(path: Path) -> tuple[list[str], np.ndarray]:
    """解析 Sports2D 的 .mot：endheader 后一行列名，随后 tab 分白数据（含空值）。"""
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r", "")
    lines = text.splitlines()
    end = lines.index("endheader")
    cols = [c.strip() for c in lines[end + 1].split("\t") if c.strip()]
    data = []
    for ln in lines[end + 2:]:
        parts = ln.split("\t")
        row = [float(p) if p.strip() not in ("", "NaN") else np.nan
               for p in parts[:len(cols)]]
        if len(row) == len(cols):
            data.append(row)
    return cols, np.array(data, dtype=float)


def jump_stats(sports2d_dir: Path, person_id: int, fps: float) -> str:
    """选中人物角度曲线的跳变占比，按最差前三列返回。"""
    hits = sorted(sports2d_dir.glob(f"*angles_person{person_id:02d}.mot"))
    if not hits:
        return "no-mot"
    cols, arr = read_angle_mot(hits[0])
    stats = []
    for j, name in enumerate(cols):
        if name == "time":
            continue
        seq = arr[:, j]
        seq = seq[np.isfinite(seq)]
        if len(seq) < 30:
            continue
        rate = np.abs(np.diff(seq)) * fps
        stats.append(((rate > JUMP_RATE_DEG_S).mean(), name, float(np.nanmax(rate))))
    if not stats:
        return "no-data"
    stats.sort(reverse=True)
    return "; ".join(f"{name}:{pct*100:.1f}%>600d/s(max{mx:.0f})"
                     for pct, name, mx in stats[:3])


def video_fps(video: Path) -> float:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps


def main() -> int:
    VERIFY_PREVIEW.mkdir(exist_ok=True)
    lines = []
    for d in sorted(RESULTS.iterdir()):
        if not d.is_dir():
            continue
        bv = d.name
        rep_path = artifacts.report_path(d)
        ann = artifacts.annotated_path(d)
        if not rep_path.exists() or not ann.exists():
            lines.append(f"{bv}\tINCOMPLETE (report={rep_path.exists()} ann={ann.exists()})")
            continue
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        view = rep.get("view", {})
        strip_frames(ann, VERIFY_PREVIEW / f"{bv}.jpg")
        pid = rep.get("person", {}).get("id", 0)
        jumps = jump_stats(d / f"{bv}_Sports2D", pid, video_fps(ann))
        mets = "; ".join(
            f"{m['metric']}:med={m.get('median_deg')}pct={m.get('pct_in_range')}"
            for m in rep.get("metrics", []))
        person = rep.get("person", {})
        lines.append(
            f"{bv}\tview={view.get('detected')}/score={view.get('score'):.2f}"
            f"/valid={view.get('n_valid_frames')}\tperson={pid}({person.get('selection')})\t{mets}\tjumps[{jumps}]")
    out = ROOT / "datasets/recreational_ski/verify_summary.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
