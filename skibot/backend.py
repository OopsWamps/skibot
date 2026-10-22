"""姿态后端 adapter（一期 #5）：Sports2D 调用、产物发现、抢救契约。

cli 只经此模块与 Sports2D 打交道。抢救契约（一期 #4 的管线 + 已知风险
"收尾阶段可能挂死"）：子进程超时即终止，只要 px trc 已落盘，流程继续；
discover_tracks 找不到产物才算失败。二期若换自管模型（docs/DECISIONS.md
二期待办），换掉这一个文件。
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

DEFAULT_TRACK_ORDER = "greatest_displacement"  # 仅影响 Sports2D 内部编号，最终由 select_person 定夺


def run_sports2d(video: Path, result_dir: Path, track_order: str = DEFAULT_TRACK_ORDER,
                 timeout_s: int = 7200) -> None:
    """调 sports2d 命令行（稳定接口）。子进程 + 超时：它收尾阶段可能挂死
    （make_c3d 已排除一个；真挂了只要 px trc 已落盘，上层继续走抢救路径）。
    不限人数：选人发生在拿到全部轨迹之后（select_person）。
    """
    exe = Path(sys.executable).parent / ("sports2d.exe" if sys.platform == "win32" else "sports2d")
    cmd = [
        str(exe),
        "-i", str(video),
        "-r", str(result_dir),
        "--person_ordering_method", track_order,
        "--make_c3d", "false",           # Windows 上 C3D 转换挂死，2026-10-18 实测
        "--backend", "onnxruntime",      # 显式 onnxruntime + DML（rtmlib 本地补丁映射 cuda→DmlExecutionProvider）；
        "--device", "cuda",              # 默认 openvino CPU 对长视频 900s 跑不完（2026-10-20 大众素材批量 3 条被超时杀）
        "--nb_persons_to_detect", "8",   # 大众素材雪场人流密集，全人数时逐人推理+滤波拖垮管线（23 人素材后处理 1.5h）；
                                         # 主角=画面最大且持续在场，必在前 8；路人轨迹短本来就选不中
                                         # 注意：不要传 --calculate_angles false——sports2d 会静默跳过整个后处理，
                                         # trc 不落盘且无报错（2026-10-21 实测，三连失败后二分定位）
        "--show_realtime_results", "false",
        "--save_img", "false",
        "--show_graphs", "false",
        "--save_graphs", "false",
    ]
    try:
        subprocess.run(cmd, check=False, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"[skibot] Sports2D 超过 {timeout_s}s 未结束，终止子进程，尝试抢救已有产物")


def find_all_trc_px(result_dir: Path) -> list[tuple[int, Path]]:
    """找全部人物的 px trc，按 person 编号排序。返回 [(pid, path), ...]。"""
    hits: dict[int, Path] = {}
    for p in Path(result_dir).rglob("*_px_person*.trc"):
        m = re.search(r"_px_person(\d+)\.trc$", p.name)
        if m:
            hits[int(m.group(1))] = p
    return sorted(hits.items())


def discover_tracks(work: Path) -> dict[int, Path]:
    """抢救契约的验收端：拿全部人物 trc，一个都没有才算失败。"""
    tracks = dict(find_all_trc_px(work))
    if not tracks:
        raise FileNotFoundError(f"在 {work} 下没有人物 trc，检查 Sports2D 日志")
    return tracks
