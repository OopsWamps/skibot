"""评估收尾：assess → 原子落盘报告 → 渲染标注视频，只此一条链。

此前 cli.analyze_one / reselect_one / render 子命令 / 抢救路径四处各串一遍，
且只有改选带原子替换——生产 web 主路径（任务表 worker 跑的 analyze_one）
反而没有报告/视频错配保护。收进一个 module 后：报告与标注视频都先写临时
文件、渲染成功后按「视频 → 报告」顺序原子替换，任何一步失败都保持旧产物；
替换走 Windows 锁重试（artifacts）。复用同一结果目录重跑时，旧 trc 的
清场也属于管线职责，由 analyze_one 在跑姿态检测前完成。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from . import artifacts
from .assess import Assessment, assess
from .render import render_annotated
from .report import save_json
from .skeleton3d import Lifter


def assess_and_publish(
    kp: np.ndarray,
    src: str | Path | None,
    *,
    refs: dict,
    level: str,
    label: str,
    report_path: str | Path,
    video_path: str | Path | None = None,
    person: dict | None = None,
    frame_height: float | None = None,
    fps: float | None = None,
    lifter: Lifter | None = None,
    slope_sag_deg: float | None = None,
    quality: dict | None = None,
    calibrated: bool = False,
) -> Assessment:
    """评估一次，报告与标注视频成对发布。

    video_path 给出时同时渲染标注视频（此时 src 必须给出——渲染要画面）；
    None 时只落报告（抢救路径、render 子命令不需要落盘的场景自行 assess）。
    失败语义：任何一步抛出，旧的报告/标注视频原样保留（先备份旧视频，
    报告替换失败则回滚视频），临时文件清理干净；发布后的报告与叠加层出自
    同一次评估。
    """
    if video_path is not None and src is None:
        raise ValueError("要渲染标注视频就必须给视频源 src")
    res = assess(kp, level=level, refs=refs, label=label, person=person,
                 calibrated=calibrated, quality=quality,
                 frame_height=frame_height, lifter=lifter,
                 slope_sag_deg=slope_sag_deg, fps=fps)

    report_path = Path(report_path)
    tmp_report = report_path.with_name(report_path.name + ".tmp")
    tmp_video: Path | None = None
    backup: Path | None = None
    if video_path is not None:
        video_path = Path(video_path)
        tmp_video = video_path.with_suffix(".tmp.mp4")
        backup = video_path.with_name(video_path.name + ".bak")
    had_old = video_path is not None and video_path.exists()
    try:
        save_json(res.report, tmp_report)
        if video_path is not None and tmp_video is not None and backup is not None:
            if src is None:  # 开头已把 src 与 video_path 成对校验；再判一次收窄类型
                raise ValueError("要渲染标注视频就必须给视频源 src")
            render_annotated(src, kp, res, tmp_video)
            if not tmp_video.exists() or tmp_video.stat().st_size == 0:
                raise RuntimeError("标注视频渲染失败：输出为空")
            # 先把旧视频挪开备份：报告替换若失败，把旧视频换回来，
            # 报告与视频仍停在旧产物，不会一半新一半旧
            if had_old:
                artifacts.replace_with_retry(video_path, backup)
            artifacts.replace_with_retry(tmp_video, video_path)
        artifacts.replace_with_retry(tmp_report, report_path)
    except Exception:
        if had_old and video_path is not None and backup is not None and backup.exists():
            try:
                artifacts.replace_with_retry(backup, video_path)  # 回滚到旧视频
            except OSError as e:
                # 回滚也失败：磁盘停在「新视频 + 旧报告」的错配态，必须喊
                # 出来——旧视频留在 .bak，让用户能手工恢复
                print(f"[finalize] 警告：旧标注视频回滚失败（{e}）；"
                      f"旧视频保留在 {backup}，当前视频已是新渲染产物，"
                      "报告与视频可能错配，请重新发布", file=sys.stderr)
        raise
    else:
        if backup is not None:
            backup.unlink(missing_ok=True)
    finally:
        if tmp_video is not None:
            tmp_video.unlink(missing_ok=True)
        tmp_report.unlink(missing_ok=True)
    return res
