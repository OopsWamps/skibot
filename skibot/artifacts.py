"""分析产物（artifacts）：结果目录布局与原子落盘的唯一事实源。

结果子目录 = result_dir/<视频 stem>（web 流里 stem == task_id），内含
report.json、<stem>_annotated.mp4（标注视频）、<stem>_blurred.mp4（打码
副本）、candidate_person<pid>.jpg（改选预览）。此前这套命名散在 cli /
web.tasks / web.app / reselect 四处各写一份字面量——cli 与 reselect 指向
同一个打码副本，只因 work 目录名恰好等于 stem；现在只此一处。

打码源解析（source_video）与 Windows 文件锁重试（replace_with_retry /
write_text_atomic / read_text_retry）也在这里。blur 的三态语义：
- True：必须用打码副本；副本丢了就报错，绝不静默回退到未打码原片
  （否则标注视频会把已打过码的脸重新暴露）；
- False：用上传原件，即使目录里还留着上一次打码留下的旧副本；
- None（旧任务或 CLI 未传）：用副本是否存在推断。
"""

from __future__ import annotations

import re
import time
from pathlib import Path


def work_dir(result_dir: Path, video_stem: str) -> Path:
    """一次分析的结果子目录：result_dir/<stem>。"""
    return Path(result_dir) / video_stem


REPORT_NAME = "report.json"
PREVIEW_GLOB = "candidate_person*.jpg"


def report_path(work: Path) -> Path:
    return Path(work) / REPORT_NAME


def annotated_path(work: Path) -> Path:
    return Path(work) / f"{work.name}_annotated.mp4"


def blurred_path(work: Path) -> Path:
    return Path(work) / f"{work.name}_blurred.mp4"


def preview_path(work: Path, pid: int) -> Path:
    return Path(work) / f"candidate_person{pid}.jpg"


# 预览文件名 → pid（render_previews 清理幽灵预览用）
PREVIEW_RE = re.compile(r"candidate_person(\d+)\.jpg")


def files_url(work: Path, filename: str) -> str:
    """产物在 /files 静态挂载下的地址（web 壳把 result_dir 挂在 /files）。"""
    return f"/files/{work.name}/{filename}"


def source_video(work: Path, fallback_video: Path,
                 blur: bool | None = None) -> Path:
    """首次分析用的视频源：打过码就用打码副本，否则用上传原件。

    blur 由调用方从任务表/CLI 传入（首次分析的真实设置），三态语义见
    module docstring。改选与预览都读同一个源，保证报告与叠加层一致。
    """
    blurred = blurred_path(work)
    if blur is True:
        if not blurred.exists():
            raise FileNotFoundError(
                f"首次分析开了人脸打码，但打码副本 {blurred.name} 已丢失；"
                "请重新分析或显式传入未打码源")
        return blurred
    if blur is False:
        return fallback_video
    return blurred if blurred.exists() else fallback_video


def replace_with_retry(src: Path, dst: Path, attempts: int = 5) -> None:
    """os.replace 覆盖：Windows 上浏览器/杀毒可能短暂持有目标文件的读句柄，
    短退避重试几次再报错。"""
    for attempt in range(attempts):
        try:
            Path(src).replace(dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.1 * (attempt + 1))


def write_text_atomic(path: Path, text: str, attempts: int = 3) -> None:
    """文本原子落盘：写临时文件后 os.replace，进程崩溃不会留下半个文件。

    写与替换都在重试环内：Windows 上杀毒/索引器可能短暂锁住刚写出的
    文件。仍失败则清掉临时文件、抛出最后一次 OSError，由调用方决定
    放弃还是上抛。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    for attempt in range(attempts):
        try:
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(path)
            return
        except OSError:
            if attempt == attempts - 1:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.05 * (attempt + 1))


def read_text_retry(path: Path, attempts: int = 3) -> str:
    """读文本，容忍 Windows 上杀毒/索引器对文件的短暂锁定。"""
    for attempt in range(attempts):
        try:
            return Path(path).read_text(encoding="utf-8")
        except OSError as e:
            if attempt == attempts - 1:
                raise
            time.sleep(0.05 * (attempt + 1))
    raise AssertionError("unreachable")
