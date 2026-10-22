"""任务表与后台执行：上传的视频排队进 analyze_one，内存态、单工作线程串行。

分析是分钟级 CPU 任务（Sports2D 子进程 + DML 推理），并发只会互相拖慢，
内部工具单线程排队足够；服务重启丢任务表——已完成的分析产物在结果目录，
重新上传同一视频即可恢复查看。
"""

from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..cli import AnalyzeOptions, analyze_one

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"


@dataclass
class Task:
    id: str
    video_path: Path   # 上传落盘位置（uploads/<id>/<原文件名>）
    video_name: str
    level: str
    blur: bool
    status: str = STATUS_QUEUED
    error: str = ""
    report_path: Path | None = None
    created_at: datetime = field(default_factory=datetime.now)


class TaskBoard:
    """内存任务表 + 单工作线程。enqueue 前调用方已把视频落盘。"""

    def __init__(self, result_dir: Path):
        self.result_dir = result_dir
        self._tasks: dict[str, Task] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._restore()
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def _restore(self) -> None:
        """服务重启后从结果目录重建已完成任务——历史报告重启后仍可访问。

        任务 id 取视频 stem（与 report.json 所在目录同名）；上传期任务
        （uploads/ 下）无法恢复执行状态，不重建。
        """
        import json

        for rp in sorted(self.result_dir.glob("*/report.json")):
            stem = rp.parent.name
            if stem in self._tasks:
                continue
            try:
                level = json.loads(rp.read_text(encoding="utf-8")).get("level",
                                                                       "intermediate")
            except Exception:
                level = "intermediate"
            task = Task(id=stem, video_path=rp.parent / f"{stem}.mp4",
                        video_name=stem, level=level, blur=False,
                        status=STATUS_DONE, report_path=rp)
            self._tasks[stem] = task
            self._order.append(stem)

    def enqueue(self, task_id: str, video_path: Path, video_name: str,
                level: str, blur: bool) -> Task:
        task = Task(id=task_id, video_path=video_path, video_name=video_name,
                    level=level, blur=blur)
        with self._cv:
            self._tasks[task_id] = task
            self._order.append(task_id)
            self._cv.notify()
        return task

    def get(self, task_id: str) -> Task | None:
        with self._lock:
            return self._tasks.get(task_id)

    def all(self) -> list[Task]:
        with self._lock:
            return [self._tasks[i] for i in reversed(self._order)]

    def _loop(self) -> None:
        while True:
            with self._cv:
                while True:
                    todo = [i for i in self._order
                            if self._tasks[i].status == STATUS_QUEUED]
                    if todo:
                        task = self._tasks[todo[0]]
                        break
                    self._cv.wait()
                task.status = STATUS_RUNNING
            try:
                opts = AnalyzeOptions(level=task.level, blur=task.blur,
                                      result_dir=self.result_dir)
                task.report_path = analyze_one(task.video_path, opts)
                task.status = STATUS_DONE
            except Exception as e:  # noqa: BLE001 —— 任务失败进表，不拖垮服务
                task.status = STATUS_FAILED
                task.error = f"{type(e).__name__}: {e}"
                traceback.print_exc()
