"""Web 壳：上传页 + 任务列表 + 报告页。

技术选型（docs/DECISIONS.md 二期待办"Web 壳"）：FastAPI + Jinja2 模板 +
原生 JS，无前端构建链；分析走 tasks.TaskBoard 后台线程。
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from starlette.staticfiles import StaticFiles

from .tasks import STATUS_DONE, STATUS_FAILED, TaskBoard

TEMPLATES = Path(__file__).parent / "templates"
LEVELS = ("novice", "intermediate", "advanced")


def create_app(result_dir: Path) -> FastAPI:
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    board = TaskBoard(result_dir)
    app = FastAPI(title="skibot", docs_url=None, redoc_url=None)
    app.state.board = board  # 测试与扩展的注入点
    tpl = Jinja2Templates(directory=str(TEMPLATES))
    # 分析产物（标注视频等）静态挂载，报告页直接引用
    app.mount("/files", StaticFiles(directory=str(result_dir)), name="files")

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return tpl.TemplateResponse(request, "upload.html", {"active": "upload"})

    @app.post("/api/upload")
    async def upload(video: UploadFile = File(...),
                     level: str = Form("intermediate"), blur: str = Form("on")):
        if level not in LEVELS:
            raise HTTPException(400, f"未知水平档: {level}")
        name = Path(video.filename or "").name
        if not name.lower().endswith((".mp4", ".mov", ".m4v", ".avi")):
            raise HTTPException(400, "仅支持 mp4/mov/m4v/avi 视频")
        task_id = uuid.uuid4().hex[:12]
        dest_dir = result_dir / "uploads" / task_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / name
        with dest.open("wb") as fh:  # 分块落盘，上传文件可达数百 MB
            while chunk := await video.read(1 << 20):
                fh.write(chunk)
        board.enqueue(task_id, dest, dest.name, level, blur == "on")
        return {"task_id": task_id}

    @app.get("/tasks", response_class=HTMLResponse)
    def tasks_page(request: Request, error: str = ""):
        return tpl.TemplateResponse(request, "tasks.html",
                                    {"active": "tasks", "tasks": board.all(),
                                     "error": error})

    @app.get("/api/tasks")
    def tasks_api():
        return [_task_json(t) for t in board.all()]

    @app.get("/api/tasks/{task_id}")
    def task_api(task_id: str):
        t = board.get(task_id)
        if t is None:
            raise HTTPException(404, "任务不存在")
        return _task_json(t)

    @app.get("/report/{task_id}", response_class=HTMLResponse)
    def report_page(request: Request, task_id: str):
        t = board.get(task_id)
        if t is None:
            raise HTTPException(404, "任务不存在")
        if t.status == STATUS_FAILED:
            return RedirectResponse(f"/tasks?error={t.error}", status_code=302)
        if t.status != STATUS_DONE or t.report_path is None:
            return RedirectResponse("/tasks", status_code=302)
        report = json.loads(t.report_path.read_text(encoding="utf-8"))
        stem = t.report_path.parent
        return tpl.TemplateResponse(request, "report.html", {
            "active": "tasks", "task": t, "report": report,
            "metrics": report.get("metrics", []),
            "annotated_url": f"/files/{stem.name}/{stem.name}_annotated.mp4",
        })

    return app


def _task_json(t) -> dict:
    return {
        "id": t.id, "video": t.video_name, "level": t.level, "status": t.status,
        "error": t.error, "created": t.created_at.isoformat(timespec="seconds"),
        "report_url": f"/report/{t.id}" if t.status == STATUS_DONE else None,
    }
