"""Web 壳：上传页 + 任务列表 + 报告页。

技术选型（docs/DECISIONS.md 二期待办"Web 壳"）：FastAPI + Jinja2 模板 +
原生 JS，无前端构建链；分析走 tasks.TaskBoard 后台线程。
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from urllib.parse import urlencode

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from starlette.staticfiles import StaticFiles

from .. import artifacts
from ..precheck import PrecheckConfig, run_precheck
from ..reselect import (
    current_person, ensure_previews, list_candidates, source_video,
)
from .tasks import STATUS_FAILED, TaskBoard

TEMPLATES = Path(__file__).parent / "templates"
LEVELS = ("novice", "intermediate", "advanced")


def create_app(result_dir: Path, *, enable_precheck: bool = True,
               precheck_config: PrecheckConfig | None = None,
               analyze_fn=None) -> FastAPI:
    """result_dir 同时是任务表、上传落盘与分析产物的根目录。

    enable_precheck=False 关掉上传预检（默认开）；precheck_config 覆盖判据
    默认值（默认见 PrecheckConfig 的 docstring）；analyze_fn 注入任务表的
    分析函数（测试用替身，不跑 Sports2D）。
    """
    # 先转绝对路径：上传落盘路径会原样写进 tasks.json，默认的相对
    # --result-dir（web_results）换个工作目录启动就找不到源视频了
    result_dir = Path(result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    board = TaskBoard(result_dir, analyze_fn=analyze_fn)
    app = FastAPI(title="skibot", docs_url=None, redoc_url=None)
    app.state.board = board  # 测试与扩展的注入点
    tpl = Jinja2Templates(directory=str(TEMPLATES))
    # 未打码的上传原件（result_dir/uploads/）不对外：路由先于挂载注册，
    # /files/uploads/* 在这里就被 404，静态挂载只服务各结果子目录
    @app.get("/files/uploads/{rest:path}")
    def deny_uploads(rest: str):
        raise HTTPException(404, "源视频不对外提供；请查看报告页的标注视频")

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
        suffix = Path(name).suffix.lower()
        if suffix not in (".mp4", ".mov", ".m4v", ".avi"):
            raise HTTPException(400, "仅支持 mp4/mov/m4v/avi 视频")
        task_id = uuid.uuid4().hex[:12]
        dest_dir = result_dir / "uploads" / task_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        # 落盘用 ASCII 名 <task_id>.<ext>：中文文件名会让 Sports2D 在中文系统
        # 上把 trc 头部按 GBK 写盘；带 task_id 保证 analyze_one 的结果子目录
        # （result_dir/<video.stem>）逐任务唯一，重启恢复时 id 与目录名一致。
        dest = dest_dir / f"{task_id}{suffix}"
        with dest.open("wb") as fh:  # 分块落盘，上传文件可达数百 MB
            while chunk := await video.read(1 << 20):
                fh.write(chunk)
        # 预检是秒级阻塞调用（OpenCV），丢线程池跑，别卡住事件循环；
        # 不合格的素材直接删掉落盘文件，不进分析队列
        notes: list[str] = []
        if enable_precheck:
            result = await run_in_threadpool(run_precheck, dest, precheck_config)
            if not result.passed:
                shutil.rmtree(dest_dir, ignore_errors=True)
                raise HTTPException(422, detail={
                    "message": "视频未通过上传预检，未进入分析队列",
                    "issues": [{"check": i.check, "problem": i.problem,
                                "advice": i.advice} for i in result.issues],
                })
            notes = result.notes
        board.enqueue(task_id, dest, name, level, blur == "on")  # 显示名保留原名
        return {"task_id": task_id, "precheck_notes": notes}

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
        d = _task_json(t)
        d["reselecting"] = t.reselecting  # 页面轮询改选进度用
        d["reselect_error"] = t.reselect_error or None
        return d

    @app.post("/api/tasks/{task_id}/retry")
    def retry_api(task_id: str):
        res = board.retry(task_id)
        if res.task is None:
            raise HTTPException(404, "任务不存在")
        if not res.requeued:
            raise HTTPException(409, res.error or "只能重试失败的任务")
        d = _task_json(res.task)
        d["status"] = res.status      # 锁内快照，不受 worker 抢跑影响
        d["report_url"] = None
        return d

    @app.get("/api/tasks/{task_id}/candidates")
    def candidates_api(task_id: str):
        t = _done_task(board, task_id)
        work = t.report_path.parent
        cands = ensure_previews(work, source_video(work, t.video_path, t.blur))
        return {"current": current_person(work),
                "candidates": [{"id": c.pid, "n_valid_frames": c.n_valid,
                                "preview_url": _preview_url(work, c.pid)}
                               for c in cands]}

    @app.post("/api/tasks/{task_id}/person")
    def reselect_api(task_id: str, person: int = Form(...)):
        t = _done_task(board, task_id)
        work = t.report_path.parent
        pids = {c.pid for c in list_candidates(work)}
        if person not in pids:
            raise HTTPException(400, f"没有 person {person}，可选 {sorted(pids)}")
        # 改选作为任务的子状态进任务表，worker 串行执行；状态随 tasks.json
        # 持久化，重启不丢（此前是 web 层手搓的 set+dict+裸线程影子状态机）
        if not board.start_reselect(task_id, person):
            raise HTTPException(409, "该任务正在改选，请稍候")
        return {"status": "reselecting", "person": person}

    @app.get("/report/{task_id}", response_class=HTMLResponse)
    def report_page(request: Request, task_id: str):
        t = board.get(task_id)
        if t is None:
            raise HTTPException(404, "任务不存在")
        if t.status == STATUS_FAILED:
            return RedirectResponse("/tasks?" + urlencode({"error": t.error}),
                                    status_code=302)
        if not t.done:
            return RedirectResponse("/tasks", status_code=302)
        if t.report_path is None:  # done 已蕴含非空；显式写出让类型检查收窄
            return RedirectResponse("/tasks", status_code=302)
        report = json.loads(t.report_path.read_text(encoding="utf-8"))
        work = t.report_path.parent
        cands = ensure_previews(work, source_video(work, t.video_path, t.blur))
        return tpl.TemplateResponse(request, "report.html", {
            "active": "tasks", "task": t, "report": report,
            "metrics": report.get("metrics", []),
            "annotated_url": artifacts.files_url(
                work, artifacts.annotated_path(work).name),
            "candidates": [{"id": c.pid, "n_valid_frames": c.n_valid,
                            "preview_url": _preview_url(work, c.pid)}
                           for c in cands],
            "current_person": current_person(work),
        })

    return app


def _preview_url(work: Path, pid: int) -> str | None:
    """候选预览的静态地址；没写成功（轨迹无有效帧等）返回 None。"""
    p = artifacts.preview_path(work, pid)
    return artifacts.files_url(work, p.name) if p.exists() else None


def _done_task(board: TaskBoard, task_id: str):
    t = board.get(task_id)
    if t is None:
        raise HTTPException(404, "任务不存在")
    if not t.done:
        raise HTTPException(409, "任务尚未完成，无法改选人物")
    return t


def _task_json(t) -> dict:
    return {
        "id": t.id, "video": t.video_name, "level": t.level, "status": t.status,
        "error": t.error, "created": t.created_at.isoformat(timespec="seconds"),
        "report_url": f"/report/{t.id}" if t.done else None,
    }
