"""Web 壳冒烟测试：python tests/test_web.py

路由、上传落盘、任务状态机、报告页渲染（真任务注入完成态）。
不真跑 Sports2D。
"""

import json
import sys
import tempfile
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from skibot.web.app import create_app  # noqa: E402
from skibot.web import tasks as tasks_mod  # noqa: E402


def _report_dict() -> dict:
    return {
        "schema": "skibot.report/0.1", "level": "intermediate",
        "generated_at": "2026-10-22T00:00:00",
        "view": {"detected": "sagittal", "score": -0.5, "n_valid_frames": 100},
        "assumptions": ["测试假设"], "calibrated": False, "notice": "测试水印",
        "quality": {"person_height_ratio": 0.3, "low_conf_frame_ratio": 0.1,
                    "deep_squat_frame_ratio": 0.0},
        "metrics": [
            {"metric": "knee_flexion", "name": "膝弯曲角", "reference": [140, 170],
             "median_deg": 155.0, "pct_in_range": 0.5, "band_percentile": 50.0,
             "n_valid_frames": 100, "verdict": "膝弯曲角在参考带内"}],
        "person": {"id": 0, "selection": "测试选人"},
    }


def test_web_flow(tmp: Path):
    app = create_app(tmp / "web_results")
    client = TestClient(app)

    # 页面路由
    assert client.get("/").status_code == 200
    assert client.get("/tasks").status_code == 200
    assert client.get("/report/不存在").status_code == 404

    # 上传：落盘 + 入队
    r = client.post("/api/upload", files={"video": ("clip.mp4", b"fake", "video/mp4")},
                    data={"level": "novice", "blur": "off"})
    assert r.status_code == 200, r.text
    tid = r.json()["task_id"]
    saved = tmp / "web_results" / "uploads" / tid / "clip.mp4"
    assert saved.exists() and saved.read_bytes() == b"fake"

    # 假字节分析必然失败进表（不拖垮服务）
    deadline = time.time() + 5
    t = {"status": tasks_mod.STATUS_QUEUED}
    while time.time() < deadline:
        t = client.get(f"/api/tasks/{tid}").json()
        if t["status"] == tasks_mod.STATUS_FAILED:
            break
        time.sleep(0.1)
    assert t["status"] == tasks_mod.STATUS_FAILED, t
    assert "Error" in t["error"] or "error" in t["error"].lower() or t["error"]

    # 参数校验
    assert client.post("/api/upload", files={"video": ("c.mp4", b"x", "video/mp4")},
                       data={"level": "pro"}).status_code == 400
    assert client.post("/api/upload", files={"video": ("c.gif", b"x", "image/gif")}).status_code == 400

    # 报告页：注入完成态（真渲染路径）
    board = client.app.state.board
    task = board.get(tid)
    work = task.report_path.parent if task.report_path else tmp / "web_results" / task.video_path.parent.name
    work = tmp / "web_results" / "clip"
    work.mkdir(parents=True, exist_ok=True)
    (work / "report.json").write_text(json.dumps(_report_dict(), ensure_ascii=False),
                                      encoding="utf-8")
    (work / "clip_annotated.mp4").write_bytes(b"fake")
    task.report_path = work / "report.json"
    task.status = tasks_mod.STATUS_DONE
    page = client.get(f"/report/{tid}")
    assert page.status_code == 200
    body = page.text
    assert "膝弯曲角" in body and "155.0" in body and "测试水印" in body
    assert "50.0" in body and "clip_annotated.mp4" in body

    # 任务列表页含完成条目
    assert client.get("/tasks").status_code == 200
    print("  web：路由 / 上传落盘 / 失败进表 / 完成态报告页渲染 ✓")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        test_web_flow(Path(td))
    print("全部通过")
