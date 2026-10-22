"""B站素材采集/预览脚本族的私有共享层：请求头、view API 元数据、抽帧拼图、目录布局。

只服务 scripts/ 下的采集、下载、预览、核对脚本（先 sys.path 注入 scripts/ 再
import _footage）。不属于 skibot 包，skibot 不得反向依赖这里。
"""

import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SKI = ROOT / "datasets/recreational_ski"

# datasets/recreational_ski 布局：采集 → 人工筛选 → 下载 → 预览 → 核对 各阶段产物
CANDIDATES = SKI / "candidates.jsonl"
CANDIDATES_CHANNEL = SKI / "candidates_channel.jsonl"
CURATED = SKI / "curated.jsonl"
MANIFEST = SKI / "manifest.jsonl"
ARCHIVE = SKI / "archive.txt"
RAW = SKI / "raw"
RESULTS = SKI / "results"
PREVIEW = SKI / "preview"
VERIFY_PREVIEW = SKI / "verify_preview"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

# 不带浏览器头时 B站搜索/频道接口大概率回 412（实测）
HDRS = ["--add-headers", f"User-Agent: {UA}",
        "--add-headers", "Referer: https://www.bilibili.com/"]


def fetch_meta(aid: str | None = None, bvid: str | None = None) -> dict:
    """B站 view API 元数据：aid 与 bvid 二选一，合并原 fetch_meta / fetch_meta_bvid 两份。

    失败不抛，返回 {"aid"|"bvid": 查询值, "error": ...}；成功行的首键跟查询键走，
    与两个前身各自的 jsonl 行序保持一致。
    """
    key, val = ("aid", aid) if aid else ("bvid", bvid)
    req = urllib.request.Request(
        f"https://api.bilibili.com/x/web-interface/view?{key}={val}",
        headers={"User-Agent": UA, "Referer": "https://www.bilibili.com/"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            d = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {key: val, "error": repr(e)}
    if d.get("code") != 0:
        return {key: val, "error": f"code {d.get('code')}"}
    v = d["data"]
    row = {
        "title": v["title"],
        "duration": v["duration"], "play": v["stat"]["view"],
        "pubdate": v["pubdate"], "up": v["owner"]["name"], "mid": v["owner"]["mid"],
        "desc": (v.get("desc") or "")[:200],
        "url": f"https://www.bilibili.com/video/{v['bvid']}",
    }
    if aid:
        return {"aid": aid, "bvid": v["bvid"], **row}
    return {"bvid": v["bvid"], "aid": v["aid"], **row}


def probe(path: Path) -> dict:
    """cv2 探视频：宽/高/帧率/总帧数，写 manifest 用。"""
    cap = cv2.VideoCapture(str(path))
    info = {"w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": round(cap.get(cv2.CAP_PROP_FPS) or 0, 1),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
    cap.release()
    return info


def strip_frames(video: Path, out: Path, quality: int = 82) -> bool:
    """25/50/75% 三帧横拼一张 360px 高的 jpg，黄字标帧号。

    ≤10 帧的短片取全部帧。preview 与 verify 的两份内联拷贝收敛于此。
    返回是否写出。
    """
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    picks = [int(n * f) for f in (0.25, 0.5, 0.75)] if n > 10 else list(range(n))
    tiles = []
    for f in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(f, n - 1))
        ok, img = cap.read()
        if not ok:
            continue
        s = 360 / img.shape[0]
        img = cv2.resize(img, (int(img.shape[1] * s), 360))
        cv2.putText(img, f"f{f}", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 255), 2)
        tiles.append(img)
    cap.release()
    if not tiles:
        return False
    cv2.imwrite(str(out), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return True
