"""从 B站搜集大众滑雪素材候选：关键词搜索 → aid 去重 → view API 元数据。

产出 datasets/recreational_ski/candidates.jsonl（bvid/标题/时长/播放量/UP主/简介），
供人工筛选后用 yt-dlp 下载。只拿元数据，不下载视频——下载在筛选之后单独跑。

用法：python -I scripts/fetch_user_footage.py [--out datasets/recreational_ski/candidates.jsonl]
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent

# 大众双板形态：侧面/正面跟拍、新手真实滑行、教学示范（标准动作参照）
KEYWORDS = [
    "犁式转弯",
    "双板 犁式",
    "滑雪 侧面 拍摄",
    "滑雪场 跟拍",
    "新手滑雪 摔倒",
    "滑雪 第二次",
    "崇礼 滑雪 vlog",
    "长白山 滑雪 双板",
]

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

# 不带浏览器头时 B站搜索接口大概率回 412（实测）
HDRS = ["--add-headers", f"User-Agent: {UA}", "--add-headers", "Referer: https://www.bilibili.com/"]


def search_aids(keyword: str, n: int) -> list[str]:
    r = subprocess.run(
        ["yt-dlp", *HDRS, "--flat-playlist", "--print", "%(id)s", f"bilisearch{n}:{keyword}"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)
    return [l.strip() for l in r.stdout.splitlines() if l.strip().isdigit()]


def fetch_meta(aid: str) -> dict | None:
    req = urllib.request.Request(
        f"https://api.bilibili.com/x/web-interface/view?aid={aid}",
        headers={"User-Agent": UA, "Referer": "https://www.bilibili.com/"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            d = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"aid": aid, "error": repr(e)}
    if d.get("code") != 0:
        return {"aid": aid, "error": f"code {d.get('code')}"}
    v = d["data"]
    return {
        "aid": aid, "bvid": v["bvid"], "title": v["title"],
        "duration": v["duration"], "play": v["stat"]["view"],
        "pubdate": v["pubdate"], "up": v["owner"]["name"], "mid": v["owner"]["mid"],
        "desc": (v.get("desc") or "")[:200],
        "url": f"https://www.bilibili.com/video/{v['bvid']}",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keywords", default="",
                    help="逗号分隔，覆盖内置 KEYWORDS（增量轮次用）")
    ap.add_argument("--per-keyword", type=int, default=12)
    ap.add_argument("--out", type=Path, default=ROOT / "datasets/recreational_ski/candidates.jsonl")
    args = ap.parse_args()
    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()] or KEYWORDS

    aids: list[str] = []
    seen: set[str] = set()
    if args.out.exists():  # 增量：跳过已有候选，新结果追加
        for line in args.out.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            seen.add(r.get("aid", ""))
            if "error" not in r and r.get("aid"):
                aids.append(r["aid"])
        print(f"[candidates] 已有 {len(aids)} 条，增量采集")
    for kw in keywords:
        try:
            got = search_aids(kw, args.per_keyword)
        except Exception as e:
            print(f"[candidates] 搜索失败 {kw}: {e!r}")
            continue
        fresh = [a for a in got if a not in seen and a not in aids]
        aids.extend(fresh)
        print(f"[candidates] {kw}: {len(got)} 条（新 {len(fresh)}）")
        time.sleep(1.0)

    todo = [a for a in aids if a not in seen]
    print(f"[candidates] 待拉元数据 {len(todo)} 个")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with args.out.open("a", encoding="utf-8") as fh:
        for i, aid in enumerate(todo):
            meta = fetch_meta(aid)
            fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
            n_ok += 1 if "error" not in meta else 0
            if (i + 1) % 20 == 0:
                print(f"[candidates] {i + 1}/{len(aids)}")
            time.sleep(1.2)
    print(f"[candidates] 完成：{n_ok}/{len(aids)} 成功 → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
