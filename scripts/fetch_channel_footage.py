"""按 UP 主（频道）采集大众滑雪素材：yt-dlp 拉频道 bvid 列表 → view API 补元数据。

产出 datasets/recreational_ski/candidates_channel.jsonl（与关键词轮同构），
供人工筛后走 download_user_footage.py 下载。

用法：.venv/Scripts/python.exe scripts/fetch_channel_footage.py [--out 路径]
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _footage import CANDIDATES_CHANNEL, HDRS, fetch_meta  # noqa: E402

# 重点滑雪博主（睿君 2026-10-21 指定：阿拉丁、skicat 等）
CHANNELS = {
    "阿拉丁ski": 364913714,
    "SkiiCat": 33181532,
}


def channel_bvids(mid: int) -> list[str]:
    """yt-dlp flat-playlist 拉频道视频 bvid（标题/时长字段被 412 挡，仅 id 可用）。"""
    url = f"https://space.bilibili.com/{mid}/upload/video"
    r = subprocess.run(
        ["yt-dlp", *HDRS, "--flat-playlist", "--print", "%(id)s", url],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    return [l.strip() for l in r.stdout.splitlines() if l.strip().startswith("BV")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=CANDIDATES_CHANNEL)
    args = ap.parse_args()

    seen: set[str] = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                seen.add(json.loads(line).get("bvid", ""))
        print(f"[channel] 已有 {len(seen)} 条，增量采集")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_new = 0
    with args.out.open("a", encoding="utf-8") as fh:
        for up, mid in CHANNELS.items():
            bvids = channel_bvids(mid)
            print(f"[channel] {up}(mid={mid}): {len(bvids)} 个视频")
            for bvid in bvids:
                if bvid in seen:
                    continue
                meta = fetch_meta(bvid=bvid)
                if meta and "error" not in meta:
                    fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
                    fh.flush()
                    n_new += 1
                time.sleep(1.2)
            time.sleep(2.0)
    print(f"[channel] 完成：新增 {n_new} 条 → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
