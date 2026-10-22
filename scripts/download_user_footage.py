"""下载筛选后的 B站滑雪素材：读 curated 清单 → yt-dlp 逐条拉流 → 写 manifest。

清单是 fetch_user_footage.py 产出 candidates.jsonl 的人工筛选子集（保留 bvid 行）。
ffmpeg 用 imageio-ffmpeg 的静态二进制（系统 PATH 里没有）。下载去重靠
archive.txt，重跑只补缺失的。下完探分辨率/帧率写进 manifest.jsonl（_footage.probe）。

用法：python -I scripts/download_user_footage.py [--list curated.jsonl]
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import imageio_ffmpeg

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _footage import ARCHIVE, CURATED, MANIFEST, RAW, ROOT, probe  # noqa: E402

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", type=Path, default=CURATED)
    ap.add_argument("--format", default="bv*[height<=720]+ba/b[height<=720]/b")
    args = ap.parse_args()

    items = []
    for line in args.list.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("bvid"):
            items.append(r)
    print(f"[download] 待下载 {len(items)} 条；ffmpeg: {FFMPEG}")

    RAW.mkdir(parents=True, exist_ok=True)
    ok = 0
    with MANIFEST.open("a", encoding="utf-8") as mf:
        for i, it in enumerate(items):
            bvid = it["bvid"]
            outdir = RAW / bvid
            if any(outdir.glob(f"{bvid}.*")) and not outdir.glob("*.part"):
                print(f"[download] ({i + 1}/{len(items)}) {bvid} 已存在，跳过")
            else:
                print(f"[download] ({i + 1}/{len(items)}) {bvid} {it.get('title', '')[:40]}",
                      flush=True)
                r = subprocess.run(
                    ["yt-dlp", "--ffmpeg-location", FFMPEG,
                     "-f", args.format, "--merge-output-format", "mp4",
                     "--no-playlist", "--download-archive", str(ARCHIVE),
                     "--sleep-interval", "2", "--max-sleep-interval", "6",
                     "-o", str(outdir / f"{bvid}.%(ext)s"),
                     f"https://www.bilibili.com/video/{bvid}"],
                    capture_output=True, text=True, encoding="utf-8",
                    errors="replace", timeout=600)
                if r.returncode != 0:
                    print(f"[download] 失败 {bvid}: {r.stderr.strip()[-300:]}")
                    continue
                time.sleep(1.0)
            files = [p for p in outdir.glob(f"{bvid}.*") if p.suffix != ".part"]
            if not files:
                print(f"[download] {bvid} 无产物")
                continue
            f = files[0]
            info = probe(f)
            row = {**it, "file": str(f.relative_to(ROOT)), **info}
            mf.write(json.dumps(row, ensure_ascii=False) + "\n")
            mf.flush()
            ok += 1
            print(f"[download] {bvid}: {info['w']}x{info['h']} {info['fps']}fps "
                  f"{info['frames']} 帧", flush=True)
    print(f"[download] 完成 {ok}/{len(items)} → {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
