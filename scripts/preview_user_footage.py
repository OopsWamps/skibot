"""为下载好的素材生成抽帧预览图：每条视频取 3 帧横拼一张 jpg。

用途：人工快速判断机位（侧面/正面）、人物占幅、有没有全身入镜，
决定该条进不进验证集。预览输出 preview/<bvid>.jpg，按 manifest 逐条生成。

用法：python -I scripts/preview_user_footage.py
"""

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _footage import MANIFEST, PREVIEW, ROOT, strip_frames  # noqa: E402


def main() -> int:
    PREVIEW.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in MANIFEST.read_text(encoding="utf-8").splitlines()]
    for r in rows:
        out = PREVIEW / f"{Path(r['file']).stem}.jpg"
        if out.exists():
            continue
        if strip_frames(ROOT / r["file"], out):
            print(f"[preview] {r['bvid']} → {out.name}")
        else:
            print(f"[preview] {r['bvid']} 读不出帧")
    print(f"[preview] 完成 → {PREVIEW}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
