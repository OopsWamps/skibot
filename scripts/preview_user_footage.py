"""为下载好的素材生成抽帧预览图：每条视频取 3 帧横拼一张 jpg。

用途：人工快速判断机位（侧面/正面）、人物占幅、有没有全身入镜，
决定该条进不进验证集。预览输出 preview/<bvid>.jpg，按 manifest 逐条生成。

用法：python -I scripts/preview_user_footage.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "datasets/recreational_ski/manifest.jsonl"
PREVIEW = ROOT / "datasets/recreational_ski/preview"
TH = 360  # 预览帧高


def main() -> int:
    PREVIEW.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in MANIFEST.read_text(encoding="utf-8").splitlines()]
    for r in rows:
        out = PREVIEW / f"{Path(r['file']).stem}.jpg"
        if out.exists():
            continue
        cap = cv2.VideoCapture(str(ROOT / r["file"]))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        picks = [int(n * f) for f in (0.25, 0.5, 0.75)] if n > 10 else list(range(n))
        tiles = []
        for f in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(f, n - 1))
            ok, img = cap.read()
            if not ok:
                continue
            s = TH / img.shape[0]
            img = cv2.resize(img, (int(img.shape[1] * s), TH))
            cv2.putText(img, f"f{f}", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 255), 2)
            tiles.append(img)
        cap.release()
        if not tiles:
            print(f"[preview] {r['bvid']} 读不出帧")
            continue
        sheet = np.hstack(tiles)
        cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])
        print(f"[preview] {r['bvid']} → {out.name}")
    print(f"[preview] 完成 → {PREVIEW}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
