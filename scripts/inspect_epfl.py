"""勘察 Ski2DPose 标签与源视频的对应关系（一次性，任务 2/3 的前置）。

打印：标签层级、iid 语义（帧号还是序号）、每条视频的帧数/分辨率/fps、
标签帧与视频时长的覆盖情况。
"""

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "datasets/epfl_ski2dpose/ski2dpose_labels.json"
VIDS = ROOT / "datasets/epfl_ski2dpose/videos/Videos"

labels = json.loads(LABELS.read_text(encoding="utf-8"))
print(f"顶层键（vid）{len(labels)} 个: {sorted(labels)[:20]}")

vid0 = sorted(labels)[0]
sp = labels[vid0]
print(f"\nvid={vid0} 的 splits: {list(sp)}")

for split, imgs in sp.items():
    keys = sorted(imgs)
    parts = [k.split("_") for k in keys]
    clips = sorted({p[1] for p in parts})
    frames = sorted(int(p[2]) for p in parts)
    print(f"  split={split}: {len(imgs)} 帧, clip 集合 {clips[:6]}, frame 范围 {frames[0]}..{frames[-1]}")
    gaps = {b - a for a, b in zip(frames, frames[1:])}
    print(f"  frame 相邻差集合: {sorted(gaps)[:10]}")
    k0 = keys[0]
    rec = imgs[k0]
    print(f"  样例 rec[{k0}] 键: {list(rec)}")
    ann = rec["annotation"]
    print(f"  annotation 形状: {len(ann)} x {len(ann[0])}，首点 {ann[0]}")

import cv2  # noqa: E402

print("\n视频清单（vid/clip: fps, frames, WxH, 秒）:")
total_frames = 0
for vid in sorted(labels)[:40]:
    d = VIDS / vid
    if not d.is_dir():
        print(f"  {vid}: 无视频目录！")
        continue
    clips = sorted(d.glob("*.mp4"), key=lambda p: int(p.stem))
    descs = []
    labeled = {k.split("_")[1] for s in labels[vid].values() for k in s}
    for c in clips:
        cap = cv2.VideoCapture(str(c))
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        descs.append(f"{c.stem}:{fps:.1f}fps,{n}f,{w}x{h}")
        total_frames += n
    lab_s = f"标注覆盖 clip {sorted(labeled)[:8]}{'...' if len(labeled) > 8 else ''}" if labeled else "无标注"
    print(f"  {vid}: {len(clips)} 片段 [{'; '.join(descs[:4])}{'...' if len(descs) > 4 else ''}] | {lab_s}")
print(f"\n前 40 个 vid 的视频总帧数: {total_frames}")
