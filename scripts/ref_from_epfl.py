"""从 Ski2DPose 人工标注统计五项指标的真实分布，给参考值表提供数据支撑。

数据集是世界杯级竞技滑雪者 → 只能锚定 advanced 档；novice/intermediate
仍靠教学经验定。视角用 skibot.view 在标注骨架上分类（每段 clip 一个标签），
侧面 clip 出矢状面三指标分布，正面 clip 出额状面两指标分布。

用法：python scripts/ref_from_epfl.py（纯 numpy，秒级）
"""

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from skibot.epfl import DATASET_DIR, LABELS_JSON, VIDEOS_DIR, gt_keypoints  # noqa: E402
from skibot.metrics import METRICS, compute_series  # noqa: E402
from skibot.reference import load as load_refs  # noqa: E402
from skibot.view import classify_view  # noqa: E402

LABELS = ROOT / LABELS_JSON
VIDS = ROOT / VIDEOS_DIR
OUT = ROOT / DATASET_DIR / "gt_angle_stats.json"


def main() -> int:
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    vals: dict[str, list[float]] = {m: [] for m in METRICS}
    clips_out = []
    n_skipped = 0

    for vid, splits in sorted(labels.items()):
        for split, imgs in sorted(splits.items()):
            vpath = VIDS / vid / f"{split}.mp4"
            if not vpath.exists():
                n_skipped += len(imgs)
                continue
            import cv2
            cap = cv2.VideoCapture(str(vpath))
            w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            cap.release()

            items = sorted(imgs.items(), key=lambda kv: kv[1]["frame_idx"])
            kp = gt_keypoints(items, w, h)

            view = classify_view(kp)
            series = compute_series(kp)
            for m, (_, need) in METRICS.items():
                if view.view == need:
                    s = series[m]
                    vals[m].extend(float(v) for v in s[~np.isnan(s)])
            clips_out.append({"vid": vid, "clip": split, "view": view.view,
                              "score": round(view.score, 2),
                              "n_valid": view.n_valid_frames})

    refs = load_refs(ROOT / "reference_values.toml")
    stats = {}
    print(f"\n{'指标':<20}{'n':>6}{'p10':>8}{'p25':>8}{'中位':>8}{'p75':>8}{'p90':>8}"
          f"  占位 novice/inter/advanced")
    for m, stats_m in sorted(vals.items()):
        a = np.array(stats_m)
        if a.size == 0:
            print(f"{m:<20}{0:>6}  （没有匹配视角的 clip）")
            continue
        p10, p25, p50, p75, p90 = np.percentile(a, [10, 25, 50, 75, 90])
        cur = {lv: [refs[m][lv].lo, refs[m][lv].hi]
               for lv in ("novice", "intermediate", "advanced")}
        print(f"{m:<20}{a.size:>6}{p10:>8.1f}{p25:>8.1f}{p50:>8.1f}{p75:>8.1f}{p90:>8.1f}"
              f"  {cur['novice']} / {cur['intermediate']} / {cur['advanced']}")
        stats[m] = {"n": int(a.size), "p10": round(float(p10), 1), "p25": round(float(p25), 1),
                    "median": round(float(p50), 1), "p75": round(float(p75), 1),
                    "p90": round(float(p90), 1),
                    "proposal_advanced": [round(float(p10)), round(float(p90))],
                    "current_placeholder": {lv: list(cur[lv]) for lv in cur}}

    OUT.write_text(json.dumps({"stats": stats, "clips": clips_out,
                               "n_frames_skipped_no_video": n_skipped},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    n_front = sum(1 for c in clips_out if c["view"] == "frontal")
    n_sag = sum(1 for c in clips_out if c["view"] == "sagittal")
    print(f"\nclip 视角分布：sagittal {n_sag} / frontal {n_front} / 其他 "
          f"{len(clips_out) - n_front - n_sag}；无视频跳过 {n_skipped} 帧")
    print(f"已写出：{OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
