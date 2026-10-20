"""膝弯误差胖尾巴归因：从 reconcile_epfl.py 的逐样本明细（.detail.jsonl）分桶。

四个候选原因逐一对账：
  A 左右腿交换（swap 标记：预测的 X 侧关节离标注的另一侧更近）
  B 预测置信度低（踝/膝点被雪靴、雪雾遮挡的代理信号）
  C 深蹲姿态（gt 膝弯角小 = 弯得深，2D 压缩 + 极端姿态）
  D 关节误差贡献（髋/膝/踝哪个点的位移把角度带歪）
输出每个原因在尾部（>15°）与主体（≤15°）的出现率对比。
"""

import json
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DETAIL = ROOT / "datasets/epfl_ski2dpose/reconcile_report.detail.jsonl"

KNEE_ANGLES = ["knee_flexion_left", "knee_flexion_right"]
SIDE_OF = {"knee_flexion_left": "left", "knee_flexion_right": "right"}


def bucket(e: float) -> str:
    for lim in (5, 10, 15, 25, 45):
        if e <= lim:
            return f"≤{lim}°"
    return ">45°"


def main() -> int:
    rows = [json.loads(l) for l in DETAIL.read_text(encoding="utf-8").splitlines()]
    print(f"明细 {len(rows)} 帧")

    samples = []  # (err, swap_hits, conf_min, gt, joint_errs{hip,knee,ankle}, scale, key)
    for r in rows:
        for an in KNEE_ANGLES:
            a = r["angles"].get(an)
            if not a:
                continue
            side = SIDE_OF[an]
            swap_hits = sum(1 for s in r["swaps"] if side in s)
            trip = {}
            for jn in (f"hip_{side}", f"knee_{side}", f"ankle_{side}"):
                j = r["joints"].get(jn)
                if j:
                    trip[jn.split("_")[0]] = j["err_px"]
            confs = [j["conf"] for jn, j in r["joints"].items() if side in jn]
            samples.append({
                "err": a["err"], "gt": a["gt"], "swap": swap_hits,
                "conf_min": min(confs) if confs else 0.0,
                "trip": trip, "scale": r["scale_px"], "key": r["key"],
            })

    order = ["≤5°", "≤10°", "≤15°", "≤25°", "≤45°", ">45°"]
    by_b: dict[str, list] = {b: [] for b in order}
    for s in samples:
        by_b[bucket(s["err"])].append(s)

    print(f"\n{'桶':>6}{'n':>6}{'中位gt':>8}{'有swap标记':>10}{'conf<0.5':>9}{'gt<110°':>9}")
    for b in order:
        ss = by_b[b]
        if not ss:
            continue
        n = len(ss)
        swap = sum(1 for s in ss if s["swap"] > 0)
        lowconf = sum(1 for s in ss if s["conf_min"] < 0.5)
        deep = sum(1 for s in ss if s["gt"] < 110)
        med_gt = sorted(s["gt"] for s in ss)[n // 2]
        print(f"{b:>6}{n:>6}{med_gt:>8.0f}{swap:>9}{lowconf:>9}{deep:>9}")

    # 尾部（>15°）细看：哪个关节把角度带歪
    tail = [s for b in ("≤25°", "≤45°", ">45°") for s in by_b[b] if s["err"] > 15]
    print(f"\n尾部 >15° 共 {len(tail)} 样本")
    if tail:
        dom = Counter()
        for s in tail:
            if s["trip"]:
                dom[max(s["trip"], key=s["trip"].get)] += 1
        print("  误差主导关节:", dict(dom))
        swap_n = sum(1 for s in tail if s["swap"] > 0)
        lowconf_n = sum(1 for s in tail if s["conf_min"] < 0.5)
        deep_n = sum(1 for s in tail if s["gt"] < 110)
        small_n = sum(1 for s in tail if s["scale"] < 300)
        print(f"  有 swap 标记 {swap_n}（{swap_n / len(tail):.0%}），"
              f"conf<0.5 {lowconf_n}（{lowconf_n / len(tail):.0%}），"
              f"gt深蹲(<110°) {deep_n}（{deep_n / len(tail):.0%}），"
              f"人体尺度过小(<300px) {small_n}（{small_n / len(tail):.0%}）")
        worst = sorted(tail, key=lambda s: -s["err"])[:10]
        print("  最差 10 例：")
        for s in worst:
            print(f"    {s['key']}: err={s['err']:.0f}° gt={s['gt']:.0f}° swap={s['swap']} "
                  f"conf_min={s['conf_min']:.2f} 关节误差={ {k: round(v) for k, v in s['trip'].items()} }")
    return 0


if __name__ == "__main__":
    sys.exit(main())
