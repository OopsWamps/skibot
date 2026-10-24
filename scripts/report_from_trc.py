"""从已生成的 px trc 直接出报告，跳过 Sports2D 的收尾阶段。

用法：python scripts/report_from_trc.py <trc路径> <输出json> [level]
走 skibot.finalize.assess_and_publish 与主路径同一条评估收尾（一期 #10 的
量角对账靠它，行为必须与主路径一致）；只落报告，不出标注视频。
"""

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from skibot import artifacts  # noqa: E402
from skibot.finalize import assess_and_publish  # noqa: E402
from skibot.reference import load as load_refs  # noqa: E402
from skibot.trc_io import load_kp  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def report_from_trc(trc: Path, out_json: Path, level: str = "intermediate"):
    kp = load_kp(trc)  # (frames, 26, 3): [x, y, conf]
    mapped = [i for i in range(26) if not np.all(np.isnan(kp[:, i, 0]))]
    print(f"[salvage] Halpe 映射命中 {len(mapped)}/26: {mapped}")

    refs = load_refs(ROOT / "reference_values.toml")
    # 与主路径同一条评估收尾（一期 #10 的量角对账靠它，行为必须一致）；
    # 不给 video_path：抢救路径只落报告，不出标注视频
    res = assess_and_publish(kp, None, refs=refs, level=level, label=str(trc),
                             report_path=out_json)
    report = res.report

    v = report["view"]
    print(f"[salvage] 视角: {v['detected']} (score={v['score']:.2f}, {v['n_valid_frames']} 帧)")
    for m in report["metrics"]:
        if m["n_valid_frames"]:
            print(f"[salvage] {m['metric']}: 中位 {m['median_deg']}°, "
                  f"有效 {m['n_valid_frames']} 帧, 区间内 {m['pct_in_range']}")
        else:
            print(f"[salvage] {m['metric']}: 全部无效")

    print(f"[salvage] 报告: {out_json}")
    return report


if __name__ == "__main__":
    trc = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else artifacts.report_path(trc.parent)
    level = sys.argv[3] if len(sys.argv) > 3 else "intermediate"
    rep = report_from_trc(trc, out, level)
    print(json.dumps(rep["metrics"], ensure_ascii=False, indent=1))
