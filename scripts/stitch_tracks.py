"""缝合断裂的主轨迹（抢救路径）：开头被跟踪器串到背景人物时，从备用轨迹借回正确片段。

背景（2026-10-19，video.mp4）：Sports2D 的跟踪器按逐帧检测串联人物，主角开局
若小或被雾遮，轨迹要几秒后才"接上"主角 —— person02 前 2.7 秒的关键点其实落在
右侧远景滑雪者身上，主角当时在 person01 轨迹里；84 帧起 person02 才干净贴住主角。
select_person 是全视频统一选一条轨迹，救不了段级串轨，只能在此缝合。

缝合计划（帧号由 hip 位置 + 包围盒尺寸跳变人工核对，见 tmp_frames/cut_*.png）：
  - 0–81    用 person01（稳定贴主角，hip x≈600-612）
  - 82–83   两轨都不干净，按 81 与 84 两帧逐关键点线性补插
  - 84–末尾 用 person02（干净贴主角）

用法：python scripts/stitch_tracks.py [level]
产物：stitched trc（可再用 skibot render 重渲）+ 覆盖 report.json + 标注视频。
评估走 build_report，与 skibot.assess 同一条流程（同 DECISIONS 10 的对账要求），
唯一差别是视角改按完整信号段判定（见 classify_view_clean）。
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from skibot import metrics as M  # noqa: E402
from skibot.reference import load as load_refs  # noqa: E402
from skibot.render import render_annotated  # noqa: E402
from skibot.report import build_report, save_json  # noqa: E402
from skibot.trc_io import keypoints_halpe, parse_trc  # noqa: E402
from skibot.view import ViewResult, classify_view  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "results" / "video"
S2D_DIR = WORK / "video_blurred_Sports2D"
BLURRED = WORK / "video_blurred.mp4"
ANNOTATED = WORK / "video_annotated.mp4"

TRC_PREFIX = "video_blurred_Sports2D_px_"
MAIN_PID, DONOR_PID = "person02", "person01"
DONOR_LAST, MAIN_FIRST = 81, 84  # 供体用到 81（含），主轨 84 起干净；82-83 补插


def load_kp(pid: str) -> np.ndarray:
    names, xyz = parse_trc(S2D_DIR / f"{TRC_PREFIX}{pid}.trc")
    return keypoints_halpe(names, xyz)


def stitch(kp_main: np.ndarray, kp_donor: np.ndarray) -> np.ndarray:
    """0–DONOR_LAST 取供体，MAIN_FIRST 起取主轨，中间两帧线性补插。"""
    merged = kp_main.copy()
    merged[: DONOR_LAST + 1] = kp_donor[: DONOR_LAST + 1]
    a, b = kp_donor[DONOR_LAST], kp_main[MAIN_FIRST]
    both = (a[:, 2] > 0) & (b[:, 2] > 0)
    for f in range(DONOR_LAST + 1, MAIN_FIRST):
        t = (f - DONOR_LAST) / (MAIN_FIRST - DONOR_LAST)
        merged[f, both, :2] = a[both, :2] + t * (b[both, :2] - a[both, :2])
        merged[f, both, 2] = 1.0
    return merged


def write_trc(kp: np.ndarray, src_trc: Path, out_trc: Path) -> Path:
    """沿用主轨 trc 的表头与 Frame#/Time 列，只替换坐标区。缺点写空单元格。"""
    n_markers = kp.shape[1]
    lines = src_trc.read_text(encoding="utf-8").splitlines()
    out, f = [], 0
    for ln in lines:
        cells = ln.split("\t")
        try:
            int(cells[0])
        except (ValueError, IndexError):
            out.append(ln)  # 表头行原样保留
            continue
        vals = []
        for j in range(n_markers):
            x, y, c = kp[f, j]
            vals += [f"{x:.6f}" if c > 0 else "", f"{y:.6f}" if c > 0 else "", ""]
        f += 1
        out.append("\t".join(cells[:2] + vals))
    if f != kp.shape[0]:
        raise ValueError(f"src trc 数据行 {f} != kp 帧 {kp.shape[0]}")
    out_trc.write_text("\n".join(out) + "\n", encoding="utf-8")
    return out_trc


def classify_view_clean(kp: np.ndarray) -> ViewResult:
    """视角按完整信号段（MAIN_FIRST 起）判定。

    开头段肘/腕关键点全部缺失，view 的"左右置信度不对称"信号（侧面最强的
    一票）失效，剩余肩线/鼻偏信号对该姿态全投正面票，会把本就贴着 0.4
    边界的全片判定翻成 ambiguous。主体干净段信号齐全，判定可信，
    detail 里注明这一人工干预。
    """
    vr = classify_view(kp[MAIN_FIRST:])
    vr.detail["note"] = (f"按 {MAIN_FIRST} 帧后的完整信号段判定；"
                         "开头段肘腕缺失、侧面信号不可用，不参与投票")
    return vr


def main() -> int:
    level = sys.argv[1] if len(sys.argv) > 1 else "advanced"
    kp_main, kp_donor = load_kp(MAIN_PID), load_kp(DONOR_PID)
    if kp_main.shape != kp_donor.shape:
        raise ValueError(f"两轨形状不一致: {kp_main.shape} vs {kp_donor.shape}")

    kp = stitch(kp_main, kp_donor)
    trc = write_trc(kp, S2D_DIR / f"{TRC_PREFIX}{MAIN_PID}.trc",
                    S2D_DIR / f"{TRC_PREFIX}stitched.trc")
    print(f"[stitch] 缝合轨迹：{DONOR_PID} 0-{DONOR_LAST} 帧 + 补插 "
          f"{DONOR_LAST + 1}-{MAIN_FIRST - 1} 帧 + {MAIN_PID} {MAIN_FIRST} 帧起")
    print(f"[stitch] trc: {trc}")

    refs = load_refs(ROOT / "reference_values.toml")
    # 与 skibot.assess 同一条流程，唯一差别：视角用完整信号段判定（见函数 docstring）
    vr = classify_view_clean(kp)
    report = build_report("video.mp4", vr, M.compute_series(kp), refs, level,
                          person={"id": 2, "selection":
                                  f"轨迹缝合：主体 {MAIN_PID}，开头 0-{DONOR_LAST} 帧借用 "
                                  f"{DONOR_PID}（{MAIN_PID} 前 2.7 秒串到右侧背景人物）"})
    v = report["view"]
    print(f"[stitch] 视角: {v['detected']} (score={v['score']:.2f}, {v['n_valid_frames']} 帧)")
    for m in report["metrics"]:
        print(f"[stitch] {m['metric']}: 中位 {m['median_deg']}°, "
              f"有效 {m['n_valid_frames']} 帧, 区间内 {m['pct_in_range']}")
    save_json(report, WORK / "report.json")
    print(f"[stitch] 报告: {WORK / 'report.json'}")

    anno = render_annotated(BLURRED, kp, refs, level, v["detected"], ANNOTATED)
    print(f"[stitch] 标注视频: {anno}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
