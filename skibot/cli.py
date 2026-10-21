"""命令行入口：skibot analyze <video...> / skibot render

流程（docs/DECISIONS.md 第 4 条）：打码 → Sports2D 管线（backend）→ 主轨迹
选择（select_person）→ 评估流程（assess：视角分类 → 五指标 → 参考区间 → 报告）
→ 报告落盘 + 自绘标注视频（render，只画选中者）。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows GBK 控制台兜底

from . import __version__
from .assess import assess
from .backend import DEFAULT_TRACK_ORDER, discover_tracks, run_sports2d
from .privacy import blur_faces
from .reference import load as load_refs
from .render import render_annotated
from .report import save_json
from .select_person import sample_pants_pixels, select_person, size_score
from .skeleton import (
    L_ANKLE, L_HIP, L_KNEE, L_SHOULDER, R_ANKLE, R_HIP, R_KNEE, R_SHOULDER,
)
from .trc_io import keypoints_halpe, parse_trc
from .view import classify_view_frames

DEFAULT_REFS = Path(__file__).resolve().parent.parent / "reference_values.toml"


@dataclass
class AnalyzeOptions:
    """一次分析的全部选项。新增 CLI 旗标只动这里与 argparse，调用点不动。"""
    level: str = "intermediate"
    track_order: str = DEFAULT_TRACK_ORDER
    blur: bool = True
    result_dir: Path = Path("results")
    refs: Path = DEFAULT_REFS
    color: str | None = None
    person: int | None = None


def analyze_one(video: Path, opts: AnalyzeOptions) -> Path:
    work = opts.result_dir / video.stem
    work.mkdir(parents=True, exist_ok=True)

    src = video
    if opts.blur:
        src, n_faces = blur_faces(video, work / f"{video.stem}_blurred.mp4")
        print(f"[skibot] 人脸打码：检出 {n_faces} 处（Haar 召回有限，雪镜/面罩场景见 privacy.py 说明）")

    print(f"[skibot] Sports2D 分析中：{src}")
    run_sports2d(src, work, opts.track_order)

    tracks = discover_tracks(work)
    kps = {pid: keypoints_halpe(*parse_trc(p)) for pid, p in tracks.items()}
    print(f"[skibot] 检测到 {len(kps)} 个人物轨迹：{sorted(kps)}")

    if opts.person is not None:
        if opts.person not in kps:
            raise ValueError(f"指定了 person {opts.person}，但只有 {sorted(kps)}")
        chosen, reason = opts.person, f"人工指定 person {opts.person}"
    else:
        colors_by_pid = None
        if opts.color:
            colors_by_pid = {pid: sample_pants_pixels(src, kp) for pid, kp in kps.items()}
        sel = select_person(kps, colors_by_pid=colors_by_pid, color_name=opts.color)

        hdr = (f"{'pid':>4} {'有效帧':>6} {'尺寸px':>8} {'位移px':>9}"
               f" {'尺寸分':>6} {'位移分':>6}")
        if opts.color:
            hdr += f" {'颜色分':>6}"
        print(f"[skibot] 候选打分：\n{hdr}")
        for r in sel.rows:
            line = (f"{r.pid:>4} {r.n_valid:>6} {r.size_px:>8.1f} {r.disp_px:>9.1f}"
                    f" {r.size_n:>6.3f} {r.disp_n:>6.3f}")
            if r.color is not None:
                line += f" {r.color:>6.3f}"
            line += f"  score={r.score}"
            print(line)
        if sel.color_fallback:
            print(f"[skibot] 警告：颜色 '{opts.color}' 最高得分仅 {sel.color_best}，"
                  "未匹配到可信目标，回退到尺寸+位移判定", file=sys.stderr)
        chosen, reason = sel.pid, sel.reason
    print(f"[skibot] 选定 person {chosen}：{reason}")

    kp = kps[chosen]
    print(f"[skibot] 关键点：{tracks[chosen].name}，{kp.shape[0]} 帧")

    # 质量统计（DECISIONS 5 缓解项）：人物高度占比 + 核心点缺失帧占比
    # （trc 的 conf 是 0/1 存在标记，判核心 8 点而非全部 26 点——雪场远景
    #  缺脸/手脚趾很常见，全点口径会让提示常驻失去区分度）
    cap = cv2.VideoCapture(str(src))
    frame_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    quality: dict = {}
    if frame_h:
        quality["person_height_ratio"] = round(size_score(kp) / frame_h, 3)
    core = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]
    core_ok = (kp[:, core, 2] > 0).mean(axis=1)
    quality["low_conf_frame_ratio"] = round(float((core_ok < 0.75).mean()), 3)

    refs = load_refs(opts.refs)
    frame_view = classify_view_frames(kp)
    report = assess(kp, level=opts.level, refs=refs, label=str(video),
                    person={"id": chosen, "selection": reason}, quality=quality)
    print(f"[skibot] 视角判定：{report['view']['detected']}"
          f"（score={report['view']['score']:.2f}，{report['view']['n_valid_frames']} 有效帧）")

    out = save_json(report, work / "report.json")
    print(f"[skibot] 报告已写出：{out}")

    anno = render_annotated(src, kp, refs, opts.level, report["view"]["detected"],
                            work / f"{video.stem}_annotated.mp4", frame_view=frame_view)
    print(f"[skibot] 标注视频（自绘，仅选中者）：{anno}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="skibot", description="滑雪视频体态分析")
    parser.add_argument("--version", action="version", version=f"skibot {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="分析一段或多段滑雪视频")
    a.add_argument("videos", nargs="+", type=Path, help="视频文件路径")
    a.add_argument("--level", choices=("novice", "intermediate", "advanced"),
                   default="intermediate", help="滑雪水平档，决定参考区间")
    a.add_argument("--track-order", default=DEFAULT_TRACK_ORDER,
                   help=f"Sports2D 内部人物排序策略（默认 {DEFAULT_TRACK_ORDER}）；最终选人用 --color/--person")
    a.add_argument("--color", default=None,
                   help="按衣着颜色锁定主角，如 --color pink 或 --color #ff69b4")
    a.add_argument("--person", type=int, default=None,
                   help="直接指定 person 编号（跳过自动选人）")
    a.add_argument("--no-blur", action="store_true", help="关闭人脸打码（默认开）")
    a.add_argument("--result-dir", type=Path, default=Path("results"), help="输出目录")
    a.add_argument("--refs", type=Path, default=DEFAULT_REFS,
                   help="参考区间表路径")

    r = sub.add_parser("render", help="从已有 trc 重新渲染标注视频（跳过分析）")
    r.add_argument("--video", type=Path, required=True, help="画面来源视频（建议用打码后的）")
    r.add_argument("--trc", type=Path, required=True, help="px trc 文件")
    r.add_argument("--out", type=Path, required=True, help="输出 mp4 路径")
    r.add_argument("--view", choices=("sagittal", "frontal"), default="sagittal",
                   help="叠加哪套指标")
    r.add_argument("--level", choices=("novice", "intermediate", "advanced"),
                   default="intermediate", help="水平档，决定区间着色")
    r.add_argument("--refs", type=Path, default=DEFAULT_REFS,
                   help="参考区间表路径")

    args = parser.parse_args(argv)

    if args.cmd == "render":
        names, xyz = parse_trc(args.trc)
        kp = keypoints_halpe(names, xyz)
        refs = load_refs(args.refs)
        out = render_annotated(args.video, kp, refs, args.level, args.view, args.out)
        print(f"[skibot] 标注视频已写出：{out}")
        return 0

    if args.color and args.person is not None:
        print("[skibot] --color 与 --person 同给时以 --person 为准", file=sys.stderr)

    opts = AnalyzeOptions(level=args.level, track_order=args.track_order,
                          blur=not args.no_blur, result_dir=args.result_dir,
                          refs=args.refs, color=args.color, person=args.person)

    rc = 0
    for v in args.videos:
        if not v.exists():
            print(f"[skibot] 找不到视频：{v}", file=sys.stderr)
            rc = 1
            continue
        try:
            analyze_one(v, opts)
        except Exception as e:  # noqa: BLE001 —— CLI 顶层兜底，逐视频失败不中断批量
            print(f"[skibot] {v.name} 分析失败：{e}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
