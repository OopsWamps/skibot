"""弯末关节角 vs Bon et al. 2021 文献锚点（issue 07，实拍前门）。

对已有 px trc 的教练级素材跑三维提升与相位切分，在每个弯的弯末取外侧膝、
内侧膝、外侧髋的三点夹角（180°=伸直，与 Bon 的动捕口径一致），按弯向对照
Bon 平行式（PT）的 mean±2SD，输出对比表与通过/不通过结论。

弯末定义与相位切分一致：弯是半周期（换刃过零 → 侧倾极值 → 换刃过零），
弯末取本弯最后一帧（end_frame−1，过零换刃前）；同时报告它与过零分数帧
插值采样的最大差值作稳定性检查。内外侧由侧倾峰值的符号定：正 = 向受试者
左侧倾 = 左转，此时右腿是外侧腿（与 Bon 的 Knee_R_LT 对应关系一致）。

用法：
    python scripts/lit_anchor.py <clip_dir>... [--json <out.json>]

clip_dir 是分析结果目录（含 report.json 与 <stem>_Sports2D/ 下的
*_px_person*.trc）。逐弯数值随 --json 落盘，保证可复现。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows GBK 控制台兜底

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from skibot import artifacts  # noqa: E402
from skibot.metrics3d import _frame_series  # noqa: E402  与 compute_series3d 同一条公式
from skibot.motionbert import MotionBERTLifter  # noqa: E402
from skibot.phases import segment_phases  # noqa: E402
from skibot.render import probe_video  # noqa: E402
from skibot.skeleton import CONF_MIN, L_HIP, R_HIP  # noqa: E402
from skibot.trc_io import load_kp  # noqa: E402

# Bon et al. 2021（IJERPH 18(6):3029）平行式 PT，弯末，180°=伸直，SD 为弯级
# 变异（n=90 弯/弯型）。来源：docs/lit_angle_values.md 与 2026-10-26 全文核验。
BON_PT = {
    "outer_knee": {"left_turn": (141.46, 3.70), "right_turn": (148.74, 3.46)},
    "inner_knee": {"left_turn": (115.69, 6.23), "right_turn": (117.49, 5.50)},
    "outer_hip": {"left_turn": (164.99, 4.71), "right_turn": (166.20, 2.53)},
}
MIN_TURNS_FOR_VERDICT = 3  # 该弯向的有效弯少于此数时只给数值，不下通过判定


def _per_side_angles(sk3) -> tuple[np.ndarray, np.ndarray]:
    """逐帧逐侧的三点夹角：(T,2) 膝夹角与 (T,2) 髋夹角（180°=伸直）。

    列序 [left, right]；调 metrics3d._frame_series 保证与主指标同一条
    公式与退化门槛，不另写数学。
    """
    t = sk3.n_frames
    knee = np.full((t, 2), np.nan)
    hip = np.full((t, 2), np.nan)
    for f in range(t):
        for i, side in enumerate(("left", "right")):
            row = _frame_series(sk3, f, side=side,
                                gravity_source=sk3.gravity_source,
                                slope_sag_deg=None)
            knee[f, i] = row["knee_interior_3pt"]
            hip[f, i] = row["hip_flexion_3pt"]
    return knee, hip


def _interp_at(series: np.ndarray, frame: float) -> float:
    """分数帧线性插值采样；端点夹取。"""
    t = series.shape[0]
    if frame < 0:
        frame = 0.0
    if frame > t - 1:
        frame = float(t - 1)
    lo = int(np.floor(frame))
    hi = min(lo + 1, t - 1)
    frac = frame - lo
    a, b = series[lo], series[hi]
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return float(a * (1.0 - frac) + b * frac)


def _per_side_angles_2d(kp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2D 像素坐标的同款三点夹角 (T,2)：侧面机位下 ≈ 真实矢状面角度。

    与三维逐侧夹角同公式、同置信度门槛，只把三维坐标换成像素 (x,y)。
    用于区分「提升模型扭曲角度」与「动作/时机差异」——若 2D≈3D，
    角度偏差不来自管线。
    """
    from skibot.skeleton import (
        L_ANKLE, L_KNEE, L_SHOULDER, R_ANKLE, R_KNEE, R_SHOULDER,
        SIDE_JOINTS,
    )

    def ang(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """b 为顶点的三点夹角；任一点低置信返回 NaN。"""
        if min(a[2], b[2], c[2]) < CONF_MIN:
            return float("nan")
        v1 = a[:2] - b[:2]
        v2 = c[:2] - b[:2]
        n = np.linalg.norm(v1) * np.linalg.norm(v2)
        if n < 1e-9:
            return float("nan")
        cosv = float(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))
        return float(np.degrees(np.arccos(cosv)))

    t = kp.shape[0]
    knee2d = np.full((t, 2), np.nan)
    hip2d = np.full((t, 2), np.nan)
    sides = (SIDE_JOINTS["left"], SIDE_JOINTS["right"])  # (肩,髋,膝,踝)
    for f in range(t):
        for i, (sh, hp, kn, an) in enumerate(sides):
            knee2d[f, i] = ang(kp[f, hp], kp[f, kn], kp[f, an])
            hip2d[f, i] = ang(kp[f, sh], kp[f, hp], kp[f, kn])
    return knee2d, hip2d


def analyze_clip(clip_dir: Path, lifter: MotionBERTLifter) -> dict:
    report_path = clip_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    stem = clip_dir.name
    work = clip_dir / f"{stem}_Sports2D"
    pid = (report.get("person") or {}).get("id")
    if not isinstance(pid, int):
        raise ValueError(f"{report_path} 缺 person.id，无法定位分析对象的 trc")
    trc = work / f"{stem}_Sports2D_px_person{pid:02d}.trc"
    if not trc.exists():
        # 验收门不静默换人：多人目录里换一个 trc 就是换一个结论
        raise FileNotFoundError(f"{trc} 不存在（report 选的是 person {pid}）")
    src_video = work / f"{stem}_Sports2D.mp4"
    _, _, fps = probe_video(src_video)
    if not fps:
        raise ValueError(f"{src_video} 读不出帧率")

    kp = load_kp(trc)
    sk3 = lifter.lift(kp)
    from skibot.metrics3d import compute_series3d

    series3d = compute_series3d(sk3)
    lean = series3d["trunk_lean_frontal"]
    # 与 assess.py 同款横向轨迹：2D 髋中点，置信度门控
    left2d = kp[:, L_HIP]
    right2d = kp[:, R_HIP]
    ok2d = (left2d[:, 2] >= CONF_MIN) & (right2d[:, 2] >= CONF_MIN)
    lateral = np.where(ok2d, (left2d[:, 0] + right2d[:, 0]) / 2.0, np.nan)
    seg = segment_phases(lean, fps=fps, knee=series3d["knee_flexion"],
                         lateral=lateral)

    knee, hip = _per_side_angles(sk3)
    knee2d, hip2d = _per_side_angles_2d(kp)
    turns = []
    n_guarded = 0  # 触发解剖护栏（三点角 < 40°，物理上不可能）的弯数
    max_stab_delta = 0.0
    for turn in seg.turns:
        direction = "left_turn" if turn.peak_lean_deg > 0 else "right_turn"
        outer = 1 if direction == "left_turn" else 0   # 列序 [left, right]
        inner = 1 - outer
        f_end = max(turn.start_frame, turn.end_frame - 1)
        f_peak = turn.peak_frame
        lo, hi = turn.start_frame, turn.end_frame
        row = {
            "turn_index": turn.index,
            "direction": direction,
            "peak_lean_deg": round(turn.peak_lean_deg, 2),
            "confidence": turn.confidence,
            "sample_frame": f_end,
            "outer_knee": float(knee[f_end, outer]),
            "inner_knee": float(knee[f_end, inner]),
            "outer_hip": float(hip[f_end, outer]),
            # apex（侧倾峰值帧）采样：Bon 的“重心最低点”更接近弯的 apex，
            # 换刃前最后一帧是其下界；两档并列作定义敏感性检查
            "outer_knee_at_peak": float(knee[f_peak, outer]),
            "inner_knee_at_peak": float(knee[f_peak, inner]),
            "outer_hip_at_peak": float(hip[f_peak, outer]),
            # 弯内外侧膝的最深屈曲（夹角最小值）：管线量级 sanity
            "outer_knee_min_in_turn": float(np.nanmin(knee[lo:hi, outer])),
            "inner_knee_min_in_turn": float(np.nanmin(knee[lo:hi, inner])),
            # 2D 像素同款夹角（侧面机位 ≈ 真实矢状角）：区分提升扭曲与动作差异
            "outer_knee_2d": float(knee2d[f_end, outer]),
            "inner_knee_2d": float(knee2d[f_end, inner]),
            "outer_hip_2d": float(hip2d[f_end, outer]),
            "outer_knee_2d_at_peak": float(knee2d[f_peak, outer]),
            "inner_knee_2d_at_peak": float(knee2d[f_peak, inner]),
            "outer_hip_2d_at_peak": float(hip2d[f_peak, outer]),
        }
        # 稳定性：弯末最后一帧 vs 换刃过零分数帧插值
        cross = turn.end_time_s * seg.fps
        stab = [
            abs(row["outer_knee"] - _interp_at(knee[:, outer], cross)),
            abs(row["inner_knee"] - _interp_at(knee[:, inner], cross)),
            abs(row["outer_hip"] - _interp_at(hip[:, outer], cross)),
        ]
        finite_stab = [d for d in stab if np.isfinite(d)]
        if finite_stab:
            max_stab_delta = max(max_stab_delta, max(finite_stab))
        turns.append(row)

    valid = []
    for t in turns:
        if not all(np.isfinite(t[k]) for k in
                   ("outer_knee", "inner_knee", "outer_hip")):
            continue
        # 解剖护栏：三点夹角 < 40° 物理上不可能（骨穿透），是提升/检测
        # 的垃圾帧；整弯剔除并计数，不进聚合
        sampled = [t[k] for k in ("outer_knee", "inner_knee", "outer_hip",
                                  "outer_knee_at_peak", "inner_knee_at_peak",
                                  "outer_hip_at_peak")]
        if min(sampled) < 40.0:
            n_guarded += 1
            continue
        valid.append(t)
    return {
        "clip": str(clip_dir),
        "trc": str(trc.relative_to(ROOT)) if trc.is_relative_to(ROOT) else str(trc),
        "person_id": pid,
        "fps": fps,
        "n_frames": int(kp.shape[0]),
        "n_turns_detected": len(seg.turns),
        "n_turns_discarded": len(seg.discarded),
        "n_turns_used": len(valid),
        "n_turns_guarded": n_guarded,
        "max_stability_delta_deg": round(max_stab_delta, 2),
        "weight_name": lifter.checkpoint.name,
        "turns": valid,
    }


def _agg(turns: list[dict], key: str) -> tuple[int, float, float]:
    vals = [t[key] for t in turns if np.isfinite(t[key])]
    if not vals:
        return 0, float("nan"), float("nan")
    return len(vals), float(np.mean(vals)), float(np.std(vals))


def compare(clips: list[dict], suffix: str = "") -> list[dict]:
    """按方向×指标聚合所有弯，对照 Bon PT 的 mean±2SD。

    suffix 选采样定义："" = 弯末（换刃前最后一帧），"_at_peak" = 侧倾
    峰值帧（Bon 的“重心最低点”更接近 apex）。
    """
    rows = []
    for direction in ("left_turn", "right_turn"):
        pool = [t for c in clips for t in c["turns"]
                if t["direction"] == direction]
        for metric in ("outer_knee", "inner_knee", "outer_hip"):
            n, mean, sd = _agg(pool, metric + suffix)
            ref_mean, ref_sd = BON_PT[metric][direction]
            lo, hi = ref_mean - 2 * ref_sd, ref_mean + 2 * ref_sd
            if n == 0:
                verdict = "无数据"
            elif n < MIN_TURNS_FOR_VERDICT:
                verdict = "样本不足（仅数值对照）"
            elif lo <= mean <= hi:
                verdict = "通过"
            else:
                verdict = "不通过"
            rows.append({
                "direction": direction, "metric": metric,
                "sampling": "apex" if suffix else "turn_end",
                "n_turns": n, "our_mean": None if n == 0 else round(mean, 1),
                "our_sd": None if n == 0 else round(sd, 1),
                "bon_mean": ref_mean, "bon_sd": ref_sd,
                "band_2sd": f"[{lo:.1f}, {hi:.1f}]",
                "delta": None if n == 0 else round(mean - ref_mean, 1),
                "verdict": verdict,
            })
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="弯末关节角 vs Bon et al. 2021")
    ap.add_argument("clips", nargs="+", type=Path, help="分析结果目录（含 report.json）")
    ap.add_argument("--json", type=Path, default=None, help="逐弯数值 JSON 输出路径")
    args = ap.parse_args(argv)

    lifter = MotionBERTLifter()
    clips = []
    for d in args.clips:
        print(f"[skibot] 分析 {d} ...")
        c = analyze_clip(d, lifter)
        print(f"  弯：检出 {c['n_turns_detected']}，弃 {c['n_turns_discarded']}，"
              f"护栏剔除 {c['n_turns_guarded']}，入表 {c['n_turns_used']}；"
              f"稳定性最大差 {c['max_stability_delta_deg']}°")
        clips.append(c)

    rows_end = compare(clips, "")
    rows_peak = compare(clips, "_at_peak")
    for label, rows in (("弯末采样（换刃过零前最后一帧）", rows_end),
                        ("apex 采样（侧倾峰值帧 ≈ 重心最低点）", rows_peak)):
        print(f"\n### {label}")
        print("\n| 弯向 | 指标 | n | 本文均值±SD | Bon PT 均值±SD | ±2SD 带 | 差 | 判定 |")
        print("|---|---|---|---|---|---|---|---|")
        for r in rows:
            ours = "—" if r["our_mean"] is None else f"{r['our_mean']}±{r['our_sd']}"
            print(f"| {r['direction']} | {r['metric']} | {r['n_turns']} | {ours} "
                  f"| {r['bon_mean']}±{r['bon_sd']} | {r['band_2sd']} "
                  f"| {r['delta']} | {r['verdict']} |")

    def overall_of(rows: list[dict]) -> str:
        decisive = [r for r in rows if r["verdict"] in ("通过", "不通过")]
        if not decisive:
            return "无法判定（有效弯不足）"
        return "通过" if all(r["verdict"] == "通过" for r in decisive) else "不通过"

    overall_end = overall_of(rows_end)
    overall_peak = overall_of(rows_peak)
    print(f"\n[skibot] 总体判定：弯末采样 {overall_end}；apex 采样 {overall_peak}")

    # 2D/3D 一致性：|三维提升角 − 像素投影角| 的均值（侧面机位下像素角
    # 近似真值；差值小说明偏差来自动作与时机，不来自提升管线）
    agree = {}
    all_turns = [t for c in clips for t in c["turns"]]
    for base in ("outer_knee", "inner_knee", "outer_hip"):
        for suffix, tag in (("", "turn_end"), ("_at_peak", "apex")):
            pairs = [(t[base + suffix], t[base + "_2d" + suffix])
                     for t in all_turns
                     if np.isfinite(t[base + suffix]) and np.isfinite(t[base + "_2d" + suffix])]
            if pairs:
                diffs = [abs(a - b) for a, b in pairs]
                agree[f"{base}_{tag}"] = {"n": len(pairs),
                                          "mean_abs_3d_minus_2d_deg": round(float(np.mean(diffs)), 1)}
    for k, v in agree.items():
        print(f"[skibot] 2D/3D 一致性 {k}: |Δ| 均值 {v['mean_abs_3d_minus_2d_deg']}°（n={v['n']}）")

    if args.json:
        payload = {"clips": clips, "comparison_turn_end": rows_end,
                   "comparison_apex": rows_peak,
                   "overall_turn_end": overall_end,
                   "overall_apex": overall_peak,
                   "agreement_2d_3d": agree,
                   "bon_pt_source": "Bon et al. 2021, IJERPH 18(6):3029, PT 弯末，180°=伸直"}

        def _strict(obj):
            """NaN/Inf → None，保证 JSON 严格可解析。"""
            if isinstance(obj, float) and not np.isfinite(obj):
                return None
            if isinstance(obj, dict):
                return {k: _strict(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_strict(v) for v in obj]
            return obj

        args.json.parent.mkdir(parents=True, exist_ok=True)
        artifacts.write_text_atomic(
            args.json, json.dumps(_strict(payload), ensure_ascii=False, indent=1,
                                   allow_nan=False))
        print(f"[skibot] 逐弯数值已写出：{args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
