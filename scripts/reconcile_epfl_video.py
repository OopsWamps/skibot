"""EPFL 视频级对账：完整 skibot 管线（Sports2D 跟踪 + 时域平滑）vs 人工标注。

与 reconcile_epfl.py（逐帧单图、无时域信息、最严苛）互补：这里跑真实管线，
标注帧 i ↔ trc 第 i 帧直接对齐，并对比"原始序列"与"插值+Butterworth 平滑后"
的角度误差，量化时域滤波的收益。选人不用 select_person（那是无标注场景的
启发式），而是选与标注平均距离最小的轨迹——对账只关心"给定选人正确时，
管线还剩多少误差"。

用法：python scripts/reconcile_epfl_video.py [--only vid/clip,...] [--limit N]
跑完每条 clip 立即落盘累计结果（out.json），中断不丢进度。
"""

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from skibot.backend import discover_tracks, run_sports2d  # noqa: E402
from skibot.epfl import DATASET_DIR, EPFL_TO_HALPE, LABELS_JSON, VIDEOS_DIR, gt_keypoints  # noqa: E402
from skibot.metrics import METRICS, angle_at, compute_series  # noqa: E402
from skibot.skeleton import CONF_MIN  # noqa: E402
from skibot.trc_io import keypoints_halpe, parse_trc  # noqa: E402

LABELS = ROOT / LABELS_JSON
VIDS = ROOT / VIDEOS_DIR


def smooth(series: np.ndarray, fps: float, cutoff: float = 6.0) -> np.ndarray:
    """NaN 插值 + 4 阶 Butterworth 低通（Sports2D 默认量级），全长返回。

    序列太短时 filtfilt 的 padlen(=15) 放不下：退化成 3 点滑动平均。
    """
    valid = ~np.isnan(series)
    if valid.sum() < 5:
        return series
    idx = np.arange(series.size)
    filled = np.interp(idx, idx[valid], series[valid])
    w = 3
    if series.size > 15:
        try:
            from scipy.signal import butter, filtfilt
            b, a = butter(4, cutoff / (fps / 2.0))
            return filtfilt(b, a, filled)
        except ImportError:
            w = 5
    pad = np.pad(filled, w // 2, mode="edge")
    return np.convolve(pad, np.ones(w) / w, mode="valid")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="逗号分隔 vid/clip，只跑这些")
    ap.add_argument("--limit", type=int, default=0, help="最多跑多少条 clip（0=全部）")
    ap.add_argument("--work", type=Path, default=ROOT / DATASET_DIR / "video_reconcile")
    ap.add_argument("--out", type=Path, default=ROOT / DATASET_DIR / "reconcile_video_report.json")
    args = ap.parse_args()

    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    jobs = []
    for vid, splits in sorted(labels.items()):
        for split, imgs in sorted(splits.items()):
            jobs.append((vid, split, imgs))
    jobs.sort(key=lambda j: -len(j[2]))  # 标注帧多的 clip 先跑，中断时信息损失最小
    if args.only:
        want = {tuple(s.split("/")) for s in args.only.split(",")}
        jobs = [j for j in jobs if (j[0], j[1]) in want]
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"[video-reconcile] 待跑 {len(jobs)} 条 clip")

    # 断点续跑：已有 per-clip 结果直接载入误差样本，汇总不丢已完成 clip
    per_clip_path = args.out.with_suffix(".per_clip.jsonl")
    errs_raw: dict[str, list[float]] = {m: [] for m in METRICS}
    errs_filt: dict[str, list[float]] = {m: [] for m in METRICS}
    joint_errs: list[float] = []
    done = {}
    if per_clip_path.exists():
        for line in per_clip_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[(r["vid"], r["clip"])] = r
            for m in METRICS:
                for store, key in ((errs_raw, "raw"), (errs_filt, "filt")):
                    v = r.get(key, {}).get(m, [])
                    if isinstance(v, list):  # 早期行只存了计数，无法恢复样本
                        store[m].extend(v)
            joint_errs.extend(r.get("joints", []))
        jobs = [j for j in jobs if (j[0], j[1]) not in done]
        print(f"[video-reconcile] 已有 {len(done)} 条结果（含误差样本），续跑剩余 {len(jobs)} 条")

    import cv2
    fh = per_clip_path.open("a", encoding="utf-8")

    for n, (vid, split, imgs) in enumerate(jobs):
        vpath = VIDS / vid / f"{split}.mp4"
        if not vpath.exists():
            print(f"[video-reconcile] 缺视频，跳过 {vid}/{split}")
            continue
        cap = cv2.VideoCapture(str(vpath))
        w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()

        work = args.work / vid / split
        work.mkdir(parents=True, exist_ok=True)
        # Sports2D 产物已在盘上就直接复用（重跑/续跑不用再推理）
        try:
            tracks = discover_tracks(work)
            print(f"[video-reconcile] ({n + 1}/{len(jobs)}) {vid}/{split} 复用已有 trc", flush=True)
        except FileNotFoundError:
            print(f"[video-reconcile] ({n + 1}/{len(jobs)}) {vid}/{split} → Sports2D ...", flush=True)
            run_sports2d(vpath, work)
            try:
                tracks = discover_tracks(work)
            except FileNotFoundError as e:
                print(f"[video-reconcile] 失败：{e}")
                continue
        kps = {pid: keypoints_halpe(*parse_trc(p)) for pid, p in tracks.items()}
        n_vid_frames = max(int(r["frame_idx"]) for r in imgs.values()) + 1

        items = sorted(imgs.items(), key=lambda kv: kv[1]["frame_idx"])
        gt = gt_keypoints(items, w, h, n_frames=n_vid_frames)
        gt_series = compute_series(gt)

        # 选轨迹：与标注平均关节距离最小者（向量化：race 广角 clip 可检出 20+ 人）
        # 0.5 是 GT 可见度门槛（标注 vis∈{0,1}，>=0.5 即标注可见），
        # 与模型置信度的 CONF_MIN 是两套契约，不互换
        gt_pts, gt_ok = gt[:, :, :2], gt[:, :, 2] >= 0.5

        def traj_dist(pid: int) -> float:
            k = kps[pid]
            n = min(k.shape[0], gt.shape[0])
            ok = gt_ok[:n] & (k[:n, :, 2] >= CONF_MIN)
            d = np.linalg.norm(k[:n, :, :2] - gt_pts[:n], axis=2)
            vals = d[ok]
            return float(vals.mean()) if vals.size else 1e9

        cand = sorted(((traj_dist(p), p) for p in kps))
        if not cand or cand[0][0] >= 1e9:
            print("[video-reconcile] 无可用轨迹与标注重叠")
            continue
        best_err, pid = cand[0]
        kp = kps[pid]
        our_series = compute_series(kp)
        our_smooth = {m: smooth(s, fps) for m, s in our_series.items()}

        clip_rows = {"vid": vid, "clip": split, "person": pid,
                     "n_tracks": len(kps), "traj_px": round(best_err, 1),
                     "n_cmp": 0, "joints": [],
                     "raw": {m: [] for m in METRICS}, "filt": {m: [] for m in METRICS}}
        for iid, rec in items:
            f = rec["frame_idx"]
            if f >= kp.shape[0]:
                continue
            for m in METRICS:
                ov, gv = our_series[m][f], gt_series[m][f]
                if not (np.isnan(ov) or np.isnan(gv)):
                    clip_rows["raw"][m].append(abs(float(ov - gv)))
                if not (np.isnan(our_smooth[m][f]) or np.isnan(gv)):
                    clip_rows["filt"][m].append(abs(float(our_smooth[m][f] - gv)))
            for jname, hidx in EPFL_TO_HALPE.items():
                # gt 侧 0.5 = 二值 GT 可见度（另一套契约），kp 侧 CONF_MIN = 模型置信度
                if gt[f, hidx, 2] >= 0.5 and kp[f, hidx, 2] >= CONF_MIN:
                    e = float(np.linalg.norm(kp[f, hidx, :2] - gt[f, hidx, :2]))
                    joint_errs.append(e)
                    clip_rows["joints"].append(e)
        clip_rows["n_cmp"] = len(clip_rows["raw"]["knee_flexion"])
        for m in METRICS:
            errs_raw[m].extend(clip_rows["raw"][m])
            errs_filt[m].extend(clip_rows["filt"][m])
        fh.write(json.dumps(clip_rows) + "\n")
        fh.flush()
        print(f"[video-reconcile] {vid}/{split}: person {pid}/{len(kps)} 轨迹, "
              f"轨迹距标注中位 {best_err:.0f}px, 对比帧 {clip_rows['n_cmp']}", flush=True)

    fh.close()

    def pct(arr, q):
        return float(np.percentile(arr, q)) if len(arr) else float("nan")

    je = np.array(joint_errs)
    summary = {"n_clips_done_before": len(done), "joint_err_px": {
        "n": int(je.size), "median": pct(je, 50), "p90": pct(je, 90)} if je.size else None,
        "angles": {}}
    print(f"\n[video-reconcile] 关节误差 px：n={je.size} 中位 {pct(je, 50):.1f} p90 {pct(je, 90):.1f}")
    print(f"{'指标':<20}{'n':>6}{'原始中位':>10}{'原始p90':>9}{'≤5°':>7}"
          f"{'平滑中位':>10}{'平滑p90':>9}{'≤5°':>7}")
    for m in METRICS:
        r, s = np.array(errs_raw[m]), np.array(errs_filt[m])
        r5 = float((r <= 5).mean()) if r.size else float("nan")
        s5 = float((s <= 5).mean()) if s.size else float("nan")
        summary["angles"][m] = {
            "raw": {"n": int(r.size), "median": pct(r, 50), "p90": pct(r, 90), "pct_within_5deg": round(r5, 3)},
            "smooth": {"n": int(s.size), "median": pct(s, 50), "p90": pct(s, 90), "pct_within_5deg": round(s5, 3)},
        }
        print(f"{m:<20}{r.size:>6}{pct(r, 50):>10.1f}{pct(r, 90):>9.1f}{r5:>7.0%}"
              f"{pct(s, 50):>10.1f}{pct(s, 90):>9.1f}{s5:>7.0%}")
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[video-reconcile] 汇总已写出：{args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
