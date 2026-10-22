"""EPFL Ski2DPose 对账：人工标注 vs 我们管线的 RTMPose 关键点与角度。

放行标准（docs/DECISIONS.md 第 10 条）：关键点抽查 + 角度误差 ±5°。
对每张标注图：跑 rtmlib（与 Sports2D 同款 balanced 模式），与人工标注逐关节比
距离（像素 + 按人体尺度归一），并对比屈髋/膝弯角度的绝对差。

除汇总外还落盘逐样本明细（.detail.jsonl：每关节误差、每角度 gt/got、
左右腿 swap 检测标记）和膝弯误差最差帧的叠加图（tail_overlays/），
供尾部归因用，避免二次推理。

用法：python scripts/reconcile_epfl.py [--max N]
默认 --images datasets/epfl_ski2dpose/images --labels .../ski2dpose_labels.json
"""

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skibot.epfl import DATASET_DIR, EPFL_JOINTS, EPFL_TO_HALPE, IMAGES_DIR, LABELS_JSON  # noqa: E402
from skibot.metrics import angle_at  # noqa: E402
from skibot.skeleton import CONF_MIN  # noqa: E402

# 角度对账：(顶点, 端点1, 端点2) 的 EPFL 关节名，名称与 metrics.py 的约定一致
ANGLE_TRIPLETS = {
    "hip_flexion_right": ("hip_right", "shoulder_right", "knee_right"),
    "hip_flexion_left": ("hip_left", "shoulder_left", "knee_left"),
    "knee_flexion_right": ("knee_right", "hip_right", "ankle_right"),
    "knee_flexion_left": ("knee_left", "hip_left", "ankle_left"),
}


# swap 检测：预测的 X 侧关节若离标注的另一半更近，判为左右腿交换
SWAP_JOINTS = ["hip_right", "knee_right", "ankle_right",
               "hip_left", "knee_left", "ankle_left"]


def draw_overlay(img, epfl_full, ours, key, err):
    """绿=人工标注骨架，红=模型预测，用于最差帧人工复核。

    epfl_full 是 24x3（含 vis 列），ours 是 26x3 halpe（未检出为 NaN）。
    """
    import cv2
    gt = np.full((24, 2), np.nan)
    for j in range(24):
        if epfl_full[j][2] == 1:
            gt[j] = epfl_full[j, :2]
    for tri in ANGLE_TRIPLETS.values():
        ia, ib, ic = (EPFL_JOINTS.index(n) for n in tri)
        ha, hb, hc = (EPFL_TO_HALPE[n] for n in tri)
        if not np.isnan(gt[[ia, ib, ic], 0]).any():
            cv2.line(img, tuple(gt[ia].astype(int)), tuple(gt[ib].astype(int)), (80, 220, 80), 2)
            cv2.line(img, tuple(gt[ib].astype(int)), tuple(gt[ic].astype(int)), (80, 220, 80), 2)
        if not np.isnan(ours[[ha, hb, hc], 0]).any():
            cv2.line(img, tuple(ours[ha, :2].astype(int)), tuple(ours[hb, :2].astype(int)), (60, 60, 230), 2)
            cv2.line(img, tuple(ours[hb, :2].astype(int)), tuple(ours[hc, :2].astype(int)), (60, 60, 230), 2)
    for jname, hidx in EPFL_TO_HALPE.items():
        j = EPFL_JOINTS.index(jname)
        if not np.isnan(gt[j, 0]):
            cv2.circle(img, tuple(gt[j].astype(int)), 4, (80, 220, 80), -1)
        if not np.isnan(ours[hidx, 0]):
            cv2.circle(img, tuple(ours[hidx, :2].astype(int)), 3, (60, 60, 230), -1)
    cv2.putText(img, f"{key} knee_err={err:.0f}deg", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=ROOT / IMAGES_DIR)
    ap.add_argument("--labels", type=Path, default=ROOT / LABELS_JSON)
    ap.add_argument("--out", type=Path,
                    default=ROOT / DATASET_DIR / "reconcile_report.json")
    ap.add_argument("--max", type=int, default=0, help="最多对账多少张（0=全部）")
    ap.add_argument("--overlays", type=int, default=24,
                    help="膝弯误差最差 N 帧输出叠加图（0=不输出）")
    args = ap.parse_args()

    from rtmlib import BodyWithFeet  # 与 Sports2D balanced 同款
    body = BodyWithFeet(mode="balanced")

    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    jobs = []
    for vid, splits in labels.items():
        for split, imgs in splits.items():
            for iid, rec in imgs.items():
                p = args.images / vid / str(int(split)) / f"{iid}.webp"
                if not p.exists():  # 兼容解包层级差异
                    hits = list(args.images.rglob(f"{iid}.webp"))
                    if not hits:
                        continue
                    p = hits[0]
                jobs.append((p, rec))
    jobs.sort()
    if args.max:
        jobs = jobs[: args.max]
    print(f"[reconcile] 待对账 {len(jobs)} 张")
    if not jobs:
        print("[reconcile] 没找到图片，确认 --images 目录与解包结构", file=sys.stderr)
        return 1

    joint_errs, joint_errs_norm, angle_errs = [], [], {k: [] for k in ANGLE_TRIPLETS}
    detail_rows: list[dict] = []
    worst_knee: list[tuple[float, str, Path]] = []  # (膝弯最大误差, key, 图片路径)
    kept_frames: dict[str, tuple] = {}  # key -> (epfl, ours) 供叠加图绘制
    n_done = 0
    for i, (p, rec) in enumerate(jobs):
        import cv2
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        epfl = np.array(rec["annotation"], dtype=float)  # (24,3) 归一化
        epfl[:, 0] *= w
        epfl[:, 1] *= h

        kps, scores = body(img)
        kps, scores = np.asarray(kps), np.asarray(scores)  # (N,26,2), (N,26)
        if kps.shape[0] == 0:
            continue
        # 一张图可能检出多人：选与人工标注平均距离最小的候选（对应产品的选人环节）
        def cand_dist(ci: int) -> float:
            ds = []
            for jname, hidx in EPFL_TO_HALPE.items():
                jj = EPFL_JOINTS.index(jname)
                if rec["annotation"][jj][2] == 1:
                    ds.append(float(np.linalg.norm(kps[ci, hidx] - epfl[jj, :2])))
            return float(np.mean(ds)) if ds else 1e9
        ci = int(np.argmin([cand_dist(c) for c in range(kps.shape[0])]))
        ours = np.full((26, 3), np.nan)
        ours[:, :2] = kps[ci]
        ours[:, 2] = scores[ci]

        vis_pts = epfl[[EPFL_JOINTS.index(j) for j in EPFL_TO_HALPE if rec["annotation"][EPFL_JOINTS.index(j)][2] == 1], :2] \
            if any(rec["annotation"][EPFL_JOINTS.index(j)][2] == 1 for j in EPFL_TO_HALPE) else np.zeros((1, 2))
        scale = float(np.linalg.norm(vis_pts.max(0) - vis_pts.min(0))) or 1.0

        row: dict = {"key": p.stem, "scale_px": round(scale, 1),
                     "joints": {}, "swaps": {}, "angles": {}}
        jidx = {n: EPFL_JOINTS.index(n) for n in SWAP_JOINTS}
        for jname, hidx in EPFL_TO_HALPE.items():
            j = EPFL_JOINTS.index(jname)
            if rec["annotation"][j][2] != 1 or ours[hidx, 2] < CONF_MIN:
                continue
            e = float(np.linalg.norm(ours[hidx, :2] - epfl[j, :2]))
            joint_errs.append(e)
            joint_errs_norm.append(e / scale)
            row["joints"][jname] = {"err_px": round(e, 1), "conf": round(float(ours[hidx, 2]), 2)}
        for jn in SWAP_JOINTS:
            h = EPFL_TO_HALPE[jn]
            if ours[h, 2] < CONF_MIN:
                continue
            other = jn.replace("_right", "_left").replace("_left", "_right")
            ho = EPFL_TO_HALPE[other]
            d_own = float(np.linalg.norm(ours[h, :2] - epfl[jidx[jn], :2]))
            d_oth = float(np.linalg.norm(ours[h, :2] - epfl[jidx[other], :2]))
            if d_oth < d_own:
                row["swaps"][jn] = round(d_own - d_oth, 1)

        for aname, (v, a, b) in ANGLE_TRIPLETS.items():
            iv, ia, ib = (EPFL_JOINTS.index(x) for x in (v, a, b))
            if min(rec["annotation"][iv][2], rec["annotation"][ia][2], rec["annotation"][ib][2]) != 1:
                continue
            hv, ha, hb = EPFL_TO_HALPE[v], EPFL_TO_HALPE[a], EPFL_TO_HALPE[b]
            if min(ours[hv, 2], ours[ha, 2], ours[hb, 2]) < CONF_MIN:
                continue
            gt = angle_at(epfl[iv], epfl[ia], epfl[ib])
            got = angle_at(ours[hv], ours[ha], ours[hb])
            if not (np.isnan(gt) or np.isnan(got)):
                err = abs(got - gt)
                angle_errs[aname].append(err)
                row["angles"][aname] = {"gt": round(gt, 1), "got": round(got, 1),
                                        "err": round(err, 1)}
                if aname.startswith("knee_flexion"):
                    worst_knee.append((err, row["key"], p))
                    kept_frames[row["key"]] = (epfl.copy(), ours.copy())
        detail_rows.append(row)

        n_done += 1
        if (i + 1) % 100 == 0:
            print(f"[reconcile] {i + 1}/{len(jobs)}")

    # 原始数据先落盘：汇总再崩也不用重跑推理
    np.savez(args.out.with_suffix(".raw.npz"), joint_errs=np.array(joint_errs),
             joint_errs_norm=np.array(joint_errs_norm),
             **{f"angle_{k}": np.array(v) for k, v in angle_errs.items()})

    detail_path = args.out.with_suffix(".detail.jsonl")
    with detail_path.open("w", encoding="utf-8") as fh:
        for r in detail_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[reconcile] 逐样本明细已写出：{detail_path}（{len(detail_rows)} 行）")

    if args.overlays and worst_knee:
        import cv2
        worst_knee.sort(key=lambda t: -t[0])
        odir = args.out.parent / "tail_overlays"
        odir.mkdir(exist_ok=True)
        for err, key, p in worst_knee[: args.overlays]:
            img = cv2.imread(str(p))
            if img is None or key not in kept_frames:
                continue
            epfl_k, ours_k = kept_frames[key]
            cv2.imwrite(str(odir / f"{key}_err{err:.0f}.jpg"),
                        draw_overlay(img, epfl_k, ours_k, key, err))
        print(f"[reconcile] 最差 {min(args.overlays, len(worst_knee))} 帧叠加图：{odir}")

    def pct(arr, q):
        return float(np.percentile(arr, q)) if len(arr) else float("nan")

    je, jen = np.array(joint_errs), np.array(joint_errs_norm)
    summary = {
        "n_images": n_done,
        "n_joint_samples": len(je),
        "joint_err_px": {"median": pct(je, 50), "mean": float(je.mean()) if len(je) else None,
                         "p90": pct(je, 90), "p95": pct(je, 95)},
        "joint_err_pct_of_scale": {"median": pct(jen, 50), "p90": pct(jen, 90), "p95": pct(jen, 95)},
        "angles": {},
    }
    print(f"\n[reconcile] 图像 {n_done}，关节样本 {len(je)}")
    print(f"[reconcile] 关节误差 px: 中位 {pct(je, 50):.1f} / p90 {pct(je, 90):.1f} / p95 {pct(je, 95):.1f}")
    print(f"[reconcile] 关节误差(人体尺度%): 中位 {pct(jen, 50) * 100:.1f}% / p90 {pct(jen, 90) * 100:.1f}%")

    for aname, errs in angle_errs.items():
        e = np.array(errs)
        within5 = float((e <= 5).mean()) if len(e) else 0.0
        summary["angles"][aname] = {
            "n": len(e), "median": pct(e, 50), "p90": pct(e, 90), "p95": pct(e, 95),
            "pct_within_5deg": round(within5, 3),
        }
        print(f"[reconcile] {aname}: n={len(e)} 中位 {pct(e, 50):.1f}° "
              f"p90 {pct(e, 90):.1f}° ≤5°占比 {within5:.0%}")

    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[reconcile] 汇总已写出：{args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
