"""EPFL Ski2DPose 对账：人工标注 vs 我们管线的 RTMPose 关键点与角度。

放行标准（docs/DECISIONS.md 第 10 条）：关键点抽查 + 角度误差 ±5°。
对每张标注图：跑 rtmlib（与 Sports2D 同款 balanced 模式），与人工标注逐关节比
距离（像素 + 按人体尺度归一），并对比屈髋/膝弯角度的绝对差。

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

# EPFL 24 关节顺序 → Halpe-26 索引（只对账我们有映射的点）
EPFL_JOINTS = [
    "head", "neck",
    "shoulder_right", "elbow_right", "hand_right", "pole_basket_right",
    "shoulder_left", "elbow_left", "hand_left", "pole_basket_left",
    "hip_right", "knee_right", "ankle_right",
    "hip_left", "knee_left", "ankle_left",
    "ski_tip_right", "toes_right", "heel_right", "ski_tail_right",
    "ski_tip_left", "toes_left", "heel_left", "ski_tail_left",
]
EPFL_TO_HALPE = {
    "head": 17, "neck": 18,
    "shoulder_right": 6, "elbow_right": 8, "hand_right": 10,
    "shoulder_left": 5, "elbow_left": 7, "hand_left": 9,
    "hip_right": 12, "knee_right": 14, "ankle_right": 16,
    "hip_left": 11, "knee_left": 13, "ankle_left": 15,
}
# 角度对账：(顶点, 端点1, 端点2) 的 EPFL 关节名，名称与 metrics.py 的约定一致
ANGLE_TRIPLETS = {
    "hip_flexion_right": ("hip_right", "shoulder_right", "knee_right"),
    "hip_flexion_left": ("hip_left", "shoulder_left", "knee_left"),
    "knee_flexion_right": ("knee_right", "hip_right", "ankle_right"),
    "knee_flexion_left": ("knee_left", "hip_left", "ankle_left"),
}


def angle_at(a, b, c) -> float:
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-9:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=ROOT / "datasets/epfl_ski2dpose/images")
    ap.add_argument("--labels", type=Path,
                    default=ROOT / "datasets/epfl_ski2dpose/ski2dpose_labels.json")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "datasets/epfl_ski2dpose/reconcile_report.json")
    ap.add_argument("--max", type=int, default=0, help="最多对账多少张（0=全部）")
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

        for jname, hidx in EPFL_TO_HALPE.items():
            j = EPFL_JOINTS.index(jname)
            if rec["annotation"][j][2] != 1 or ours[hidx, 2] < 0.3:
                continue
            e = float(np.linalg.norm(ours[hidx, :2] - epfl[j, :2]))
            joint_errs.append(e)
            joint_errs_norm.append(e / scale)

        for aname, (v, a, b) in ANGLE_TRIPLETS.items():
            iv, ia, ib = (EPFL_JOINTS.index(x) for x in (v, a, b))
            if min(rec["annotation"][iv][2], rec["annotation"][ia][2], rec["annotation"][ib][2]) != 1:
                continue
            hv, ha, hb = EPFL_TO_HALPE[v], EPFL_TO_HALPE[a], EPFL_TO_HALPE[b]
            if min(ours[hv, 2], ours[ha, 2], ours[hb, 2]) < 0.3:
                continue
            gt = angle_at(epfl[iv], epfl[ia], epfl[ib])
            got = angle_at(ours[hv], ours[ha], ours[hb])
            if not (np.isnan(gt) or np.isnan(got)):
                angle_errs[aname].append(abs(got - gt))

        n_done += 1
        if (i + 1) % 100 == 0:
            print(f"[reconcile] {i + 1}/{len(jobs)}")

    # 原始数据先落盘：汇总再崩也不用重跑推理
    np.savez(args.out.with_suffix(".raw.npz"), joint_errs=np.array(joint_errs),
             joint_errs_norm=np.array(joint_errs_norm),
             **{f"angle_{k}": np.array(v) for k, v in angle_errs.items()})

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
