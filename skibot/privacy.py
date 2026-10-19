"""人脸打码。

决策依据 docs/DECISIONS.md 第 11 条：标注视频默认人脸打码（可关）。
实现思路参考 Pose2Sim Utilities/face_blurring.py，用 OpenCV Haar 级联逐帧检测+高斯模糊。

已知局限：雪镜、面罩、口罩、侧脸会显著降低 Haar 召回率。真视频验证阶段统计漏检率，
不合格则换 DNN 检测器（如 YuNet）。默认在"输入视频"上打码——原始画面全程受保护，
且 Sports2D 文档确认对姿态精度影响很小。
"""

from __future__ import annotations

from pathlib import Path

import cv2

_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def blur_faces(src: str | Path, dst: str | Path, blur_ksize: int = 51) -> tuple[Path, int]:
    """对 src 视频逐帧人脸模糊，写出 dst。返回 (dst, 检出总脸数)。"""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise FileNotFoundError(f"打不开视频: {src}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    detector = cv2.CascadeClassifier(_CASCADE)
    total = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        total += len(faces)
        for (x, y, fw, fh) in faces:
            roi = frame[y:y + fh, x:x + fw]
            frame[y:y + fh, x:x + fw] = cv2.GaussianBlur(roi, (blur_ksize, blur_ksize), 30)
        writer.write(frame)

    cap.release()
    writer.release()
    return dst, total
