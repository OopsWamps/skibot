"""上传预检：用廉价信号（容器与编码、分辨率、时长、抽帧人物占比）快速判
材质，不合格当场拒绝并给重拍建议，不进入分析队列（spec「质量门控」第一段）。

设计约束：预检只做秒级检查；抽帧人物检测用 HOG 行人检测器（滑雪蹲姿漏检
常见），检不出按"无法判定"放行，只有确凿检出且全部过小才拒绝——预检是
硬门，宁放勿杀；报告里的一期"人物偏小"水印（SMALL_SUBJECT_RATIO=0.2）
仍然是软提示。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


@dataclass
class PrecheckConfig:
    """预检判据。默认值的依据：

    - min_duration_s=3.0：一个弯约 1~2s（README），相位切分只保留完整换刃
      （phases.MIN_DURATION_S=0.5 的弯才有效）；3s 才可能含 1~2 个完整弯。
    - max_frames=10000 / max_duration_s=600.0：分析是单线程排队的分钟级
      CPU 任务，README 建议长视频（>10000 帧）先降采样到 15fps；帧数直接
      对应工作量，时长只是兜底（600s@15fps 恰好 9000 帧）。两者任一超限
      都拒绝，60fps 的 200s 素材也会被帧数拦住。
    - min_short_side=480 / min_long_side=720：手机竖拍 720x1280、横拍
      1280x720 都通过；同样人物占比下分辨率减半意味着关节像素减半、
      关键点定位噪声上升——一期报告里小人物已作为质量提示
      （report.SMALL_SUBJECT_RATIO），这里是它的硬门版本。
    - min_person_ratio=0.18：与内置 HOG 的实际分辨力对齐——检测窗口固定
      64x128，画面里人高不足 128px 就检不出（检不出按无法判定放行）。
      720p 下 128px 相当于帧高 17.8%，所以阈值不能低于 0.18，否则这条
      硬门形同虚设；1080p 下 0.18 对应约 194px 的人物，与一期
      SMALL_SUBJECT_RATIO=0.2 的软水印量级一致。允许的 480p/540p 素材上
      HOG 下限（26.7%/23.7%）已高于阈值，硬门对这些分辨率退化为放行。
    - n_probe_frames=5：HOG 在 720p 原生分辨率约 0.12s/帧、1080p 约
      0.27s/帧，抽帧铺在视频中段 80%，预检总耗时控制在秒级。
    """

    min_duration_s: float = 3.0
    max_duration_s: float = 600.0
    max_frames: int = 10000
    min_short_side: int = 480
    min_long_side: int = 720
    min_person_ratio: float = 0.18
    n_probe_frames: int = 5

    def __post_init__(self) -> None:
        # 判据配错（负数、NaN、倒挂）会静默关掉检查，宁可启动时报错
        if not 0 < self.min_duration_s < self.max_duration_s:
            raise ValueError("时长判据必须是 0 < min < max")
        if self.max_frames <= 0:
            raise ValueError("max_frames 必须为正")
        if self.min_short_side <= 0 or self.min_long_side <= 0:
            raise ValueError("分辨率下限必须为正")
        if not 0 < self.min_person_ratio <= 1:  # NaN 也会落进这里
            raise ValueError("min_person_ratio 必须在 (0, 1]")
        if self.n_probe_frames < 1:
            raise ValueError("n_probe_frames 至少为 1")


@dataclass
class PrecheckIssue:
    check: str
    problem: str   # 拒绝原因，用户可读
    advice: str    # 重拍/处理建议


@dataclass
class PrecheckResult:
    passed: bool
    issues: list[PrecheckIssue] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # "无法判定"类放行说明
    person_ratio: float | None = None               # None = 无法判定


HOG_MIN_PERSON_PX = 128  # HOG 默认检测窗口 64x128，低于这个高度检不出
MAX_PROBE_LONG_SIDE = 1920  # 4K 素材缩到 1080p 量级，控制单帧检测耗时


def _default_probe(frame: np.ndarray) -> list[float]:
    """内置 HOG 行人检测：返回检出人物高度占帧高的比例列表。

    原生分辨率检测（720p 约 0.12s/帧）；只有长边超过 1920 才缩放，
    比例对缩放不变。HOG 窗口下限 128px：人比它矮就检不出，调用方按
    "无法判定"放行；滑雪蹲姿与宽松雪服下漏检同样放行。
    """
    h, w = frame.shape[:2]
    if max(h, w) > MAX_PROBE_LONG_SIDE:
        scale = MAX_PROBE_LONG_SIDE / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    rects, _ = _hog().detectMultiScale(frame, hitThreshold=0.0,
                                       winStride=(8, 8), padding=(4, 4),
                                       scale=1.05)
    fh = frame.shape[0]
    return [rect[3] / fh for rect in rects]


_HOG = None


def _hog() -> cv2.HOGDescriptor:
    global _HOG
    if _HOG is None:
        _HOG = cv2.HOGDescriptor()
        # OpenCV 的 Python 绑定在运行时动态挂这个函数，类型存根里没有
        _HOG.setSVMDetector(
            cv2.HOGDescriptor_getDefaultPeopleDetector())  # type: ignore[attr-defined]
    return _HOG


def run_precheck(path: Path, config: PrecheckConfig | None = None,
                 person_probe: Callable[[np.ndarray], list[float]] | None = None,
                 ) -> PrecheckResult:
    """对已落盘的视频跑预检。person_probe 可注入替身（测试/换检测器）；
    注入的检测器抛异常与内置 HOG 一样按"无法判定"处理，不误杀。

    元数据检查（容器/编码、分辨率、时长）全过才做抽帧人物占比——前者失败
    时素材已不可用，没必要再花检测时间。
    """
    cfg = config or PrecheckConfig()
    issues: list[PrecheckIssue] = []
    notes: list[str] = []
    person_ratio: float | None = None

    path = Path(path)
    if not path.is_file():
        # 路径写错/上传临时文件被清理：基础设施问题，不是素材问题，
        # 不能给"转码重传"的建议
        return PrecheckResult(False, [PrecheckIssue(
            "file", "视频文件不存在或不可读",
            "重新选择本地视频文件后上传")])

    cap = cv2.VideoCapture(str(path))
    opened = bool(cap.isOpened())
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
    fps = cap.get(cv2.CAP_PROP_FPS) if opened else 0.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) if opened else 0.0
    cap.release()

    if not opened or width <= 0 or height <= 0:
        issues.append(PrecheckIssue(
            "container", "视频无法解码：容器或编码不受支持",
            "用手机直拍或转码为 H.264 编码的 mp4 后重传"))
        return PrecheckResult(passed=False, issues=issues)

    short, long = sorted((width, height))
    if short < cfg.min_short_side or long < cfg.min_long_side:
        issues.append(PrecheckIssue(
            "resolution", f"分辨率 {width}x{height} 过低（要求短边 ≥ "
                          f"{cfg.min_short_side}、长边 ≥ {cfg.min_long_side}）",
            "相机设置选 720p 以上；已录制的低清素材需要重新拍摄"))

    # 帧数上限只看帧数：帧数可读、帧率读不出（VFR/浏览器录制）的素材
    # 同样占单线程队列的工作量
    if frames > cfg.max_frames:
        issues.append(PrecheckIssue(
            "duration", f"视频 {int(frames)} 帧，超过处理上限 "
                        f"{cfg.max_frames} 帧",
            "裁出精华滑行段，或降采样到 15fps（ffmpeg -vf fps=15）后重传"))
    if fps <= 0 or frames <= 0:
        # 部分 VFR 手机素材、浏览器录制、fragmented mp4 的帧率/帧数是 0，
        # 但文件可解码；分析链路也不依赖这两个字段（render.probe_video 对
        # fps=0 回退 30）。跳过时长判据并注明，不当拒。
        if fps <= 0 and frames <= 0:
            notes.append("无法读取帧率与帧数，跳过时长与人物占比判据（放行）")
        elif fps <= 0:
            notes.append(f"无法读取帧率，跳过时长判据（放行；帧数 {int(frames)}）")
        else:
            notes.append("无法读取帧数，跳过时长与人物占比判据（放行）")
    else:
        duration = frames / fps
        if duration < cfg.min_duration_s:
            issues.append(PrecheckIssue(
                "duration", f"视频仅 {duration:.1f}s，太短",
                "录制完整滑行段：至少包含 1~2 个连续换刃（3s 以上），"
                "起滑前多留 1s"))
        elif duration > cfg.max_duration_s:
            issues.append(PrecheckIssue(
                "duration", f"视频 {duration / 60:.1f} 分钟，超过 "
                            f"{cfg.max_duration_s / 60:.0f} 分钟上限",
                "裁出精华滑行段，或降采样到 15fps（ffmpeg -vf fps=15）后重传"))

    if not issues:
        person_ratio, probe_notes, decode_failed = _probe_person(
            path, cfg, int(frames) if frames > 0 else None, person_probe)
        notes.extend(probe_notes)
        if decode_failed:
            # 元数据可读但一帧都解不出来：这是编码/文件损坏，不是检测漏检，
            # 不能按"无法判定"放行，否则会拖到 Sports2D 阶段才失败
            issues.append(PrecheckIssue(
                "container", "视频无法解码：抽帧读不出画面（文件损坏或编码不受支持）",
                "确认视频能在本地播放器完整播放；不能播放就转码为 H.264 的 "
                "mp4 后重传"))
        elif person_ratio is not None and person_ratio < cfg.min_person_ratio:
            issues.append(PrecheckIssue(
                "person", f"画面中人物高度仅占帧高的 "
                          f"{person_ratio * 100:.0f}%，太小",
                "靠近拍摄：让被分析者身高约占画面高度的 1/3 以上，"
                "避免全身只占一小条"))

    return PrecheckResult(passed=not issues, issues=issues, notes=notes,
                          person_ratio=person_ratio)


def _probe_person(path: Path, cfg: PrecheckConfig, n_frames: int | None,
                  person_probe: Callable[[np.ndarray], list[float]] | None,
                  ) -> tuple[float | None, list[str], bool]:
    """抽帧检测人物占比：返回 (最大占比或 None, 放行说明, 是否解码失败)。

    帧数已知时按 10%~90% 比例 seek，跳过片头。帧数未知（VFR、索引缺失）
    时无法确定采样位置：只做解码检查，跳过人物占比判据并注明——若强行从
    前几帧抽样，片头无人会把长素材误判成"人物太小"。按位置 seek 读不出时
    再试顺序读第 0 帧，区分两件事：
    - 顺序读也失败 → 文件不可解码，decode_failed=True，由调用方按容器问题
      拒绝（拖到 Sports2D 阶段才发现就晚了）；
    - 顺序读成功 → 只是该容器不支持按帧 seek，按"无法判定"放行。
    帧可读但检测器未检出/抛异常同样按"无法判定"放行，不误杀；两者在
    notes 里分开写，坏掉的检测器不会伪装成雪服漏检。
    """
    probe = person_probe or _default_probe
    cap = cv2.VideoCapture(str(path))
    ratios: list[float] = []
    readable = 0
    probe_errors = 0

    def _run(frame: np.ndarray) -> None:
        nonlocal probe_errors
        try:
            ratios.extend(probe(frame))
        except Exception:
            probe_errors += 1

    try:
        if not n_frames or n_frames <= 1:
            # 帧数未知：先确认能解码，再跳过人物占比（无法确定比例）
            for _ in range(3):
                ok, _ = cap.read()
                if ok:
                    return None, ["无法确定视频帧数，跳过人物占比判据（放行）"], False
            return None, [], True

        lo, hi = 0.1, 0.9
        positions = [lo + (hi - lo) * i / max(cfg.n_probe_frames - 1, 1)
                     for i in range(cfg.n_probe_frames)]
        for pos in positions:
            # set 返回 False 说明容器不能按帧定位；此时 read 会从当前位置
            # 继续，若照常采样就全落在片头，会把人物在后的素材误判成太小
            if not cap.set(cv2.CAP_PROP_POS_FRAMES,
                           int(pos * (n_frames - 1))):
                continue
            ok, frame = cap.read()
            if not ok:
                continue
            readable += 1
            _run(frame)

        if readable == 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, _ = cap.read()
            if not ok:
                return None, [], True
            return None, ["抽帧定位失败（容器不支持按帧跳转），"
                          "无法判定人物占比（放行）"], False
        if not ratios:
            if probe_errors:
                return None, ["人物检测器异常，无法判定人物占比（放行）"], False
            return None, ["抽帧未检出人物（滑雪蹲姿下行人检测常见漏检），"
                          "无法判定人物占比（放行）"], False
        return max(ratios), [], False
    finally:
        cap.release()
