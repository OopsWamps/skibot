"""多机位离线工具：拍手同步、棋盘格/ChArUco 标定、联合 bundle adjustment、三角化真值。

用途是 issue 06 的验证链路，不进产品运行时：
- 同步：每段开录 3 秒内拍两次手，按音频包络检测拍手时刻；同一相机两次
  拍手的偏移差必须 ≤ 半帧，否则整段标不可靠。
- 标定：棋盘格或 ChArUco，输出相机内参、畸变、位姿，世界原点是标定板
  第一个内角点（板坐标系，米）。
- 三角化：多视角 DLT，输出 (T,26,3) 三维点、有效掩码、重投影误差与视角数；
  `.npz` 可直接被验收脚本读取。

坐标与单位：K/dist 用 OpenCV 约定；R、t 是 world→camera；object points
与三角化结果的世界单位由标定板 square_size 决定，通常用米。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares  # type: ignore[import-untyped]

CALIBRATION_VERSION = "skibot.calibration/0.1"
TRIANGULATION_VERSION = "skibot.triangulation/0.1"

DEFAULT_SYNC_TOLERANCE_FRAMES = 0.5   # 两次拍手偏移差上限（半帧）
CLAP_THRESHOLD_RATIO = 0.5            # 拍手阈值：包络峰值的比例
CLAP_MIN_GAP_S = 0.2                  # 两次拍手的最小间隔
TRIANGULATION_MAX_REPROJ_PX = 5.0     # 单点接受的最大平均重投影误差
MIN_POSE_ROTATION_DEG = 1.0           # 内参标定视角的最小旋转覆盖
MIN_POSE_TRANSLATION_M = 0.01         # 内参标定视角的最小平移覆盖


@dataclass(frozen=True)
class PinholeCamera:
    """OpenCV 针孔相机：K、dist、world→camera 的 R/t。"""

    name: str
    K: np.ndarray
    dist: np.ndarray
    R: np.ndarray
    t: np.ndarray
    image_size: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        K = np.asarray(self.K, dtype=float)
        dist = np.asarray(self.dist, dtype=float).reshape(-1)
        R = np.asarray(self.R, dtype=float)
        t = np.asarray(self.t, dtype=float).reshape(3)
        if K.shape != (3, 3):
            raise ValueError(f"K 必须是 3x3，收到 {K.shape}")
        if R.shape != (3, 3):
            raise ValueError(f"R 必须是 3x3，收到 {R.shape}")
        if dist.size == 0:
            dist = np.zeros(5)
        elif dist.size < 5:
            dist = np.pad(dist, (0, 5 - dist.size))
        if not (np.isfinite(K).all() and np.isfinite(dist).all()
                and np.isfinite(R).all() and np.isfinite(t).all()):
            raise ValueError("相机参数不能含 NaN/Inf")
        object.__setattr__(self, "K", K)
        object.__setattr__(self, "dist", dist)
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "t", t)

    @property
    def P(self) -> np.ndarray:
        return self.K @ np.hstack([self.R, self.t.reshape(3, 1)])

    def project(self, points: np.ndarray, *, distort: bool = True) -> np.ndarray:
        """世界点 (...,3) → 像素 (...,2)；点在相机后方时该点为 NaN。"""
        X = np.asarray(points, dtype=float)
        Xc = X @ self.R.T + self.t
        z = Xc[..., 2]
        # 相机后方或光心平面上的点没有有意义的投影，统一记 NaN。
        safe_z = np.where(z <= 1e-12, np.nan, z)
        x = Xc[..., 0] / safe_z
        y = Xc[..., 1] / safe_z
        if distort:
            k1, k2, p1, p2, k3 = self.dist[:5]
            r2 = x * x + y * y
            radial = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3
            xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        else:
            xd, yd = x, y
        uv = np.stack([xd, yd, np.ones_like(xd)], axis=-1) @ self.K.T
        return uv[..., :2]

    def undistort(self, uv: np.ndarray) -> np.ndarray:
        pts = np.asarray(uv, dtype=float).reshape(-1, 1, 2)
        if np.allclose(self.dist, 0.0):
            return pts.reshape(-1, 2)
        out = cv2.undistortPoints(pts, self.K, self.dist, P=self.K)
        return out.reshape(-1, 2)

    def to_dict(self) -> dict[str, object]:
        return {
            "K": self.K.tolist(),
            "dist": self.dist.tolist(),
            "R": self.R.tolist(),
            "t": self.t.tolist(),
            "image_size": None if self.image_size is None else list(self.image_size),
        }

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "PinholeCamera":
        size = data.get("image_size")
        return cls(name=name, K=np.array(data["K"]), dist=np.array(data["dist"]),
                   R=np.array(data["R"]), t=np.array(data["t"]),
                   image_size=None if size is None else (int(size[0]), int(size[1])))


@dataclass(frozen=True)
class BoardPattern:
    """标定板。

    checkerboard：cols/rows 是内角点数（不是方格数），square_size 为米。
    charuco：cols/rows 是方格数，square_size/marker_size 为米，
    dictionary_id 取 cv2.aruco.DICT_*。
    """
    kind: str
    cols: int
    rows: int
    square_size: float
    marker_size: float | None = None
    dictionary_id: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("checkerboard", "charuco"):
            raise ValueError(f"未知标定板类型 {self.kind}")
        if self.cols <= 0 or self.rows <= 0 or self.square_size <= 0:
            raise ValueError("标定板尺寸必须为正")
        if self.kind == "checkerboard" and (self.cols + self.rows) % 2 == 0:
            # 对称棋盘格在 180° 旋转下与自身重合，检测器可能给出反序角点，
            # 共享板标定会静默差 180°；要求 7x6 这类非对称内角点。
            raise ValueError(
                "棋盘格内角点数之和必须为奇数（如 7x6 或 6x7），"
                "避免 180° 对称导致角点顺序歧义；对称板请改用 ChArUco")
        if self.kind == "charuco":
            if not self.marker_size or self.marker_size <= 0:
                raise ValueError("ChArUco 需要正 marker_size")
            if self.dictionary_id is None:
                object.__setattr__(self, "dictionary_id", cv2.aruco.DICT_4X4_50)

    def object_points(self) -> np.ndarray:
        if self.kind == "checkerboard":
            pts = [(x * self.square_size, y * self.square_size, 0.0)
                   for y in range(self.rows) for x in range(self.cols)]
            return np.asarray(pts, dtype=np.float32)
        pts = self.charuco_board().getChessboardCorners().astype(np.float32)
        # OpenCV 的 ChArUco 内角点从板外角开始；统一平移，使第一个
        # 内角点为 (0,0,0)，与 checkerboard 的世界原点约定一致。
        return pts - pts[0]

    def charuco_board(self):
        if self.kind != "charuco":
            raise ValueError("不是 ChArUco 板")
        dictionary = cv2.aruco.getPredefinedDictionary(int(self.dictionary_id))
        return cv2.aruco.CharucoBoard(
            (self.cols, self.rows), float(self.square_size),
            float(self.marker_size), dictionary)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind, "cols": self.cols, "rows": self.rows,
            "square_size": self.square_size, "marker_size": self.marker_size,
            "dictionary_id": self.dictionary_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BoardPattern":
        return cls(kind=str(data["kind"]), cols=int(data["cols"]),
                   rows=int(data["rows"]),
                   square_size=float(data["square_size"]),
                   marker_size=data.get("marker_size"),
                   dictionary_id=data.get("dictionary_id"))


@dataclass(frozen=True)
class BoardDetection:
    object_points: np.ndarray
    image_points: np.ndarray
    image_size: tuple[int, int]


def _gray(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3:
        return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return arr


def detect_board(image: np.ndarray | None,
                 pattern: BoardPattern) -> BoardDetection | None:
    """在一张图里检测标定板内角点；失败返回 None。

    棋盘格只用 findChessboardCorners + 亚像素，保证所有相机拿到同一种
    角点顺序；SB 检测器的顺序与经典检测器不同，混用会让两台相机的板
    坐标系差 180°。ChArUco 按 ID 配对，不受顺序影响。
    """
    arr = np.asarray(image)
    if arr.ndim < 2 or arr.size == 0:
        return None
    try:
        gray = _gray(arr)
        if gray.size == 0 or min(gray.shape[:2]) < 5:
            return None
        h, w = gray.shape[:2]
        if pattern.kind == "checkerboard":
            size = (pattern.cols, pattern.rows)
            found, corners = cv2.findChessboardCorners(
                gray, size,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
            if not found or corners is None:
                return None
            corners = cv2.cornerSubPix(
                gray, corners, (5, 5), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
            image_points = corners.reshape(-1, 2).astype(np.float32)
            object_points = pattern.object_points()
            if len(image_points) != len(object_points):
                return None
            return BoardDetection(object_points, image_points, (w, h))

        board = pattern.charuco_board()
        detector = cv2.aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
        if charuco_corners is None or charuco_ids is None or len(charuco_ids) < 4:
            return None
        ids = charuco_ids.flatten().astype(int)
        all_points = pattern.object_points()
        return BoardDetection(all_points[ids],
                              charuco_corners.reshape(-1, 2).astype(np.float32), (w, h))
    except (cv2.error, ValueError):
        return None


@dataclass
class CameraCalibration:
    camera: PinholeCamera
    rms: float
    n_views: int
    view_poses: list[tuple[np.ndarray, np.ndarray]]


def _pose_spread(poses: Sequence[tuple[np.ndarray, np.ndarray]]
                 ) -> tuple[float, float]:
    """视角位姿的最大旋转差（度）与最大平移差（与 square_size 同单位）。"""
    max_rot = 0.0
    max_trans = 0.0
    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            Rrel = poses[i][0] @ poses[j][0].T
            cos = float(np.clip((np.trace(Rrel) - 1.0) / 2.0, -1.0, 1.0))
            max_rot = max(max_rot, float(np.degrees(np.arccos(cos))))
            max_trans = max(max_trans, float(np.linalg.norm(poses[i][1] - poses[j][1])))
    return max_rot, max_trans


def calibrate_from_detections(detections: Sequence[BoardDetection], *,
                              camera_name: str = "cam0") -> CameraCalibration:
    """用已知检测点做内参标定；至少两个视角，且视角位姿要有差异。

    重复位姿（同一张图的多次检测）在网上看 rms 接近 0，但焦距不可观。
    这里用标定出的每视角位姿检查旋转/平移覆盖：两者都低于阈值时直接
    报错。不能用检测点像素标准差——逐图独立角点噪声也能把标准差抬高。
    允许各视角角点数不同（ChArUco 部分可见）。
    """
    if len(detections) < 2:
        raise ValueError("标定至少需要两个视角")
    image_size = detections[0].image_size
    if any(d.image_size != image_size for d in detections):
        raise ValueError("同一相机的标定图尺寸必须一致")
    object_views = [d.object_points.astype(np.float32) for d in detections]
    image_views = [d.image_points.astype(np.float32) for d in detections]
    w, h = image_size
    K0 = np.array([[1.2 * w, 0.0, w / 2.0],
                   [0.0, 1.2 * w, h / 2.0],
                   [0.0, 0.0, 1.0]])
    rms, K, dist, rvecs, tvecs, std_int, _, _ = cv2.calibrateCameraExtended(
        object_views, image_views, image_size, K0, np.zeros(5))
    if not (0.0 < K[0, 0] < 10.0 * w and 0.0 < K[1, 1] < 10.0 * h):
        raise ValueError(
            f"内参退化（fx={K[0,0]:.1f}, fy={K[1,1]:.1f}），"
            "请增加不同位姿的标定图")
    if not (0.0 <= K[0, 2] <= w and 0.0 <= K[1, 2] <= h):
        raise ValueError(f"主点落在图像外（cx={K[0,2]:.1f}, cy={K[1,2]:.1f}），标定退化")
    std = np.asarray(std_int, dtype=float).ravel()
    std_fx, std_fy = float(std[0]), float(std[1])
    if std_fx > 0.1 * abs(K[0, 0]) or std_fy > 0.1 * abs(K[1, 1]):
        raise ValueError(
            f"内参不确定度过大（fx±{std_fx:.1f}, fy±{std_fy:.1f}），"
            "视角位姿覆盖不足；请增加不同的标定角度/位置")
    poses = []
    for rvec, tvec in zip(rvecs, tvecs):
        R, _ = cv2.Rodrigues(rvec)
        poses.append((R, tvec.reshape(3)))
    rot_spread, trans_spread = _pose_spread(poses)
    if rot_spread < MIN_POSE_ROTATION_DEG and trans_spread < MIN_POSE_TRANSLATION_M:
        raise ValueError(
            f"标定视角位姿重复（旋转覆盖 {rot_spread:.2f}°、"
            f"平移覆盖 {trans_spread*1000:.1f}mm），无法稳定估计内参；"
            "请提供不同角度/位置的标定图")
    camera = PinholeCamera(camera_name, K, dist, poses[0][0], poses[0][1],
                           image_size)
    return CameraCalibration(camera, float(rms), len(detections), poses)


def calibrate_camera(images: Sequence[np.ndarray], pattern: BoardPattern, *,
                     camera_name: str = "cam0") -> CameraCalibration:
    detections = [d for d in (detect_board(img, pattern) for img in images) if d]
    return calibrate_from_detections(detections, camera_name=camera_name)


@dataclass
class CalibrationSet:
    cameras: dict[str, PinholeCamera]
    world_origin: str
    pattern: BoardPattern
    rms: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "version": CALIBRATION_VERSION,
            "world_origin": self.world_origin,
            "pattern": self.pattern.to_dict(),
            "rms": self.rms,
            "cameras": {name: cam.to_dict() for name, cam in self.cameras.items()},
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                     encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationSet":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("version") != CALIBRATION_VERSION:
            raise ValueError(f"未知标定文件版本 {data.get('version')}")
        cameras = {name: PinholeCamera.from_dict(name, cam)
                   for name, cam in data["cameras"].items()}
        return cls(cameras, str(data["world_origin"]),
                   BoardPattern.from_dict(data["pattern"]), data.get("rms"))


@dataclass(frozen=True)
class Observation:
    camera: int
    point: int
    uv: tuple[float, float]


@dataclass
class BundleAdjustmentResult:
    cameras: list[PinholeCamera]
    initial_rms: float
    final_rms: float
    success: bool
    n_invalid: int = 0


def _pose_rvec(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(R)
    return np.concatenate([rvec.reshape(3), t.reshape(3)])


def bundle_adjust(cameras: Sequence[PinholeCamera], points_3d: np.ndarray,
                  observations: Sequence[Observation], *,
                  fix_first: bool = False) -> BundleAdjustmentResult:
    """固定 K/dist 与三维点，优化相机位姿，最小化重投影误差。

    三维点（标定板内角点）固定，世界原点由标定板定义；fix_first 只用于
    需要人为固定参考相机的调用。
    """
    cams = list(cameras)
    pts = np.asarray(points_3d, dtype=float)
    if len(observations) == 0:
        raise ValueError("没有观测，无法做 bundle adjustment")
    free = [i for i in range(len(cams)) if not (fix_first and i == 0)]
    x0 = np.concatenate([_pose_rvec(cams[i].R, cams[i].t) for i in free]) if free else np.array([])

    def current_poses(x: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        poses = [(cam.R, cam.t) for cam in cams]
        for offset, idx in enumerate(free):
            block = x[offset * 6:(offset + 1) * 6]
            R, _ = cv2.Rodrigues(block[:3].reshape(3, 1))
            poses[idx] = (R, block[3:].reshape(3))
        return poses

    def evaluate(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        poses = current_poses(x)
        out = np.empty(2 * len(observations), dtype=float)
        valid = np.ones(len(observations), dtype=bool)
        for k, obs in enumerate(observations):
            R, t = poses[obs.camera]
            cam = cams[obs.camera]
            Xc = R @ pts[obs.point] + t
            if Xc[2] <= 1e-9:
                out[2 * k:2 * k + 2] = 1e3  # 罚项，只用于把优化器推离退化区
                valid[k] = False
                continue
            one = PinholeCamera(cam.name, cam.K, cam.dist, R, t)
            uv = one.project(pts[obs.point])
            out[2 * k:2 * k + 2] = uv - np.asarray(obs.uv, dtype=float)
        return out, valid

    def residual(x: np.ndarray) -> np.ndarray:
        return evaluate(x)[0]

    def rms_of(r: np.ndarray, valid: np.ndarray) -> float:
        if not valid.any():
            return float("nan")
        return float(np.sqrt(np.mean(r[np.repeat(valid, 2)] ** 2)))

    initial_res, initial_valid = evaluate(x0)
    initial = rms_of(initial_res, initial_valid)
    if not initial_valid.any():
        # 全部观测点在相机后方：没有可优化信息，不能把 1e3 罚项当 RMS 成功。
        return BundleAdjustmentResult(
            cams, float("nan"), float("nan"), False, len(observations))
    if free:
        result = least_squares(residual, x0, loss="soft_l1", f_scale=2.0,
                               max_nfev=200)
        final_x = result.x
        success = bool(result.success)
    else:
        final_x = x0
        success = True
    final_res, final_valid = evaluate(final_x)
    final = rms_of(final_res, final_valid)
    n_invalid = int((~final_valid).sum())
    # 有观测点在相机后方时不能宣称成功；调用方必须看到 n_invalid。
    success = success and n_invalid == 0
    poses = current_poses(final_x)
    refined = [PinholeCamera(cam.name, cam.K, cam.dist, R, t, cam.image_size)
               for cam, (R, t) in zip(cams, poses)]
    return BundleAdjustmentResult(refined, initial, final, success, n_invalid)


def _board_point_indices(pattern: BoardPattern,
                         detection: BoardDetection) -> list[int]:
    """检测点 → 板点索引。棋盘格按检测顺序，ChArUco 按最近板点。"""
    if pattern.kind == "checkerboard":
        if len(detection.object_points) != len(pattern.object_points()):
            raise ValueError("棋盘格检测点数与板点数不一致")
        return list(range(len(detection.object_points)))
    board_points = pattern.object_points()
    indices = []
    for xyz in detection.object_points:
        d = np.linalg.norm(board_points - xyz, axis=1)
        idx = int(np.argmin(d))
        if d[idx] > 1e-6:
            raise ValueError("ChArUco 检测点无法对应到板点")
        indices.append(idx)
    return indices


def calibrate_shared_board(
    intrinsic_detections: Mapping[str, Sequence[BoardDetection]],
    extrinsic_detections: Mapping[str, BoardDetection],
    pattern: BoardPattern,
) -> CalibrationSet:
    """多台相机看同一块静止标定板：内参图集与外参图集必须分开。

    intrinsic_detections：每台相机、板移动的多个不同位姿，用于内参。
    extrinsic_detections：每台相机同一静止板位姿各一张，用于外参和
    联合 bundle adjustment（固定板点，优化各相机位姿）。
    世界原点 = 板第一个内角点。
    """
    if set(intrinsic_detections) != set(extrinsic_detections):
        raise ValueError("内参图集与外参图集的相机名不一致")
    cameras: dict[str, PinholeCamera] = {}
    for name, dets in intrinsic_detections.items():
        calib = calibrate_from_detections(list(dets), camera_name=name)
        ext = extrinsic_detections[name]
        if calib.camera.image_size != ext.image_size:
            raise ValueError(
                f"相机 {name} 内参图尺寸 {calib.camera.image_size} 与外参图"
                f"尺寸 {ext.image_size} 不一致；K 不能跨分辨率直接复用")
        rvec, tvec = cv2.solvePnP(
            ext.object_points.astype(np.float32),
            ext.image_points.astype(np.float32),
            calib.camera.K, calib.camera.dist,
            flags=cv2.SOLVEPNP_ITERATIVE)[1:3]
        R, _ = cv2.Rodrigues(rvec)
        cameras[name] = PinholeCamera(
            name, calib.camera.K, calib.camera.dist, R, tvec.reshape(3),
            ext.image_size)

    points_3d = pattern.object_points()
    observations: list[Observation] = []
    for cam_idx, name in enumerate(cameras):
        ext = extrinsic_detections[name]
        indices = _board_point_indices(pattern, ext)
        for point_idx, uv in zip(indices, ext.image_points):
            observations.append(Observation(
                cam_idx, point_idx, (float(uv[0]), float(uv[1]))))

    result = bundle_adjust(list(cameras.values()), points_3d, observations)
    if (not result.success or result.n_invalid > 0
            or not np.isfinite(result.final_rms)):
        raise ValueError(
            f"联合 bundle adjustment 不可靠：success={result.success}, "
            f"n_invalid={result.n_invalid}, rms={result.final_rms}")
    refined = {cam.name: cam for cam in result.cameras}
    return CalibrationSet(
        refined,
        "标定板第一个内角点为原点；+x/+y 按 OpenCV 棋盘格角点顺序、"
        "+z 垂直板面向外；单位由 square_size 决定",
        pattern, rms=result.final_rms)


def calibrate_shared_board_from_images(
    intrinsic_images: dict[str, Sequence[np.ndarray]],
    extrinsic_images: dict[str, np.ndarray],
    pattern: BoardPattern,
) -> CalibrationSet:
    """标定图的便捷入口：检测后转给 calibrate_shared_board。"""
    intrinsic: dict[str, list[BoardDetection]] = {}
    for name, images in intrinsic_images.items():
        dets = [d for d in (detect_board(img, pattern) for img in images) if d]
        if not dets:
            raise ValueError(f"相机 {name} 的内参图集没有检测到标定板")
        intrinsic[name] = dets
    extrinsic: dict[str, BoardDetection] = {}
    for name, image in extrinsic_images.items():
        det = detect_board(image, pattern)
        if det is None:
            raise ValueError(f"相机 {name} 的外参图没有检测到标定板")
        extrinsic[name] = det
    return calibrate_shared_board(intrinsic, extrinsic, pattern)


def triangulate_point(observations: Sequence[tuple[PinholeCamera, np.ndarray]]
                      ) -> np.ndarray | None:
    """至少两个视角的 DLT 三角化；点在任一相机后方或退化时返回 None。"""
    if len(observations) < 2:
        return None
    rows = []
    for cam, uv in observations:
        uv_und = cam.undistort(np.asarray(uv, dtype=float).reshape(1, 2))
        if not np.isfinite(uv_und).all():
            return None  # 观测本身无效（例如点在相机后方被记成 NaN）
        u, v = uv_und[0]
        P = cam.P
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    Xh = vt[-1]
    if abs(float(Xh[3])) < 1e-12:
        return None
    X = Xh[:3] / Xh[3]
    if not np.isfinite(X).all():
        return None
    for cam, _ in observations:
        if (cam.R @ X + cam.t)[2] <= 0:
            return None
    return X


@dataclass
class TriangulationResult:
    points: np.ndarray          # (T, J, 3)
    valid: np.ndarray           # (T, J)
    reproj_error: np.ndarray    # (T, J) 平均像素误差，无效为 NaN
    n_views: np.ndarray         # (T, J)
    camera_names: list[str]
    max_reproj_px: float

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        if p.suffix.lower() != ".npz":
            # 追加而不是替换后缀：save("truth.data") 应写 truth.data.npz，
            # 这样 load("truth.data") 也能找回。
            p = Path(str(p) + ".npz")
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p, points=self.points, valid=self.valid,
            reproj_error=self.reproj_error, n_views=self.n_views,
            camera_names=np.asarray(self.camera_names, dtype=str),
            max_reproj_px=np.asarray(self.max_reproj_px),
            version=np.asarray(TRIANGULATION_VERSION))
        return p

    @classmethod
    def load(cls, path: str | Path) -> "TriangulationResult":
        p = Path(path)
        if not p.exists() and p.suffix.lower() != ".npz":
            alt = Path(str(p) + ".npz")
            if alt.exists():
                p = alt
        data = np.load(p, allow_pickle=False)
        version = str(data["version"])
        if version != TRIANGULATION_VERSION:
            raise ValueError(f"未知三角化文件版本 {version}")
        return cls(
            points=data["points"], valid=data["valid"].astype(bool),
            reproj_error=data["reproj_error"], n_views=data["n_views"].astype(int),
            camera_names=[str(x) for x in data["camera_names"]],
            max_reproj_px=float(data["max_reproj_px"]))


def triangulate_tracks(
    cameras: dict[str, PinholeCamera],
    tracks: dict[str, np.ndarray], *,
    max_reproj_px: float = TRIANGULATION_MAX_REPROJ_PX,
) -> TriangulationResult:
    """tracks[camera] 形状 (T,J,2)，缺失为 NaN；输出 (T,J,3) 真值与质量。"""
    if len(cameras) < 2:
        raise ValueError("三角化至少需要两台相机")
    if set(cameras) != set(tracks):
        raise ValueError("cameras 与 tracks 的相机名不一致")
    shapes = {t.shape for t in tracks.values()}
    if len(shapes) != 1:
        raise ValueError(f"各相机轨迹形状不一致：{shapes}")
    t_count, j_count = next(iter(shapes))[:2]
    points = np.full((t_count, j_count, 3), np.nan)
    valid = np.zeros((t_count, j_count), dtype=bool)
    reproj = np.full((t_count, j_count), np.nan)
    n_views = np.zeros((t_count, j_count), dtype=int)
    names = list(cameras)

    for t in range(t_count):
        for j in range(j_count):
            obs = []
            for name in names:
                uv = tracks[name][t, j]
                if np.isfinite(uv).all():
                    obs.append((cameras[name], uv))
            n_views[t, j] = len(obs)
            if len(obs) < 2:
                continue
            X = triangulate_point(obs)
            if X is None:
                continue
            errors = [float(np.linalg.norm(cam.project(X) - uv)) for cam, uv in obs]
            mean_error = float(np.mean(errors))
            if mean_error > max_reproj_px:
                continue
            points[t, j] = X
            reproj[t, j] = mean_error
            valid[t, j] = True

    return TriangulationResult(points, valid, reproj, n_views, names, max_reproj_px)


def audio_envelope(audio: np.ndarray, sample_rate: float,
                   window_s: float = 0.01) -> np.ndarray:
    """绝对波形做短窗滑动平均，得到拍手检测用的包络。"""
    x = np.abs(np.asarray(audio, dtype=float).reshape(-1))
    if x.size == 0:
        return x
    window = max(1, min(int(round(window_s * sample_rate)), x.size))
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def detect_claps(envelope: np.ndarray, sample_rate: float, *,
                 threshold_ratio: float = CLAP_THRESHOLD_RATIO,
                 min_gap_s: float = CLAP_MIN_GAP_S) -> list[float]:
    """包络超过阈值峰值的连续段各记一次拍手，返回秒。"""
    env = np.asarray(envelope, dtype=float).reshape(-1)
    if env.size == 0 or not np.isfinite(env).any():
        return []
    peak = float(np.nanmax(env))
    if peak <= 0:
        return []
    above = env >= threshold_ratio * peak
    times: list[float] = []
    i = 0
    while i < len(env):
        if not above[i]:
            i += 1
            continue
        j = i
        while j < len(env) and above[j]:
            j += 1
        idx = i + int(np.nanargmax(env[i:j]))
        times.append(idx / sample_rate)
        i = j
    merged: list[float] = []
    for t in times:
        if merged and t - merged[-1] < min_gap_s:
            # 同一拍手的多个峰：保留包络更高的那个
            prev = merged[-1]
            if env[int(round(t * sample_rate))] > env[int(round(prev * sample_rate))]:
                merged[-1] = t
        else:
            merged.append(t)
    return merged


@dataclass
class SyncResult:
    offsets_s: dict[str, float]
    reliable: bool
    reasons: dict[str, str]
    clap_times: dict[str, list[float]]
    tolerance_s: float


def sync_claps(envelopes: dict[str, np.ndarray], sample_rate: float, *,
               fps: float, tolerance_frames: float = DEFAULT_SYNC_TOLERANCE_FRAMES,
               expected_claps: int = 2) -> SyncResult:
    """按拍手时刻做相机同步；不满足半帧判据的相机整条标不可靠。

    每路应当恰好检测到 expected_claps 次拍手（默认两次）。多检或少检
    都按不可靠处理：一路漏掉第一次拍手时按索引配对会得到错误偏移且
    漂移为 0，不能靠数量相等去猜。

    offsets_s[name] 加在该相机时间上对齐参考相机：cam_time + offset = ref_time。
    参考相机 offset 固定为 0；不可靠的相机不出现在 offsets_s 里。

    已知限制：两路各恰好两次拍手、其中一路漏第一次而补检了后续事件、
    且拍手间隔恰好相同时，索引配对存在 ±间隔 的固有歧义；需要外部
    录制起始时间先验或人工确认。
    """
    if fps <= 0:
        raise ValueError(f"fps 必须是正数，收到 {fps}")
    if len(envelopes) < 2:
        raise ValueError("同步至少需要两路音频")
    names = list(envelopes)
    clap_times = {name: detect_claps(envelopes[name], sample_rate)
                  for name in names}
    ref = names[0]
    tolerance_s = tolerance_frames / fps
    offsets: dict[str, float] = {}
    reasons: dict[str, str] = {}
    reliable = True

    if len(clap_times[ref]) != expected_claps:
        reliable = False
        reasons[ref] = (f"参考相机拍手数 {len(clap_times[ref])} != "
                        f"{expected_claps}")
    for name in names[1:]:
        times = clap_times[name]
        if len(times) != expected_claps or len(clap_times[ref]) != expected_claps:
            reliable = False
            reasons[name] = (f"拍手数不是 {expected_claps} 次：参考 "
                             f"{len(clap_times[ref])}，本机 {len(times)}")
            continue
        diffs = [clap_times[ref][i] - times[i] for i in range(expected_claps)]
        drift = max(diffs) - min(diffs)
        if drift > tolerance_s + 1e-9:  # 恰半帧要通过，吸收浮点减法误差
            reliable = False
            reasons[name] = (f"两次拍手偏移差 {drift*1000:.1f}ms "
                             f"> {tolerance_s*1000:.1f}ms")
            continue
        offsets[name] = float(np.mean(diffs))
    if len(clap_times[ref]) != expected_claps:
        offsets = {}
    else:
        offsets.setdefault(ref, 0.0)
    return SyncResult(offsets, reliable, reasons, clap_times, tolerance_s)
