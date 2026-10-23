"""多机位工具测试：python tests/test_multiview.py

合成相机覆盖：三角化还原、轨迹质量掩码、标定内参、bundle adjustment、
ChArUco 检测、拍手同步与半帧判据、标定/三角化文件读写。
"""

import sys
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skibot.multiview import (  # noqa: E402
    BoardDetection, BoardPattern, CalibrationSet, Observation, PinholeCamera,
    TriangulationResult, audio_envelope, bundle_adjust, calibrate_from_detections,
    calibrate_shared_board, detect_board, sync_claps, triangulate_point,
    triangulate_tracks,
)


def make_camera(name: str, R: np.ndarray, t: np.ndarray, *,
                f: float = 800.0, cx: float = 320.0, cy: float = 240.0,
                size: tuple[int, int] = (640, 480)) -> PinholeCamera:
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
    return PinholeCamera(name, K, np.zeros(5), R, t, size)


def scene():
    rng = np.random.default_rng(1)
    points = np.column_stack([
        rng.uniform(-0.5, 0.5, 20),
        rng.uniform(-0.3, 0.3, 20),
        rng.uniform(4.0, 6.0, 20),
    ])
    cams = [
        make_camera("cam0", np.eye(3), np.zeros(3)),
        make_camera("cam1", np.eye(3), np.array([-1.0, 0.0, 0.0])),
        make_camera("cam2", cv2.Rodrigues(np.array([0.0, 0.12, 0.0]))[0],
                    np.array([-0.5, 0.2, 0.1])),
    ]
    return points, cams


def test_triangulation_roundtrip():
    points, cams = scene()
    for X in points:
        obs = [(cam, cam.project(X)) for cam in cams]
        Xh = triangulate_point(obs)
        assert Xh is not None, X
        assert np.linalg.norm(Xh - X) < 1e-6, (X, Xh)

    rng = np.random.default_rng(2)
    for X in points[:5]:
        obs = [(cam, cam.project(X) + rng.normal(0, 0.1, 2)) for cam in cams]
        Xh = triangulate_point(obs)
        assert Xh is not None and np.linalg.norm(Xh - X) < 0.02, (X, Xh)

    behind = (0.0, 0.0, -5.0)
    obs = [(cam, cam.project(behind)) for cam in cams[:2]]
    assert triangulate_point(obs) is None
    assert triangulate_point(obs[:1]) is None
    print("  DLT 三角化：无噪声 / 0.1px 噪声 / 相机后方 ✓")


def test_triangulate_tracks_and_quality():
    points, cams = scene()
    T, J = 2, 5
    true_points = np.stack([points[:J], points[5:5 + J]])
    tracks = {cam.name: np.stack([cam.project(true_points[t]) for t in range(T)])
              for cam in cams}
    result = triangulate_tracks({cam.name: cam for cam in cams}, tracks)
    assert result.valid.all()
    assert np.nanmax(result.reproj_error) < 1e-6
    assert (result.n_views == 3).all()
    assert np.allclose(result.points, true_points, atol=1e-6)

    # 缺一个视角仍可三角化；只剩一个视角则无效
    tracks["cam1"][0, 0] = np.nan
    result2 = triangulate_tracks({cam.name: cam for cam in cams}, tracks)
    assert result2.valid[0, 0] and result2.n_views[0, 0] == 2
    tracks["cam2"][0, 0] = np.nan
    result3 = triangulate_tracks({cam.name: cam for cam in cams}, tracks)
    assert not result3.valid[0, 0] and result3.n_views[0, 0] == 1

    # 明显错误的观测被重投影误差拒绝
    tracks["cam1"][1, 1] += np.array([100.0, 100.0])
    result4 = triangulate_tracks({cam.name: cam for cam in cams}, tracks)
    assert not result4.valid[1, 1]

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "truth.npz"
        result.save(path)
        loaded = type(result).load(path)
        assert np.allclose(loaded.points, result.points, equal_nan=True)
        assert np.array_equal(loaded.valid, result.valid)
        assert loaded.camera_names == result.camera_names
    print("  轨迹三角化：缺视角 / 单视角 / 重投影拒绝 / npz 读写 ✓")


def _checkerboard_detections(n_views: int = 8) -> tuple[BoardPattern, list[BoardDetection]]:
    pattern = BoardPattern("checkerboard", 7, 6, 0.03)
    obj = pattern.object_points()
    K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    detections = []
    rng = np.random.default_rng(3)
    for _ in range(n_views):
        rvec = rng.uniform(-0.25, 0.25, 3)
        tvec = np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), 1.0])
        img_pts = cv2.projectPoints(obj, rvec, tvec, K, np.zeros(5))[0].reshape(-1, 2)
        detections.append(BoardDetection(obj.copy(), img_pts.astype(np.float32), (640, 480)))
    return pattern, detections


def test_calibration_from_synthetic_views():
    pattern, detections = _checkerboard_detections()
    calib = calibrate_from_detections(detections, camera_name="cam0")
    assert calib.rms < 0.5, calib.rms
    assert abs(calib.camera.K[0, 0] - 800.0) / 800.0 < 0.05, calib.camera.K
    assert abs(calib.camera.K[1, 1] - 800.0) / 800.0 < 0.05, calib.camera.K
    assert abs(calib.camera.K[0, 2] - 320.0) < 20
    assert abs(calib.camera.K[1, 2] - 240.0) < 20
    assert len(calib.view_poses) == len(detections)
    print("  棋盘格内参标定：rms / K 接近真值 ✓")


def test_duplicate_poses_rejected():
    _, detections = _checkerboard_detections(1)
    try:
        calibrate_from_detections([detections[0]] * 3)
    except ValueError:
        pass
    else:
        raise AssertionError("重复位姿应报错")

    det = detections[0]
    rng = np.random.default_rng(7)
    noisy = [
        BoardDetection(
            det.object_points.copy(),
            (det.image_points + rng.normal(0.0, 1.0, det.image_points.shape)).astype(np.float32),
            det.image_size)
        for _ in range(3)
    ]
    try:
        calibrate_from_detections(noisy)
    except ValueError:
        pass
    else:
        raise AssertionError("同一静止位姿加 1px 角点噪声仍应报错")
    print("  重复标定位姿：无噪声/1px 噪声都拒绝 ✓")


def test_symmetric_checkerboard_rejected():
    try:
        BoardPattern("checkerboard", 6, 6, 0.03)
    except ValueError:
        pass
    else:
        raise AssertionError("对称棋盘格应被拒绝")
    assert BoardPattern("checkerboard", 7, 6, 0.03).cols == 7
    print("  对称棋盘格（6x6）拒绝，7x6 接受 ✓")


def test_shared_board_resolution_mismatch():
    pattern = BoardPattern("checkerboard", 7, 6, 0.03)
    _, dets = _checkerboard_detections(8)
    intrinsic = {}
    for name in ("cam0", "cam1"):
        intrinsic[name] = [
            BoardDetection(d.object_points.copy(),
                           (d.image_points * 2.0).astype(np.float32),
                           (1280, 960))
            for d in dets
        ]
    extrinsic = {
        name: BoardDetection(dets[0].object_points.copy(),
                             dets[0].image_points.astype(np.float32), (640, 480))
        for name in ("cam0", "cam1")
    }
    try:
        calibrate_shared_board(intrinsic, extrinsic, pattern)
    except ValueError as exc:
        assert "尺寸" in str(exc), exc
    else:
        raise AssertionError("内外参图尺寸不一致应报错")
    print("  内参/外参图分辨率不一致：拒绝 ✓")


def test_shared_board_calibration():
    pattern = BoardPattern("checkerboard", 7, 6, 0.03)
    obj = pattern.object_points()
    f = 900.0
    K = np.array([[f, 0.0, 320.0], [0.0, f, 240.0], [0.0, 0.0, 1.0]])
    true_cams = [
        PinholeCamera("cam0", K, np.zeros(5), np.eye(3),
                      np.array([0.0, 0.0, 1.0]), (640, 480)),
        PinholeCamera("cam1", K, np.zeros(5),
                      cv2.Rodrigues(np.array([0.0, 0.08, 0.0]))[0],
                      np.array([0.1, 0.05, 1.0]), (640, 480)),
    ]
    rng = np.random.default_rng(4)
    intrinsic = {}
    extrinsic = {}
    for cam in true_cams:
        dets = []
        for _ in range(8):
            Rp, _ = cv2.Rodrigues(rng.uniform(-0.15, 0.15, 3))
            tp = cam.t + rng.uniform(-0.08, 0.08, 3)
            view = PinholeCamera(cam.name, cam.K, cam.dist, Rp @ cam.R, tp,
                                 cam.image_size)
            dets.append(BoardDetection(
                obj.copy(), view.project(obj).astype(np.float32), cam.image_size))
        intrinsic[cam.name] = dets
        extrinsic[cam.name] = BoardDetection(
            obj.copy(), cam.project(obj).astype(np.float32), cam.image_size)

    calib = calibrate_shared_board(intrinsic, extrinsic, pattern)
    assert calib.rms < 0.5, calib.rms
    for true in true_cams:
        cam = calib.cameras[true.name]
        assert abs(cam.K[0, 0] - f) / f < 0.05, cam.K
        assert np.linalg.norm(cam.t - true.t) < 0.02, (cam.t, true.t)
        cos = (np.trace(cam.R @ true.R.T) - 1.0) / 2.0
        assert cos > np.cos(np.radians(2.0)), (cam.R, true.R)
    print("  共享静止板：内参分离 + 外参 PnP + 联合 BA ✓")


def test_charuco_variable_corner_counts():
    pattern = BoardPattern("charuco", 5, 7, 0.03, marker_size=0.015)
    obj = pattern.object_points()
    K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    rng = np.random.default_rng(8)
    dets = []
    for count in (24, 21, 18, 23):
        idx = np.sort(rng.choice(len(obj), count, replace=False))
        Rp, _ = cv2.Rodrigues(rng.uniform(-0.2, 0.2, 3))
        t = np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), 1.0])
        cam = PinholeCamera("cam", K, np.zeros(5), Rp, t, (640, 480))
        dets.append(BoardDetection(
            obj[idx].copy(), cam.project(obj[idx]).astype(np.float32), (640, 480)))
    calib = calibrate_from_detections(dets, camera_name="charuco")
    assert abs(calib.camera.K[0, 0] - 800.0) / 800.0 < 0.05, calib.camera.K
    print("  ChArUco 部分可见：角点数不同仍可标定 ✓")


def _checkerboard_image() -> np.ndarray:
    cols, rows, square, margin = 7, 6, 60, 60
    w = (cols + 1) * square + 2 * margin
    h = (rows + 1) * square + 2 * margin
    img = np.full((h, w), 255, np.uint8)
    for y in range(rows + 1):
        for x in range(cols + 1):
            if (x + y) % 2 == 0:
                x0, y0 = margin + x * square, margin + y * square
                img[y0:y0 + square, x0:x0 + square] = 0
    return img


def test_checkerboard_detection_and_empty():
    pattern = BoardPattern("checkerboard", 7, 6, 0.03)
    detection = detect_board(_checkerboard_image(), pattern)
    assert detection is not None, "棋盘格检测失败"
    assert len(detection.image_points) == 42
    assert detect_board(np.zeros((0, 0), np.uint8), pattern) is None
    assert detect_board(np.zeros((10, 10), np.uint8), pattern) is None
    assert detect_board(np.zeros((0, 0, 3), np.uint8), pattern) is None
    assert detect_board(None, pattern) is None
    assert detect_board(np.zeros(10, np.uint8), pattern) is None
    print("  棋盘格检测 / 空图（灰度/彩色/None/1-D）与极小图返回 None ✓")


def test_charuco_detection_and_calibration_io():
    pattern = BoardPattern("charuco", 5, 7, 0.03, marker_size=0.015)
    assert np.allclose(pattern.object_points()[0], [0.0, 0.0, 0.0])
    image = pattern.charuco_board().generateImage((640, 480), marginSize=10)
    detection = detect_board(image, pattern)
    assert detection is not None, "ChArUco 检测失败"
    assert len(detection.image_points) >= 4
    assert detection.image_points.shape == detection.object_points.shape[:1] + (2,)

    cams = {
        "cam0": make_camera("cam0", np.eye(3), np.zeros(3)),
        "cam1": make_camera("cam1", np.eye(3), np.array([-0.5, 0.0, 0.0])),
    }
    calib = CalibrationSet(cams, "板第一个内角点", pattern, rms=0.3)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "calibration.json"
        calib.save(path)
        loaded = CalibrationSet.load(path)
        assert loaded.world_origin == calib.world_origin
        assert set(loaded.cameras) == {"cam0", "cam1"}
        assert np.allclose(loaded.cameras["cam1"].t, cams["cam1"].t)
        assert loaded.pattern.kind == "charuco"
    print("  ChArUco 检测 / 标定 JSON 读写 ✓")


def test_bundle_adjustment():
    points, true_cams = scene()
    observations = [
        Observation(ci, pi, tuple(cam.project(X)))  # type: ignore[arg-type]
        for ci, cam in enumerate(true_cams)
        for pi, X in enumerate(points)
    ]
    initial = []
    for cam in true_cams:
        R0, _ = cv2.Rodrigues(np.array([0.02, -0.015, 0.01]))
        initial.append(PinholeCamera(cam.name, cam.K, cam.dist,
                                     R0 @ cam.R, cam.t + np.array([0.02, -0.01, 0.01]),
                                     cam.image_size))
    result = bundle_adjust(initial, points, observations)
    assert result.final_rms < result.initial_rms
    assert result.final_rms < 0.5, result.final_rms
    for refined, true in zip(result.cameras, true_cams):
        assert np.linalg.norm(refined.t - true.t) < 0.01, (refined.t, true.t)

    fixed = bundle_adjust(initial, points, observations, fix_first=True)
    assert fixed.final_rms < fixed.initial_rms
    assert np.allclose(fixed.cameras[0].R, initial[0].R)
    assert result.n_invalid == 0
    print("  bundle adjustment：重投影下降 / 位姿回收 / 固定首相机 ✓")


def test_bundle_all_behind():
    points = np.array([[0.0, 0.0, -5.0], [0.1, 0.0, -5.0]])
    cam = make_camera("cam0", np.eye(3), np.zeros(3))
    obs = [Observation(0, 0, (0.0, 0.0)), Observation(0, 1, (0.0, 0.0))]
    result = bundle_adjust([cam], points, obs)
    assert not result.success
    assert np.isnan(result.final_rms)
    assert result.n_invalid == 2
    print("  bundle adjustment：全部点在后方 → 失败而不是 1000px 成功 ✓")


def test_bundle_partial_behind():
    points = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, -5.0]])
    cam = make_camera("cam0", np.eye(3), np.zeros(3))
    obs = [Observation(0, 0, (320.0, 240.0)), Observation(0, 1, (0.0, 0.0))]
    result = bundle_adjust([cam], points, obs)
    assert not result.success
    assert result.n_invalid == 1
    assert np.isfinite(result.final_rms)
    print("  bundle adjustment：部分观测在后方 → success=False / n_invalid=1 ✓")


def test_save_paths_and_rms():
    result = TriangulationResult(
        points=np.zeros((1, 1, 3)), valid=np.ones((1, 1), bool),
        reproj_error=np.zeros((1, 1)), n_views=np.ones((1, 1), int),
        camera_names=["cam0"], max_reproj_px=5.0)
    with tempfile.TemporaryDirectory() as td:
        for name in ("truth", "truth.data"):
            path = Path(td) / name
            saved = result.save(path)
            assert saved.exists() and saved.suffix == ".npz"
            loaded = TriangulationResult.load(path)
            assert loaded.camera_names == ["cam0"]

    pattern = BoardPattern("checkerboard", 7, 6, 0.03)
    cams = {"cam0": make_camera("cam0", np.eye(3), np.zeros(3))}
    calib = CalibrationSet(cams, "原点", pattern, rms=0.123456)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "calibration.json"
        calib.save(path)
        loaded = CalibrationSet.load(path)
        assert abs(loaded.rms - 0.123456) < 1e-12
    print("  save/load：无后缀路径 + rms 精确往返 ✓")


def _clap_envelope(times: list[float], sample_rate: int = 1000,
                   duration: float = 3.0) -> np.ndarray:
    t = np.arange(int(duration * sample_rate)) / sample_rate
    env = np.zeros_like(t)
    for c in times:
        env += np.exp(-0.5 * ((t - c) / 0.005) ** 2)
    return env


def test_clap_sync_and_half_frame():
    sr = 1000
    ref = [0.5, 1.0]
    envelopes = {
        "cam0": _clap_envelope(ref, sr),
        "cam1": _clap_envelope([0.51, 1.01], sr),
        "cam2": _clap_envelope([0.51, 1.03], sr),
        "cam3": _clap_envelope([0.5], sr),
    }
    result = sync_claps(envelopes, sr, fps=50.0)
    assert not result.reliable
    assert "cam2" in result.reasons and "偏移差" in result.reasons["cam2"]
    assert "cam3" in result.reasons
    assert abs(result.offsets_s["cam1"] + 0.01) < 0.002  # cam 时间 + offset = ref
    assert result.tolerance_s == 0.01

    only_good = sync_claps({k: envelopes[k] for k in ("cam0", "cam1")}, sr, fps=50.0)
    assert only_good.reliable
    assert abs(only_good.offsets_s["cam1"] + 0.01) < 0.002

    # 漏掉第一次拍手：拍手数不是期望的 2 次，不能靠索引配对猜偏移
    missed = sync_claps({"cam0": _clap_envelope([0.4, 0.9, 1.4], sr),
                         "cam1": _clap_envelope([0.9, 1.4], sr)}, sr, fps=50.0)
    assert not missed.reliable and "拍手数" in missed.reasons["cam1"]
    missed_both = sync_claps({"cam0": _clap_envelope([0.9, 1.4, 1.9], sr),
                              "cam1": _clap_envelope([0.4, 0.9, 1.4], sr)},
                             sr, fps=50.0)
    assert not missed_both.reliable

    # 恰半帧：浮点减法不能把它误判成超过
    exact = sync_claps({"cam0": _clap_envelope([0.5, 1.0], sr),
                        "cam1": _clap_envelope([0.51, 1.02], sr)}, sr, fps=50.0)
    assert exact.reliable

    audio = np.zeros(int(3.0 * sr))
    for c in ref:
        audio[int(c * sr)] = 1.0
    assert len(audio_envelope(audio, sr)) == len(audio)
    print("  拍手同步：偏移差 / 半帧 / 漏拍 / 恰半帧 / 包络 ✓")


if __name__ == "__main__":
    test_triangulation_roundtrip()
    test_triangulate_tracks_and_quality()
    test_calibration_from_synthetic_views()
    test_duplicate_poses_rejected()
    test_symmetric_checkerboard_rejected()
    test_shared_board_calibration()
    test_shared_board_resolution_mismatch()
    test_charuco_variable_corner_counts()
    test_checkerboard_detection_and_empty()
    test_charuco_detection_and_calibration_io()
    test_bundle_adjustment()
    test_bundle_all_behind()
    test_bundle_partial_behind()
    test_save_paths_and_rms()
    test_clap_sync_and_half_frame()
    print("全部通过")
