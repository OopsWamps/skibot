# 06 — 多机位标定与三角化工具

Status: resolved
Blocked by: 01 — 冻结三维指标定义与改写决策记录

## What to build

离线工具：多路视频同步、相机标定、三维关节真值三角化，输出可复用的标定文件与真值轨迹，供验收脚本消费。

## Acceptance criteria

- [x] 合成相机测试：已知三维点投影到两视图后可还原，误差在数值容差内
- [x] 支持拍手同步与半帧判据，不满足的段落被标记而不是静默通过
- [x] 支持 ChArUco/棋盘格标定与联合 bundle adjustment
- [x] 标定结果含相机内参、位姿与世界原点
- [x] 三角化输出可被验收脚本直接读取

## Notes

来源规格：`.scratch/phase-2/spec.md`（验证与采集）。

## Answer

实现落在 `skibot/multiview.py`：

- `PinholeCamera`：K/dist/world→camera 的 R/t，`project`（含 OpenCV 五项
  畸变）、`undistort`、JSON 读写。
- 标定板：`BoardPattern` 支持棋盘格（内角点数）与 ChArUco（方格数、
  marker_size、字典）。棋盘格要求内角点数之和为奇数（7x6 或 6x7），
  对称板（如 6x6）直接拒绝，避免 180° 旋转下角点反序；ChArUco 内角点
  平移到第一个为 (0,0,0)。`detect_board` 棋盘格只用
  `findChessboardCorners` + 亚像素，保证所有相机角点顺序一致；ChArUco
  按 ID 配对。空图、彩色空图、None、1-D 输入与 OpenCV 异常返回 None。
- `calibrate_from_detections` / `calibrate_camera`：`calibrateCameraExtended`
  内参标定，返回 K、dist、每视角位姿、RMS 与 fx/fy 标准差。用 fx/fy
  不确定度 + 每视角旋转/平移覆盖检查退化；重复位姿（含加 1px 角点
  噪声的同一静止位姿）直接报错。允许 ChArUco 各视角角点数不同。
- `calibrate_shared_board(intrinsic_detections, extrinsic_detections,
  pattern)`：内参图集（板移动的多个不同位姿）与外参图集（每台相机
  同一静止板位姿各一张）必须分开，且两套图分辨率一致；外参用静止板
  PnP，最后用板点做联合 bundle adjustment，只有 success 且
  n_invalid==0 才接受。`calibrate_shared_board_from_images` 是检测入口。
  世界原点 = 板第一个内角点，+x/+y 按 OpenCV 棋盘格角点顺序、+z 垂直
  板面向外，单位由 square_size 决定。
- `bundle_adjust`：固定 K/dist 与三维板点，优化各相机位姿；`fix_first`
  可固定参考相机。返回初/终重投影 RMS、优化后的相机与 `n_invalid`；
  只要有观测在相机后方（n_invalid>0）就 `success=False`；全部在后方时
  RMS 为 NaN，不把罚项当成功。
- `triangulate_point`：多视角 DLT，先按相机畸变去畸变，点落在任一相机
  后方或退化时返回 None。`triangulate_tracks`：逐帧逐点收集至少两个
  视角，平均重投影误差超过阈值即无效；输出
  `(T,J,3)` 点、`valid`、`reproj_error`、`n_views`。
  `TriangulationResult.save/load` 写读 `.npz`（含 camera_names、
  max_reproj_px、version）；无 `.npz` 后缀时保存追加 `.npz`，load 对原路径
  自动回退查找。验收脚本可直接 `np.load` 或 `TriangulationResult.load`。
- 同步：`audio_envelope` 做短窗包络，`detect_claps` 检测拍手时刻，
  `sync_claps` 要求每路恰好 `expected_claps` 次（默认 2 次）拍手，用两次
  拍手的偏移差与半帧比较；多检/少检、偏移差超过 `tolerance_frames / fps`
  （默认半帧）即整条不可靠并写原因。恰半帧用 `+1e-9` 吸收浮点减法误差。
  `offsets_s` 的语义是 `cam_time + offset = ref_time`。已知限制：两路各
  恰好两次、其中一路漏第一次而补检了后续事件、且拍手间隔恰好相同时，
  索引配对存在 ±间隔 的固有歧义；需要外部录制起始时间先验或人工确认。
- `CalibrationSet`：JSON 含 version、world_origin、pattern、rms 与各相机
  K/dist/R/t/image_size；rms 保存原值不四舍五入，`save/load` 可精确往返。

测试：新增 `tests/test_multiview.py`，覆盖合成相机无噪声/0.1px 噪声
三角化、相机后方、轨迹缺视角/单视角/重投影拒绝/npz 往返、棋盘格内参
回收、重复位姿拒绝（无噪声与 1px 噪声）、对称棋盘格拒绝、共享静止板
（内参分离 + PnP + 联合 BA）、内外参图分辨率不一致拒绝、ChArUco 变长
角点标定、棋盘格渲染检测/空图（灰度/彩色/None/1-D）、ChArUco 检测与
原点、bundle adjustment 重投影下降/位姿回收/全部或部分点在后方、
无后缀路径 save/load、rms 精确往返，以及拍手同步的偏移差/半帧/漏拍/
恰半帧/包络。`mypy skibot` 21 个源文件无问题。

未做：独立 CLI（库 + save/load 已可被验收脚本调用）；真实雪场标定；
音频文件解码（调用方用 `audio_envelope` 提供包络）；多相机时间插值。
`calibrate_shared_board` 要求内参图集（板移动到多个不同位姿）与外参
图集（每台相机同一静止板位姿各一张）分开传入；现场采集协议需要保证
每个相机至少两个不同位姿的内参图。

## Comments

2026-10-23：独立验证工作流确认 11 处，已修：

- `calibrate_shared_board` 分离内参图集与外参图集，不再把移动板视角
  当成固定板点观测；新增 `calibrate_shared_board_from_images` 检测入口。
- `calibrate_from_detections` 检查检测点像素离散度与 K 量级/主点范围，
  重复位姿或退化内参直接报错。
- 棋盘格只用 `findChessboardCorners` + 亚像素，统一角点顺序；SB 与
  经典检测器混用会让两台相机板坐标系差 180°。
- `sync_claps` 要求每路恰好两次拍手，多检/少检都不可靠；恰半帧加
  `1e-9` 容差，避免浮点减法误判。
- `TriangulationResult.save` 对无 `.npz` 后缀追加而不是替换，`load`
  对原路径自动回退。
- `CalibrationSet` 的 rms 保存原值，不四舍五入。
- `bundle_adjust` 暴露 `n_invalid`；全部观测在相机后方时 `success=False`、
  RMS 为 NaN，不把 1e3 罚项当成功。
- `detect_board` 对空图/极小图与 OpenCV 异常返回 None。
- 测试补共享静止板、重复位姿、渲染棋盘格检测/空图、全部点在后方、
  无后缀路径、rms 往返、漏拍与恰半帧。

相关测试与 `mypy skibot` 全部通过。

2026-10-23：第二轮复核确认 8 处，已修：

- 重复位姿检查改用 `calibrateCameraExtended` 的 fx/fy 标准差与每视角
  旋转/平移覆盖；1px 角点噪声不再能把同一静止位姿抬过阈值。
- 对称棋盘格（cols+rows 为偶数）直接拒绝，避免 180° 角点反序。
- 共享板标定检查内参图与外参图分辨率一致，不一致报错。
- `bundle_adjust` 只要有观测在相机后方（`n_invalid>0`）就
  `success=False`；`calibrate_shared_board` 同时检查 success 与 n_invalid。
- 允许 ChArUco 各视角角点数不同（部分可见），不再强制等长。
- `detect_board` 对空彩色图、None、1-D 输入在进入 OpenCV 前返回 None。
- ChArUco 板点平移到第一个内角点为 (0,0,0)，与世界原点文档一致。
- 拍手漏拍的等数歧义写入 docstring；需要外部先验或人工确认。

相关测试与 `mypy skibot` 全部通过。
