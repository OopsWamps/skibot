# 05 — MotionBERT 单目提升接入

Status: resolved
Blocked by: 03 — 三维骨架契约与提升适配器

## What to build

把 MotionBERT 接到提升适配器后面，真实视频端到端产出三维骨架与报告。输出带数据来源与置信度，模型权重可离线加载。

## Acceptance criteria

- [x] 从现有二维关键点产物可复现三维骨架，不重跑姿态检测
- [x] 一段真实视频端到端出报告，含三维指标与数据来源标注
- [x] 模型许可为 Apache-2.0 一类，权重可离线加载，不依赖联网
- [x] 长视频的处理时间与内存占用有记录
- [x] 模型输出与适配器契约一致（root-relative、26 关节、可见性）

## Answer

实现 `skibot/motionbert.py`（MotionBERTLifter）+ `skibot/mbmodel/`（vendor 的
DSTformer 模型定义，Apache-2.0，license 随目录附上）。CLI `--lifter motionbert`
开启三维旁路（默认 none，二维报告逐位不变，spec「三维提升」的回归基线保留）；
`scripts/lift_trc.py` 从已有 px trc 复现三维骨架；`scripts/check_motionbert.py`
环境自检（torch/CUDA/权重/前向）。

预处理复刻官方 wild 推理路径（`infer_wild.py` + `dataset_wild.py`）：
halpe2h36m 折到 H36M-17（脊点 = 颈与髋中点的中点）→ 整段 bbox 归一化 [-1,1]
（crop_scale ratio=1，确定性）→ 243 帧切窗（尾窗可短）→ 单遍推理（无翻转
平均，确定性）。输出与输入同尺度，不反归一化——指标只用角度，尺度不变；
不输出世界轨迹与绝对位置。

输出帧两处修正（真实素材核对，见 Comments）：模型输出 y 朝下，绕 x 轴转
180°（y、z 取反，保持手性）转到契约朝向；H36M-17 的 0 号 pelvis 通道输出
不是解剖位点，Halpe 髋中点（19）由左右髋中点重建。17 点映射回 Halpe-26，
无对应的点（眼耳、脚趾脚跟）保持 NaN + conf 0；置信度沿用输入二维检测。
重力继承训练数据约定，如实标 `gravity_source="adapter"`。

torch 是可选依赖（pyproject `[lift3d]` extra），默认路径（`--lifter none`）
不导入 torch。测试 `tests/test_motionbert.py`：映射表逐项核对、bbox 归一化
与退化分支、切窗与契约（注入假推理函数，不加载 torch）、trc 缺测 NaN 不进
网络、输出帧旋转与髋中点重建、assess 标注与二维回归、真实权重集成用例
（torch/权重缺席时跳过）。

## Comments

### 2026-10-26 外部依赖就位（安装记录，实施未开始）

- torch 2.14.1+cu132（uv 安装，download.pytorch.org 直连；驱动 CUDA 13.4，RTX 3050 Ti 4GB 实测 CUDA 可用）。推理附带依赖 easydict、pyyaml，tqdm 原有。
- 官方权重（Apache-2.0）：HuggingFace 官方仓 `walterzhu/MotionBERT` 的 `checkpoint/pose3d/FT_MB_release_MB_ft_h36m/best_epoch.bin`（170MB），经 hf-mirror.com 下载到 `models/motionbert/FT_MB_release_MB_ft_h36m/`（已 gitignore）。
- 官方代码（Apache-2.0）：GitHub tarball 经 ghproxy.net 解包到 `third_party/MotionBERT`（已 gitignore）。推理路径所需文件齐备：`lib/model/DSTformer.py`（lift 主干）、`drop.py`、`lib/utils/{learning,tools,utils_data}.py`、`lib/data/dataset_wild.py`。
- 权重与 `configs/pose3d/MB_ft_h36m.yaml` 配对（dim_feat=512、depth=5、root-relative）；checkpoint 的 state_dict 键带 `module.` 前缀（DataParallel 存档），加载前需剥除。
- 自检通过：`.venv/Scripts/python.exe scripts/check_motionbert.py` —— 权重 strict 加载，(1,243,17,3)→(1,243,17,3) 前向 CPU/CUDA 输出最大差 1.4e-05。

### 2026-10-26 实施与验证记录

- **验收 1（trc 复现三维，不重跑检测）**：`scripts/lift_trc.py` 跑
  BV1fN6RYbEdP_15fps 的 person11 轨迹（15522 帧，17 分钟长视频）：CUDA 上
  10.63s（约 1460 FPS），CUDA 峰值显存 312MB；有效帧 5301/15522（该轨迹
  本身 88% 帧缺测，比例与二维一致）。
- **验收 2（端到端）**：EPFL 37HtpPpdvp0/0.mp4（1280 帧 25fps）全链路：
  打码 → Sports2D（6 轨迹）→ 自动选人（person 3，尺寸 78%+位移 100%）→
  提升出 `report.json`（`data_source: "motionbert"`，六项三维条目全出，
  `gravity_source: "adapter"`，trunk_lean 中位 6.8°、有效帧 836/836）→
  自绘标注视频。Sports2D 占绝大部分耗时，836 帧提升在管线内秒级完成。
- **实施中发现并修正的两个输出帧问题**（合成冒烟测不出来，真实素材才暴露）：
  1. 模型输出 y 朝下（鼻在髋上方为负）。未修正时 trunk_lean 中位 −172°。
     修正：y、z 取反（绕 x 轴 180°，保持手性），gravity_up 回 (0,1,0)。
     修正后 trunk_lean_total 中位 9.8°（p10–p90：5.2–18.9°），量级合理。
  2. H36M-17 0 号关节（pelvis）输出不是解剖位点：偏离髋中点达身高 2 倍、
     波动是其他关节 4 倍（平滑曲线，非拼接噪声，疑为根通道残留）。Halpe
     髋中点（19）改由左右髋（h36m 1/4）中点重建，与契约 root-relative
     原点定义一致。
- **trc 缺测语义差异**：trc 缺测关节坐标是 NaN（官方 AlphaPose json 记 0），
  NaN 进网络会污染整个窗口。lift 入口把非有限坐标清零，由 conf=0 与
  valid_mask 排除；回归测试钉住。
- **精度边界（留给后续票）**：本素材上 knee_flexion 中位 115.8°、
  knee_abduction 中位 28.6°，偏大——迁移分布缺口（spec 风险节）与单目深度
  歧义在此量级可见。角度量级自洽、契约一致，精度对账归 issue 07（文献锚点）
  与 09（多机位真值）。
- **素材敏感度（非本票缺陷）**：BV1fN6RYbEdP 的 720p 副本跑 Sports2D 时
  全部轨迹被删（人物过小、置信度不足，`Less than 10 valid frames`），同一
  素材一期原片可跑——远景小人仍是二维前端的失败模式，预检（issue 13）的
  人物占比门部分覆盖。
- **测试**：`tests/test_motionbert.py` 全过（含真实权重集成用例：CUDA 300
  帧 0.9~1.1s，有效帧 100%，峰值 312MB）；`tests/test_metrics3d.py`、
  `tests/test_phases.py` 回归通过；`mypy skibot` 29 文件无问题。
