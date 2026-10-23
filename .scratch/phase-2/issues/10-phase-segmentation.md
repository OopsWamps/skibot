# 10 — 三维相位切分与逐弯置信度

Status: resolved
Blocked by: 01 — 冻结三维指标定义与改写决策记录；03 — 三维骨架契约与提升适配器

## What to build

从三维侧倾切出弯与三相位，输出弯区间、换刃边界、相位归属与逐弯置信度。低幅、过短、首尾不完整的弯被丢弃；低置信弯的帧不进相位聚合。

## Acceptance criteria

- [x] 合成侧倾序列：弯数、边界位置、相位归属符合定义
- [x] 半周期幅值 <6°（换刃零点到本弯最大侧倾）、单弯时长 <0.5s、首尾不完整的弯被丢弃
- [x] 低置信弯的帧不进相位聚合
- [x] 整段无合格弯时回退全程数据并注明
- [x] 输出包含逐弯区间、边界时刻、置信度与丢弃原因

## Notes

信号为相对重力方向的带符号侧倾；画面横向轨迹与膝弯周期做交叉校验。来源规格：`.scratch/phase-2/spec.md`（相位划分）。

## Answer

实现落在 `skibot/phases.py`：

- `segment_phases(side_lean, fps, knee=, lateral=)`：主信号是
  `trunk_lean_frontal`。过零切半周期；零平台按首/中/尾分别取最后一个零、
  平台中心、第一个零，避免重复过零。首尾没有两侧换刃的段记 `incomplete`。
  半周期幅值（换刃零点到本弯最大侧倾）<6°、时长 <0.5s（用未取整的换刃
  位置）、区间内无有限值分别记 `low_amplitude` / `too_short` /
  `no_finite_data`；换刃间隔不足一帧记 `degenerate`。
- 切片边界用 `ceil`，不把换刃前一侧的样本并进本弯；幅值用峰值绝对值，
  NaN 缺口不会像区间 max-min 那样低估。极值贴边允许 entry 或 exit 为空。
- 相位：以本弯最大侧倾的 50% 为界，`|lean| ≥ 50%` 归控制段，前为入弯、
  后为出弯。`PhaseInterval` 是半开区间，同时给帧号与秒；`Turn` 另给
  未取整的起止时刻与 `amplitude_deg`。
- 置信度：横向轨迹只在主信号接近零的帧上拟合漂移，过零点对齐换刃；
  膝弯的局部极小对齐换刃、局部极大对齐侧倾极值；按 0.5s 容差取平均。
  低于 0.5 的弯记 `low_confidence`，不进 `turns`。没有交叉信号时置信度
  为 `None`，不误判。
- `PhaseSegmentation` 含 `turns`（只含合格弯）、`discarded`（区间、原因、
  置信度）、`fps` 与 `fallback`；没有合格弯时 `fallback="no_valid_turns"`。
- `phase_labels` 输出逐帧 `entry/control/exit`，未进合格弯的帧为空。
- `aggregate_phases` 只聚合 `hip_flexion`、`knee_flexion`、
  `trunk_lean_sagittal`，只用合格弯的帧；无数返回 `None`，调用方回退
  全程值并注明。全部为 NaN 的相位中位数为 `None`，不写 0。
- `assess(..., fps=...)`：有 lifter 且给 fps 时计算
  `Assessment.phases`；缺 fps 的抢救路径为 `None`。CLI 已把
  `probe_video` 的帧率传进 assess。

测试：新增 `tests/test_phases.py`，覆盖两个三角弯的边界与 50% 相位、
首尾丢弃、低幅/过短丢弃、低置信弯不进聚合、无过零回退、非法输入、
零平台对称、重复过零、半帧时长、极值贴边、跨 NaN 幅值、3s 长弯
横向去趋势，以及 assess 的 fps 接入。相关旧测试全部通过，
`mypy skibot` 无问题。

未做：报告的 `phases` 对象与相位参考带在 issue 12；人工标注验收在
issue 11（阈值 6°/0.5s/0.5/0.5s 仍是 TODO_VALIDATE）。

## Comments

2026-10-23：独立验证工作流确认 8 处，已修：

- 换刃切片从 `round` 改 `ceil`，不把换刃前另一侧的样本并入本弯；
- 幅值改用峰值绝对值，跨 NaN 缺口不再低估；`Turn` 新增 `amplitude_deg`；
- 横向去趋势只在主信号接近零的帧上拟合，3s 长弯不再被当成漂移减掉；
- 时长用未取整的换刃位置，半帧边界不再从 0.5s 缩成 0.4s；
- 去掉规格外的“极值贴边 = degenerate”，允许空 entry/exit；
- 首尾零平台对称处理（首取最后一个零、尾取第一个零）；
- 零平台分支更新 `last_idx`/`last_sign`，消除重复过零与零长度丢弃；
- README 二期进度同步为“相位切分已实现”。

新增对应测试，全部通过；`mypy skibot` 无问题。

2026-10-23：第二轮复核又修 5 处：

- 交叉信号加显著性门槛（横向去趋势后 <0.5 px、膝弯范围 <0.5° 视为无信号）；
  纯漂移、平坦 + 1e-6 噪声不再伪造置信度，也不会把合格弯误丢成低置信。
- 去趋势只在主信号过零附近、首尾完整段内拟合，10 个 3s 弯的长序列
  不再丢最后一个弯。
- 测试补了能区分修复的用例：3.4 帧换刃（ceil vs round）、三帧前导零平台、
  精确时长 0.49s、噪声门槛、长序列去趋势。
- 常量/参数改名 `MIN_AMPLITUDE_DEG` / `min_amplitude_deg`；
  `peak_to_peak_deg` 只作区间 max-min 诊断，不参与筛选。
- GLOSSARY、规格与验收项从“峰峰值”改为“半周期幅值”。

相关测试与 `mypy skibot` 全部通过。
