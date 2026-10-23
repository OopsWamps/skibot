# 04 — JCS 六项三维指标

Status: resolved
Blocked by: 01 — 冻结三维指标定义与改写决策记录；02 — 指标注册表整改；03 — 三维骨架契约与提升适配器

## What to build

在三维骨架上实现六项 JCS 指标与三点夹角辅助量。评估流程同时产出二维与三维两套结果；视角门控取消，视角分类只进质量统计；二维报告作为回归对照保持不变。

## Acceptance criteria

- [x] 合成直立骨架的六项结果符合定义（伸直、零倾角等边界值）
- [x] 合成屈曲骨架的结果随角度单调变化
- [x] 报告每项带基准标识与三点夹角辅助量
- [x] 二维指标在相同输入下逐值不变（除新增字段）
- [x] 视角门控取消后所有帧参与计算，视角只进质量统计
- [x] 缺数据帧在六项上均为空值，不产生 0 值

## Notes

来源规格：`.scratch/phase-2/spec.md`（指标基准、报告与动画）。

## Answer

实现落在 `skibot/metrics3d.py`：

- `compute_series3d(Skeleton3D, slope_sag_deg=None, gravity_source=None)`
  逐帧产出六项指标的全部分量与两个辅助量。公式逐条对应 ADR-0001 §5；
  退化向量、缺关节、低于 `CONF_MIN_3D` 的关节一律 NaN，不插值、不补 0。
- 侧别按契约置信度选（`select_side`，平票取左）；膝内外翻两侧都算，
  主值取更大的一侧，正方向统一为膝向中线。
- 躯干前倾按重力参考：`gravity_source` 只接受 calibrated / adapter /
  image_vertical，unknown 时四个分量全为 NaN；坡面版本只在传入
  `slope_sag_deg` 时出数，否则 `trunk_lean_slope` 为 NaN。
- `assess(..., lifter=..., slope_sag_deg=...)`：`Assessment` 多带
  `series3d`；二维序列、报告、帧级视角逐位不变。三维指标不使用视角门控，
  所有 `Skeleton3D.valid_mask` 为真的帧都参与；视角归属改为写进
  `quality.view_frame_ratio`（sagittal / frontal / none 占比）。
- `report.py` 在 `metrics[]` 追加六项 `basis="3d"`：`reference` /
  `pct_in_range` / `band_percentile` / `verdict` 全为 null（三维带待
  issue 09/12），`median_deg` 无有效帧时为 null。`hip_flexion` 与
  `knee_flexion` 带 `helper`（三点夹角）；`trunk_lean` 带矢状/冠状/总倾角、
  坡面版本、重力与坡度来源；`knee_abduction` 带左右分侧值与主侧；
  H-Ssep 与 obliquity 带 `abs_median_deg`。
- `MockLifter` 的重力来源标为 `image_vertical`，不冒充适配器规范化输出。
- 二维报告回归：`test_skeleton3d.py` 与 `test_metrics3d.py` 都比对
  “有 lifter / 无 lifter”的二维条目逐值相等。

测试：新增 `tests/test_metrics3d.py`，覆盖 ADR-0001 §9 的八组验收用例，
外加 `assess` 报告接入、缺帧 null、视角门控取消与质量统计。全量相关测试
（metrics / assess / skeleton3d / render / web）通过，`mypy skibot` 无问题。

未定稿：置信度阈值仍为 0.3（需要 issue 05 的真实适配器与 issue 09 的
多机位素材）；坡面法向版本只接好了参数，现场 IMU 读数在 issue 08。

## Comments

2026-10-23：按 ADR-0001 §5 落地。三维参考带与相位聚合不在本票内，
留在 issue 09/10/12。

2026-10-23：独立验证工作流（六项指标各一个怀疑者 + 接入审查 + 边界审查）
确认 7 个问题，已修：

- `trunk_lean_total` 不再依赖肩线/前向轴退化，只依赖躯干轴与重力；
- `gravity_up` 支持 `(3,)` 或 `(T,3)`，逐帧指标取对应帧的重力；
- 报告顶层加 `data_source`（`Skeleton3D.source`），区分 mock 与真实模型；
- 退化阈值改为相对骨架尺度，坐标单位缩放 1e-8~1e8 结果不变；
- 二维条目补齐 `definition`；
- `report_schema` 与 `GLOSSARY` 的二维/三维表述修正。

新增对应测试，全部通过。`select_side` 在选中侧关节低置信时返回 NaN 的行为
按 ADR §7 保留，不按通用回退规则改。

2026-10-23：第二轮复核又发现 5 处，已修：

- 叉积 `u_t × s_body` 用无量纲阈值，骨架尺度到 1e12 时躯干前倾不再静默变 NaN；
  缩放测试扩到 1e12；
- `quality.view_frame_ratio` 只在传入 lifter 时写入，无 lifter 的二维报告
  与改动前逐位一致；
- GLOSSARY 的视角表述限定为二维列；
- issue 03 的 `gravity_up` 契约说明同步为 `(3,)` 或 `(T,3)`；
- `test_web` 增加三维条目（reference/verdict 为 null、带 helper、median 为
  null）的报告页渲染覆盖。

相关测试与 `mypy skibot` 全部通过。
