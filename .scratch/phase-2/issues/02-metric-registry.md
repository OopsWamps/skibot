# 02 — 指标注册表整改：更名、六项、基准标识

Status: resolved
Blocked by: 01 — 冻结三维指标定义与改写决策记录

## What to build

把 knee_valgus 换成与新定义一致的中性名称，注册表扩为六项（肩髋对齐拆成两项），每项带二维/三维基准标识。报告、参考值表、渲染和测试同步更新，旧字段有兼容或迁移说明。

## Acceptance criteria

- [x] 报告、参考值表、注册表、文档使用同一套新名字
- [x] 注册表能表达六项指标与每项的基准（二维/三维）
- [x] 旧报告字段的兼容或迁移说明写入文档
- [x] 现有测试全绿；二维报告除新增字段与更名外，结构与数值不变

## Notes

这是一次小范围的宽改动：注册表被报告、渲染和测试共同消费，优先做成"旧名仍可用、新名先加"的过渡形态，再迁移调用点，避免一次性改名打断全绿。来源规格：`.scratch/phase-2/spec.md`（指标基准）。

## Answer

注册表在 `skibot/metrics.py`：

- `METRICS`：六项规范指标，每项带 `plane`、`bases`（`2d`/`3d`）、`phase`、
  `aliases`、`definitions`。
- `LEGACY_METRICS`：`shoulder_hip_align` 二维遗留合并键，`replaced_by` 指向
  `hip_shoulder_separation` 与 `thoraco_pelvic_obliquity`。
- `TWO_D_METRICS`：二维管线实际产出的五键，报告与渲染按它迭代。
- `metrics_for_basis("2d"|"3d")`、`resolve_metric`、`metric_label`、
  `metric_plane` 是消费入口。

更名与兼容：

- `knee_valgus` → `knee_abduction`。`compute_series` 同时返回
  `knee_valgus` 别名（与 `knee_abduction` 同一数组）；`reference.lookup`
  按别名解析，旧 TOML 只有旧键时回退。
- 报告 schema 升到 `skibot.report/0.2`；指标条目新增 `basis`、`definition`、
  `legacy`、`replaced_by`、`note`。二维报告的数值与结构除这些字段与更名外
  不变。
- `reference_values.toml` 的 `[knee_valgus]` 节更名为 `[knee_abduction]`，
  注释写明仍是二维投影口径；三维带未生成。`[shoulder_hip_align]` 保留并标
  为遗留。
- 迁移说明写在 `docs/report_schema.md`；`docs/GLOSSARY.md` 补旧名兼容词条。

验证：`test_metrics.py`、`test_assess.py`、`test_render.py`、`test_trc.py`、
`test_epfl.py`、`test_select_person.py`、`test_web.py` 全部通过；
`mypy skibot` 无问题。

未做：三维六项的实际计算与三维报告列（issue 04/12）；注册表已留好接口。

## Comments

2026-10-23：按"新名先加、旧名兼容"完成。三维指标尚未实现，报告暂时只有
`basis=2d` 的二维五项。
