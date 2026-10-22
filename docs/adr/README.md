# 架构决策记录（ADR）

每条 ADR 记录一次不可从代码直接看出的取舍：背景、决定、后果。
定义类 ADR 一经接受即冻结；改定义要新开 ADR，并重标受影响的真值与参考带。

| 编号 | 标题 | 状态 |
|---|---|---|
| [0001](0001-3d-metric-definitions.md) | 三维指标定义与坐标约定 | Accepted, 2026-10-23 |
| [0002](0002-reference-band-strategy.md) | 三维参考带的生成口径与过渡策略 | Accepted, 2026-10-23 |

关联文档：

- 二期规格：`.scratch/phase-2/spec.md`
- 决策记录：`docs/DECISIONS.md`（一期 #3、#10 已按 0001/0002 改写）
- 文献数值：`docs/lit_angle_values.md`
- 任务票：`.scratch/phase-2/issues/`

## 写新 ADR

文件名 `NNNN-<slug>.md`，编号递增；状态用 `Proposed` / `Accepted` / `Superseded by NNNN`。
被取代的 ADR 不删除，在顶部标注取代关系。
