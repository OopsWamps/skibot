# 01 — 冻结三维指标定义与改写决策记录

Status: resolved
Blocked by: None — can start immediately

## What to build

在实地采集与指标实现之前，把六项三维指标的定义、marker/segment 方案、零点、符号和参考系写成 ADR；改写一期决策 #3 与 #10；记录参考带的取舍。定义冻结后改定义等于重标真值，所以这张票必须最先完成。

## Acceptance criteria

- [x] ADR 覆盖六项指标：屈髋角、膝弯曲角、膝内扣角、躯干前倾角、肩髋横断面（H-Ssep）、肩髋冠状面（thoraco-pelvic obliquity）的三维定义与公式
- [x] 写明使用哪些 Halpe-26 关节点、近端段选择（骨盆还是下背）与左右腿符号规则
- [x] 写明零点与参考系：重力、坡面、体固各自用在哪一项
- [x] 改写一期决策 #3：从"单目三维明确不做"改为"二维为主、三维用于动画与指标，指标迁移以三层验证为前提"
- [x] 改写一期决策 #10 的放行标准为三维口径（within-athlete、按方位角分层、中位 ≤5°、p90 ≤10°）
- [x] 记录参考带取舍：advanced 带由验证过的管线跑 EPFL 生成，不在三维真值上重算，理由与误差上界写清
- [x] 文档可被采集协议与后续实现票直接引用

## Notes

来源规格：`.scratch/phase-2/spec.md`（指标基准、参考区间、决策记录）。

## Answer

定义冻结在 `docs/adr/0001-3d-metric-definitions.md`，参考带口径在
`docs/adr/0002-reference-band-strategy.md`；一期决策 #3、#10 已在
`docs/DECISIONS.md` 改写，并新增「二期决策（2026-10-23）」一节。

- 六项主指标：`hip_flexion`、`knee_flexion`、`trunk_lean`、
  `knee_abduction`、`hip_shoulder_separation`、`thoraco_pelvic_obliquity`。
  `knee_valgus` 迁到中性名 `knee_abduction`，正方向 = 膝向中线。
- 睿君 2026-10-23 拍板：屈髋近端段用下背–大腿（躯干轴）；坡度 β 用现场
  手机 IMU 倾角仪测量。
- 关节点只用 Halpe-26 的 5/6、11–16；20–25（脚部）保留在骨架契约里给动画
  与质量检查，不进六项定义；17–19 不用，躯干/骨盆中点由左右点平均。
- 零点与参考系：体固用于屈髋、膝弯、膝内外翻、肩髋两项；重力用于躯干前倾；
  坡面法向版是躯干前倾的派生量。
- 参考带：advanced 用验证过的三维管线跑 EPFL，按相位 p25~p75；带宽按验收
  p90 误差对称加宽；novice/intermediate 的相位形状从 advanced 加性外推，
  量纲不同的指标三维带暂为 `null`。
- 未决项（不阻塞定义冻结）：runtime 重力来源（水平相机假设 vs 适配器规范化
  输出）、置信度阈值 0.3、髋旋转泄漏进 `knee_abduction` 的量级。均由
  issue 04/09 验证；超出误差预算则新开 ADR 改定义。

本票只改文档，无代码改动，未跑测试。

## Comments

2026-10-23：从 `.scratch/phase-2/issues/01` 认领并完成。下游可直接引用
ADR-0001 §2 的指标表、§3 的骨架契约、§5 的公式与符号，以及 ADR-0002 §2
的参考带规则。
