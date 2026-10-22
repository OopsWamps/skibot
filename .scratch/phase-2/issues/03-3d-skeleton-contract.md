# 03 — 三维骨架契约与提升适配器

Status: resolved
Blocked by: 01 — 冻结三维指标定义与改写决策记录

## What to build

定义三维骨架的数据契约（26 关节、root-relative、可见性/置信度）与"二维关键点 → 三维骨架"的提升适配器接口。先用假模型把评估流程端到端跑通，换真实模型时不用改评估流程。

## Acceptance criteria

- [x] 契约写清坐标约定（root-relative、重力参考）、单位、缺帧与置信度语义
- [x] 适配器接口固定，替换模型不影响评估流程
- [x] 假模型可由合成二维骨架驱动，评估流程产出三维结果
- [x] 缺帧、全零帧、低置信帧有明确且可测的行为

## Notes

来源规格：`.scratch/phase-2/spec.md`（三维提升）。

## Answer

实现落在 `skibot/skeleton3d.py`：

- `Skeleton3D`：`joints (T,26,3)`、`confidence (T,26)`、`gravity_up (3,)`、
  `source`、`meta`。构造时校验形状、置信度范围、重力单位向量，并对髋点齐全
  的帧强制减去髋中点。`root_defined` 标出原点可定义的帧；`valid_mask` 要求
  至少 4 个核心关节坐标有限、置信度 ≥ `CONF_MIN_3D`（暂定 0.3），且不是
  全零帧。
- `Lifter` Protocol：`lift(keypoints: (T,26,>=3)) -> Skeleton3D`。换模型只换
  适配器，评估流程与指标代码不动。
- `MockLifter`：二维骨架放 z=0、翻转图像 y、居中交给契约；只证明形状与流程，
  不提供精度结论。
- `assess(..., lifter=...)`：传入时 `Assessment.skeleton3d` 多带一份骨架；
  二维报告、序列、帧级视角逐位不变（测试逐项比对）。帧数不匹配抛
  `ValueError`。
- 缺帧：关节坐标 NaN + confidence 0；全零帧按缺帧处理；低置信关节保留坐标
  但不进指标。语义写在模块 docstring，坐标约定引用 ADR-0001 §3。

测试：新增 `tests/test_skeleton3d.py`（契约校验、假模型翻转/居中/缺测、
缺帧/全零/低置信的 valid_mask、assess 接入后二维输出不变）。全量测试通过，
`mypy skibot` 无问题。

未做：MotionBERT 真实适配器（issue 05）与三维指标计算（issue 04）。

## Comments

2026-10-23：按 ADR-0001 §3 落地。`CONF_MIN_3D=0.3` 与 `MIN_CORE_JOINTS=4`
是暂定值，issue 04 用合成与真实素材定稿。
