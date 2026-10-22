# 05 — MotionBERT 单目提升接入

Status: ready-for-agent
Blocked by: 03 — 三维骨架契约与提升适配器

## What to build

把 MotionBERT 接到提升适配器后面，真实视频端到端产出三维骨架与报告。输出带数据来源与置信度，模型权重可离线加载。

## Acceptance criteria

- [ ] 从现有二维关键点产物可复现三维骨架，不重跑姿态检测
- [ ] 一段真实视频端到端出报告，含三维指标与数据来源标注
- [ ] 模型许可为 Apache-2.0 一类，权重可离线加载，不依赖联网
- [ ] 长视频的处理时间与内存占用有记录
- [ ] 模型输出与适配器契约一致（root-relative、26 关节、可见性）

## Notes

基线模型为 MotionBERT（原生支持 Halpe-26，约 10 FPS）。VideoPose3D 因 CC BY-NC 只作内部研究对照，不进产品。来源规格：`.scratch/phase-2/spec.md`（三维提升）。
