# 13 — 上传预检门控

Status: ready-for-agent
Blocked by: None — can start immediately

## What to build

上传时用廉价信号预检：容器与编码、分辨率、时长，抽几帧测人物占比。不合格当场拒绝，给出原因与重拍建议，不进入分析队列。

## Acceptance criteria

- [ ] 预检判据可配置，默认值有依据
- [ ] 拒绝响应包含具体原因与重拍建议
- [ ] 合格素材正常入队，预检耗时不影响上传体验
- [ ] 页面能展示拒绝原因
- [ ] 抽帧检测失败时按"无法判定"处理，不误杀

## Notes

来源规格：`.scratch/phase-2/spec.md`（质量门控）。
