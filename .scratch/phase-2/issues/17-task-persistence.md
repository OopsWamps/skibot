# 17 — 任务表持久化

Status: ready-for-agent
Blocked by: None — can start immediately

## What to build

任务元数据落成 JSON，原子写（临时文件 + rename）。服务重启后，排队中的任务重新入队；运行中被中断的标为失败并允许重试；原始文件名、水平档、打码开关原样恢复。

## Acceptance criteria

- [ ] 重启后排队任务继续执行
- [ ] 运行中中断的任务标为失败，可手动重试
- [ ] 原始文件名、水平档、打码开关恢复
- [ ] 写入为原子操作，文件损坏不阻塞启动
- [ ] 已完成任务的恢复行为与现状一致

## Notes

来源规格：`.scratch/phase-2/spec.md`（Web 与产品）。
