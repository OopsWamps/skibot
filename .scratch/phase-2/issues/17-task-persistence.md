# 17 — 任务表持久化

Status: resolved
Blocked by: None — can start immediately

## What to build

任务元数据落成 JSON，原子写（临时文件 + rename）。服务重启后，排队中的任务重新入队；运行中被中断的标为失败并允许重试；原始文件名、水平档、打码开关原样恢复。

## Acceptance criteria

- [x] 重启后排队任务继续执行
- [x] 运行中中断的任务标为失败，可手动重试
- [x] 原始文件名、水平档、打码开关恢复
- [x] 写入为原子操作，文件损坏不阻塞启动
- [x] 已完成任务的恢复行为与现状一致

## Notes

来源规格：`.scratch/phase-2/spec.md`（Web 与产品）。

## Answer

实现落在 `skibot/web/tasks.py`、`skibot/web/app.py`、`skibot/web/templates/tasks.html`：

- 任务元数据整体落成结果目录下的 `tasks.json`（`serve` 默认
  `web_results/tasks.json`）。`_save` 写临时文件后 `os.replace`，进程崩溃
  不留半个文件；Windows 上杀毒/索引器短暂锁文件时短退避重试三次。路径统一
  `resolve()` 成绝对路径，换工作目录启动不丢队列。
- 启动恢复 `_load`：排队任务保持排队（worker 接手继续执行）；运行中任务
  说明上次重启时被中断，标失败并写“服务重启时任务被中断，可重试”；已完成
  任务照旧可查看（报告被删的跳过，与旧行为一致）；排队但视频丢失的标失败。
  读文件失败（OSError）与内容损坏分开处理：读失败时保留原文件、本次不恢复
  也不覆盖；JSON 解析失败或顶层 `tasks` 不是列表才回退到扫描结果目录的
  `report.json` 重建。
- 失败任务可重试：`POST /api/tasks/{id}/retry`。`retry` 在锁内完成
  404/409/200 判定并返回状态快照，避免与 worker 抢跑；源视频已丢失时保持
  失败并给出“源视频已丢失，请重新上传”，不重新入队。任务页失败行显示原因
  与重试按钮，409/网络错误先提示再刷新页面。
- 重试前 worker 清空该任务的结果目录：上一次失败的残缺 trc 若被复用，
  Sports2D 本次没跑起来时 `discover_tracks` 会把它当成本次结果，产出基于
  旧关键点的“成功”报告。
- `stop()`：停止 worker，正在跑的分析不打断但结束后不再落盘，测试与优雅
  关停用，避免旧实例用过期快照覆写 `tasks.json`。

测试：`tests/test_web.py` 新增 `test_task_persistence`（运行中→失败可重试、
排队续跑、原文件名/水平档/打码开关恢复、原子写无残留、绝对路径、残缺 trc
清理、损坏与形状非法自愈、读失败不覆盖、历史目录兜底）与重试端点用例
（404/409/200、源视频丢失）。`mypy skibot` 23 个源文件无问题。

未做：任务表版本迁移（`version` 只写不读）；多进程共享同一结果目录。

## Comments

2026-10-23：独立验证工作流（4 个怀疑者 + 逐条对抗复核）确认 8 处，已修：

- `tasks` 不是列表（null/数字/对象）时启动崩溃 → 形状校验按损坏处理。
- `created_at` 非字符串时整条记录被丢弃 → 捕获 TypeError 退化到当前时间。
- 重试端点锁外读状态与 worker 抢跑 → `retry` 返回锁内状态快照。
- 失败任务读失败（文件被锁）被当成损坏并覆盖 → OSError 单独处理，不覆盖。
- 重试不校验源视频存在 → 保持失败并给可操作提示。
- 重试复用旧工作目录，残缺 trc 被当成本次结果 → 开跑前清空结果目录。
- 测试实例僵尸 worker 覆写表 → 新增 `stop()`，测试在“重启”前停表。
- 文档把存储路径写成 `results/tasks.json` → 改为 `<result-dir>/tasks.json`。

2026-10-23：第二轮复核（针对修复后的代码）确认 5 处，已修：

- 从报告重建的旧任务把源视频猜成结果目录里的 `<id>.mp4`（上传实际在
  `uploads/<id>/`），改选与预览全部打不开 → 按 report 的 `video` →
  `uploads/<id>/` → 猜名顺序找回；打码开关记为未知（blur=None）。
- 状态停在 RUNNING 但报告已写完（上次落盘失败）时被当失败重跑并删报告 →
  先查报告，存在就恢复为完成。
- 清理旧产物用 `rmtree(ignore_errors=True)`，删不干净会再次捡到旧 trc →
  改为严格删除，失败进任务失败表。
- 读不出的 tasks.json 只保护到下一次落盘 → 下次保存前重试读取，仍失败则
  备份为 `tasks.json.unreadable` 再重建。
- 重试 409 文案在锁外读状态 → `retry` 返回 `RetryResult`（含锁内 error）。

2026-10-23：第三轮复核确认 1 处，已修：

- `create_app` 没把相对 `--result-dir` 转绝对路径，默认 `web_results` 下
  落盘的 video_path 是相对路径，换工作目录启动后排队任务被判“视频丢失” →
  入口处 `resolve()`，并补相对路径上传的测试。

2026-10-23：复核综合再确认 1 处，已修：

- `_task_load` 不校验字段类型：`"id": ["t1"]` 会在字典查找时抛
  unhashable 把启动带崩；`"blur": "false"` 会被 bool() 转成 True →
  逐字段校验，坏记录跳过，未知水平档降级，blur 非布尔按未知处理。
