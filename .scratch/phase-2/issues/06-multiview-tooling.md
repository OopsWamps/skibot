# 06 — 多机位标定与三角化工具

Status: ready-for-agent
Blocked by: 01 — 冻结三维指标定义与改写决策记录

## What to build

离线工具：多路视频同步、相机标定、三维关节真值三角化，输出可复用的标定文件与真值轨迹，供验收脚本消费。

## Acceptance criteria

- [ ] 合成相机测试：已知三维点投影到两视图后可还原，误差在数值容差内
- [ ] 支持拍手同步与半帧判据，不满足的段落被标记而不是静默通过
- [ ] 支持 ChArUco/棋盘格标定与联合 bundle adjustment
- [ ] 标定结果含相机内参、位姿与世界原点
- [ ] 三角化输出可被验收脚本直接读取

## Notes

来源规格：`.scratch/phase-2/spec.md`（验证与采集）。
