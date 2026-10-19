# skibot 领域词汇表

给 docs/DECISIONS.md 里的名词一个确定的叫法；模块名跟词条走，评审与对话用同一套词。

- **视角分类（view）**：判定输入视频是侧面（矢状面）还是正面（额状面），判不了记 ambiguous。纯 2D 角度只在相机大致垂直于动作平面时可信（DECISIONS 3）。
- **矢状面 / 额状面（sagittal / frontal）**：侧面视频套矢状面指标规则，正面视频套额状面指标规则；视角不符的指标不进报告。
- **骨架布局（skeleton）**：Halpe-26 关键点索引契约，含 trc marker 别名、肢体链、conf 语义。全项目以 (frames, 26, 3) 的 [x, y, conf] 数组说话。
- **五项指标**：侧面——屈髋角 hip_flexion、膝弯曲角 knee_flexion、躯干前倾角 trunk_lean；正面——膝盖内扣角 knee_valgus、肩髋对齐 shoulder_hip_align（DECISIONS 7）。注册表在 metrics.METRICS。
- **参考区间（reference）**：新手/中级/高级三档角度区间，存 reference_values.toml；未校准前报告强制带水印（DECISIONS 8）。
- **评估流程（assess）**：关键点 → 视角分类 → 五指标 → 参考区间 → 报告 dict。主路径与抢救路径共用同一条流程。
- **主轨迹选择（select_person）**：从多人物轨迹里锁定分析对象。尺寸+位移自动打分；--color 按裤腿颜色锁定；--person 人工直指定（DECISIONS 6）。
- **姿态后端（backend）**：Sports2D 管线（YOLOX 检测 + RTMPose top-down，RTMLib 推理）。skibot 经子进程调用，产物是 px trc（DECISIONS 5）。
- **抢救路径（salvage）**：Sports2D 收尾挂死/超时后，从已落盘的 px trc 直接出报告（scripts/report_from_trc.py）。验证阶段的量角对账也走这条路（DECISIONS 10）。
- **标注视频（render）**：自绘骨架+角度值叠加，只画选中者，替代 Sports2D 的全人物标注视频；数值按参考区间绿/红着色（DECISIONS 9）。
- **打码（privacy）**：人脸高斯模糊，默认开（DECISIONS 11）。
