# skibot 领域词汇表

给 docs/DECISIONS.md 里的名词一个确定的叫法；模块名跟词条走，评审与对话用同一套词。

- **视角分类（view）**：判定输入视频是侧面（矢状面）还是正面（额状面），判不了记 ambiguous；两侧都成规模记 mixed，两侧指标都出（验证进展 #10）。纯 2D 角度只在相机大致垂直于动作平面时可信（一期 #3）。
- **帧级视角归属（frame_view）**：view.classify_view_frames 的逐帧归属数组：'f'=正面、'l'=侧面、''=无信号帧。votes==0（信号持平）归 'l'，与 ''（信号不存在）是两回事；'' 帧不贡献任何指标（验证进展 #10）。
- **矢状面 / 额状面（sagittal / frontal）**：侧面视频套矢状面指标规则，正面视频套额状面指标规则；视角不符的指标不进报告。
- **骨架布局（skeleton）**：Halpe-26 关键点索引契约，含 trc marker 别名、肢体链、conf 语义。全项目以 (frames, 26, 3) 的 [x, y, conf] 数组说话。
- **五项指标（一期，二维）**：侧面——屈髋角 hip_flexion、膝弯曲角 knee_flexion、躯干前倾角 trunk_lean；正面——膝盖内扣角 knee_valgus、肩髋对齐 shoulder_hip_align（一期 #7）。注册表在 metrics.METRICS；二期迁到三维后保留为回归对照。
- **六项指标（二期，三维）**：hip_flexion、knee_flexion、trunk_lean、knee_abduction、hip_shoulder_separation、thoraco_pelvic_obliquity（ADR-0001）。knee_valgus 的中性名是 knee_abduction，正方向 = 膝向中线（内扣方向）；肩髋对齐拆成横断面 H-Ssep 与冠状面 obliquity，不再合成一个数。
- **旧名兼容（legacy metrics）**：`knee_valgus` 是 `knee_abduction` 的别名；`shoulder_hip_align` 是二维遗留合并键，不在六项规范指标里，三维拆成 `hip_shoulder_separation` 与 `thoraco_pelvic_obliquity`。报告字段迁移见 `docs/report_schema.md`。
- **参考系（frame）**：体固（两个身体节段/线段之间）、重力（躯干轴与重力轴之间）、坡面（躯干轴与坡面法线之间）。六项指标只用这三种；相机/世界坐标不进指标（ADR-0001 §4）。
- **三维骨架契约（skeleton3d）**：Halpe-26、root-relative、重力对齐、逐关节置信度；坐标 +x 解剖右、+y 重力反方向、+z 解剖前（ADR-0001 §3，issue 03）。`Skeleton3D` 构造时强制逐帧减去髋中点；缺测关节 NaN + conf=0；全零帧按缺帧处理（`valid_mask=False`）。
- **三维提升适配器（lifter）**：二维 Halpe-26 关键点 → `Skeleton3D` 的纯适配器。输入 `(T,26,3)` 的 `[x,y,conf]`，输出 root-relative、重力对齐的 26 关节与逐关节置信度。接口在 `skibot/skeleton3d.py`；`MockLifter` 只用于契约测试，MotionBERT 适配器在 issue 05 接入。
- **相位（phase）**：一次完整侧倾振荡切成的入弯/控制/出弯。相位聚合只覆盖屈髋、膝弯、躯干前倾矢状分量；正面三项维持全程口径（spec「相位划分」）。
- **三维参考带（band3d）**：指标 × 档位 × 相位；advanced 由验证过的三维管线跑 EPFL 生成，带宽不得窄于验收 p90 误差（ADR-0002）。
- **重力来源（gravity_source）**：calibrated / adapter / image_vertical。报告必须标注；不可信时躯干前倾不出数（ADR-0001 §5.1）。
- **参考区间（reference）**：新手/中级/高级三档角度区间，存 reference_values.toml；未校准前报告强制带水印（一期 #8）。
- **评估流程（assess）**：关键点 → 视角分类 → 五指标 → 参考区间 → 评估结果。主路径与抢救路径共用同一条流程。二期在关键点之后插入三维提升与相位切分，仍是一次评估出齐报告、相位数据与动画数据（spec「Testing Decisions」）。
- **评估结果（Assessment）**：assess 的唯一出口，报告 dict + 角度序列 + 帧级视角归属三件套；报告与标注视频叠加层必须出自同一次评估。
- **质量提示（quality）**：人物偏小、低置信度帧、深蹲相位等报告水印。统计在 assess（核心八点口径见 skeleton.CORE_JOINTS），判读阈值与文案在 report（验证进展 #3）。
- **主轨迹选择（select_person）**：从多人物轨迹里锁定分析对象。尺寸+位移自动打分；--color 按裤腿颜色锁定；--person 人工直指定（一期 #6）。
- **姿态后端（backend）**：Sports2D 管线（YOLOX 检测 + RTMPose top-down，RTMLib 推理）。skibot 经子进程调用，产物是 px trc（一期 #5）。
- **抢救路径（salvage）**：Sports2D 收尾挂死/超时后，从已落盘的 px trc 直接出报告（scripts/report_from_trc.py）。验证阶段的量角对账也走这条路（一期 #10）。
- **EPFL 对账（reconcile）**：与 EPFL Ski2DPose 人工标注的量化对账工具链（图像级/视频级误差、尾部归因、参考值提案，验证进展 #1-5）。数据集 adapter 在 skibot/epfl.py。
- **标注视频（render）**：自绘骨架+角度值叠加，只画选中者，替代 Sports2D 的全人物标注视频；数值按参考区间绿/红着色（一期 #9）。
- **打码（privacy）**：人脸高斯模糊，默认开（一期 #11）。
