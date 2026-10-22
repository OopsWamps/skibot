# skibot

滑雪视频体态分析：上传一段滑雪视频，输出关键点轨迹、五项体态指标和结构化报告。

架构基于 [Sports2D](https://github.com/davidpagnon/sports2d)（BSD-3-Clause，JOSS 2024），
在其管线上叠自建模块：主轨迹选择、视角分类（帧级门控）、五项指标、参考区间判断、
报告生成、标注视频渲染、人脸打码。

## 快速开始

```bash
uv venv .venv --python 3.11
uv pip install -e .
.venv/Scripts/python.exe scripts/patch_rtmlib_dml.py   # Windows NVIDIA 卡走 DML GPU 推理
skibot analyze video.mp4 --level intermediate
```

产物在 `results/<视频名>/`：`report.json`（结构化报告）、`*_annotated.mp4`
（骨架叠加标注视频）、`*_Sports2D/`（轨迹 trc 与运动学文件）。

## 命令行

```bash
skibot analyze <video...> [--level novice|intermediate|advanced]
              [--no-blur] [--result-dir DIR] [--color pink] [--person 3] [--refs TOML]
skibot render --video v.mp4 --trc t.trc --out o.mp4 [--view sagittal|frontal]
```

- 默认开启人脸打码（Haar，雪镜/面罩召回有限，见 `privacy.py` 说明）
- 长视频（>10000 帧）建议先降采样到 15fps 再分析（`ffmpeg -vf fps=15`），
  角度曲线 15Hz 对 1~2 秒一个弯的滑行够用
- 跟拍镜头（正侧面来回摆）按帧级视角门控处理：报告判 `mixed`，
  侧面指标取自侧面帧、正面指标取自正面帧
- 报告默认带"参考值未校准"水印——novice/intermediate 区间是占位值，
  待真人教练确认（评审材料见 `docs/`）

## 结构

```text
skibot/
├── cli.py            # 命令行入口，串起整个流程
├── backend.py        # Sports2D 适配：调用（DML GPU/限 8 人/7200s 超时）、产物发现、抢救契约
├── trc_io.py         # trc 格式解析 + Halpe-26 重排
├── skeleton.py       # Halpe-26 布局契约：索引、别名、肢体链、conf 语义
├── select_person.py  # 主轨迹选择（尺寸/位移/颜色打分）
├── view.py           # 视角分类：整视频判定 + 帧级归属（跟拍混合视角）
├── metrics.py        # 五项体态指标计算 + 指标注册表（支持帧级视角门控）
├── assess.py         # 评估流程：关键点 → 报告（主路径与抢救路径共用）
├── reference.py      # 三档水平参考区间加载与查询
├── report.py         # 结构化报告（质量提示：深蹲 caveat、小人物、核心点缺失；带内百分位）
├── render.py         # 自绘标注视频（只画选中者，指标按区间着色）
└── privacy.py        # 人脸打码
reference_values.toml   # 参考区间表（novice/intermediate 为占位值；advanced 为 EPFL 竞技 p25~p75）
docs/DECISIONS.md       # 全部产品与技术决策记录
docs/coach_review_simulated.md   # AI 模拟教练评审（待真人终审）
docs/lit_angle_values.md         # 文献角度数值汇集
docs/GLOSSARY.md        # 领域词汇表
```

## 素材工具链（scripts/）

B站公开视频的采集与分析链路（内部验证用，见 docs/DECISIONS.md 第 12 条）：
按关键词或 UP 主频道采集元数据 → 人工筛 → 下载 → 预览拼图 → 批量分析 →
trc 反查连续滑行段。入口：`fetch_user_footage.py`、`fetch_channel_footage.py`、
`download_user_footage.py`、`preview_user_footage.py`、`verify_user_footage.py`、
`find_continuous_runs.py`、`_run_batch.py`（批量驱动）。

## 验证状态

- 竞技素材（EPFL Ski2DPose，158 clip）：关节误差与五指标误差归因见 docs/DECISIONS.md 第 5 条
- 大众素材（B站，43 条分析、20+ 连续滑行段）：管线验证结论见
  `datasets/recreational_ski/recreational_validation_report.md`

## 一期范围

分析五项指标：屈髋角、膝弯曲角、躯干前倾角（侧面视频）；膝盖内扣角、肩髋对齐（正面视频）。
弯相位划分、单目 3D、评分制、Web UI、knee_valgus 更名均在二期（见 docs/DECISIONS.md）。
