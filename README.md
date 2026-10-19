# skibot

滑雪视频体态分析：上传一段滑雪视频，输出关键点轨迹、五项体态指标和结构化报告。

架构基于 [Sports2D](https://github.com/davidpagnon/sports2d)（BSD-3-Clause，JOSS 2024），
在其管线上叠四个自建模块：视角分类、膝盖内扣角、参考区间判断、报告生成。

## 快速开始

```bash
uv venv .venv --python 3.11
uv pip install -e .
skibot analyze video.mp4 --level intermediate
```

## 结构

```text
skibot/
├── cli.py            # 命令行入口，串起整个流程
├── backend.py        # Sports2D 适配：调用、产物发现、超时抢救
├── trc_io.py         # trc 格式解析 + Halpe-26 重排
├── skeleton.py       # Halpe-26 布局契约：索引、别名、肢体链、conf 语义
├── select_person.py  # 主轨迹选择（尺寸/位移/颜色打分）
├── view.py           # 视角分类（侧面矢状面 / 正面额状面）
├── metrics.py        # 五项体态指标计算 + 指标注册表
├── assess.py         # 评估流程：关键点 → 报告（主路径与抢救路径共用）
├── reference.py      # 三档水平参考区间加载与查询
├── report.py         # 结构化报告生成（JSON）
├── render.py         # 自绘标注视频（只画选中者，指标按区间着色）
└── privacy.py        # 人脸打码
GLOSSARY.md             # 领域词汇表
reference_values.toml   # 参考区间表（数值待专业确认）
docs/DECISIONS.md       # 全部产品与技术决策记录
```

## 一期范围

分析五项指标：屈髋角、膝弯曲角、躯干前倾角（侧面视频）；膝盖内扣角、肩髋对齐（正面视频）。
弯相位划分、单目 3D、评分制、Web UI 均在二期（见 docs/DECISIONS.md）。
