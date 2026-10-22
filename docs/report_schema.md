# 报告 schema 迁移（0.1 → 0.2）

日期：2026-10-23。对应二期任务票 02。
三维相位对象与六项指标的三维列在 issue 12 接入；本文只覆盖二维过渡期。

## 变化

- `schema`：`skibot.report/0.1` → `skibot.report/0.2`。
- `metrics[].metric`：`knee_valgus` → `knee_abduction`。数值不变，仍是二维
  投影口径（髋-膝-踝夹角，180° = 伸直）；三维 JCS 定义见 ADR-0001 §5.5。
- `metrics[]` 新增字段：
  - `basis`：`"2d"` 或 `"3d"`。当前报告全部是 `"2d"`。
  - `definition`：同一指标在不同基准下口径不同时写明，例如
    `knee_abduction` 的 `"2d_projection"`。
  - `legacy`：`true` 表示二维遗留项，没有单一三维替代。
  - `replaced_by`：遗留项被哪些三维指标取代（`shoulder_hip_align` →
    `hip_shoulder_separation` + `thoraco_pelvic_obliquity`）。
  - `note`：口径说明，给用户看的短句。
- `shoulder_hip_align` 保留为遗留项（`legacy: true`），值仍是旧合并口径。
  三维报告不再有这一项，改出横断面 H-SEP 与冠状面 obliquity 两项。
- `phases` 对象尚未出现，等 issue 12。

## 字段映射

| 0.1 | 0.2 | 说明 |
|---|---|---|
| `metrics[].metric = "knee_valgus"` | `"knee_abduction"` | 旧名作别名保留 |
| — | `metrics[].basis = "2d"` | 新字段；旧报告缺失时按 `"2d"` 读 |
| — | `metrics[].definition` | 口径不同时才有 |
| — | `metrics[].legacy` / `replaced_by` | 只有 `shoulder_hip_align` |
| — | `metrics[].note` | 口径说明 |
| 其余字段 | 不变 | `reference`、`median_deg`、`pct_in_range`、`band_percentile`、`n_valid_frames`、`verdict` |

## 兼容

- 读旧报告：把 `knee_valgus` 当 `knee_abduction` 的别名；`basis` 缺失时按
  `"2d"` 解释。旧文件不自动改写。
- 读新报告：`basis` 缺失按 0.1 处理；`legacy` 缺失按 `false` 处理。
- 参考表查询：`reference.lookup("knee_valgus", ...)` 解析到
  `knee_abduction`；旧 TOML 只有 `[knee_valgus]` 时回退到旧键。
- `reference_values.toml` 的带全部是二维口径。三维带按 ADR-0002 在 issue 12
  生成，不写进本文件的二维节。

## 调用点

- 注册表：`skibot/metrics.py` 的 `METRICS`（六项规范指标）、`LEGACY_METRICS`
  （`shoulder_hip_align`）、`TWO_D_METRICS`（二维五键）、`metrics_for_basis`。
- 报告：`skibot/report.py` 按 `TWO_D_METRICS` 与 `metric_plane` 渲染。
- 渲染：`skibot/render.py` 同上；叠加层仍用指标键名（cv2 字体不支持中文）。
- Web 报告页：显示 `basis`、`definition`、`legacy`、`note`。
