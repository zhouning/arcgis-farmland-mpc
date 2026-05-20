# 用户手册 / User Guide

> farmland-mpc v0.2 · 2026-05
> 适用对象：已完成 [DEPLOYMENT.md](DEPLOYMENT.md) 部署、需要在真实区域数据上执行规划的用户。

---

## 1. 工具箱概述

`farmland-mpc` 把 **对比学习世界模型 (contrastive world model) + 模型预测控制 (MPC)** 方法封装成两种接口：

- **ArcGIS Pro 工具箱**（GUI）：5 个工具，规划员鼠标点击即可操作
- **Python CLI**（命令行）：`farmland-mpc {prepare,sample,train,plan}` 4 个子命令

两接口共用同一份算法源码（`farmland_mpc/` 包），产出物兼容。

### 1.1 四阶段管道 Pipeline

| 阶段 | ArcGIS UI | CLI 命令 | 是否反复 | 县级耗时 |
|------|-----------|----------|----------|----------|
| A | Tool 1: Prepare Data & Blocks | `farmland-mpc prepare` | 每区域一次 | ~10-15 min |
| B | Tool 2: Sample Transitions | `farmland-mpc sample` | 每区域一次 | ~15-25 min |
| C | Tool 3: Train Contrastive Ensemble | `farmland-mpc train` | 每区域一次 | ~30-60 min |
| D | Tool 4: MPC Planning | `farmland-mpc plan` | **反复规划** | ~7 min / episode |
| — | Tool 5: Check Dependencies | (无对应 CLI) | 按需 | 秒级 |

> **关键概念**：A–C 是**一次性装备**（per-region setup），D 是**反复规划**（你会反复调超参数重跑 Tool 4）。

### 1.2 算法简述

- **训练数据来源**：随机策略采样 (state, action, reward, next_state) 元组 + 同 state 下不同 action 的 pairwise margin ranking 对
- **模型**：3-member ensemble，每 member 是 237K 参数的 transition net，导出为 ONNX
- **规划**：MPC 在 ensemble 上 rollout top-K 候选 H 步，选累计预测回报最高的 action
- **目标**：最小化坡度 + 最大化连通性 + 最大化 baimu（百亩方）面积

---

## 2. 输入数据契约

Phase A（Tool 1 / `prepare`）对输入有**严格格式要求**，源数据通常是第三次全国国土调查（三调）成果。

### 2.1 DLTB 地类图斑（必需）

| 字段名 | 类型 | 说明 | 必需 |
|--------|------|------|------|
| `BSM` | Text / Long | 图斑编号（唯一标识） | ✓ |
| `DLBM` | Text(3) | 3 位地类编码（如 `011`=水田） | ✓ |
| `DLMC` | Text | 地类中文名 | ✓ |
| `QSDWDM` | Text(9+) | 权属单位代码（前 9 位 = 乡镇级） | ✓ |
| `Shape_Length` | Double | 自动 | (auto) |
| `Shape_Area` | Double | 自动 | (auto) |

**地类编码约定**：

| 类别 | DLBM 代码 |
|------|-----------|
| 耕地 | `011`（水田）、`012`（水浇地）、`013`（旱地） |
| 园地 | `021`、`022`、`023`（不参与互换） |
| 林地 | `031`（有林地）、`032`（灌木林地）、`033`（其他林地） |
| 水域 | `04x` |
| 道路 | `1xx` |
| 建设用地 | `2xx` |

**字段名自定义**：如果你的数据字段名不一样，可在 Tool 1 UI 或 CLI 参数中覆盖：

```bash
# CLI
farmland-mpc prepare \
    --dltb data.shp --dem dem.tif --out prepared_dir/ \
    --bsm-field PATCH_ID \
    --dlbm-field LAND_CODE \
    --qsdwdm-field TOWN_CODE
```

### 2.2 DEM 栅格（必需）

- **分辨率** ≥ 30m（Copernicus GLO-30 即可；10m 或 5m 更佳）
- **覆盖范围**：覆盖 DLTB 范围即可（自动裁剪）
- **CRS**：任意（自动重投影）
- **格式**：`.tif`、`.img`、`.vrt`，rasterio 支持的栅格格式

### 2.3 XZQ 行政区（可选）

仅用于给乡镇代码注入中文名 label。如不提供，工具自动用 `QSDWDM` 前 9 位作为代码。

字段（如提供）：
- `XZQDM`（9 位行政区划代码）
- `XZQMC`（中文名）

### 2.4 投影 CRS 选择

参阅 [DEPLOYMENT.md 投影 CRS 区域对照表](DEPLOYMENT.md#投影-crs-区域对照表)。**选错 CRS 是最常见的部署陷阱**。

---

## 3. 输出物清单

### 3.1 Phase A 产出

```text
<prepared_dir>/
├── dem_slope_analysis/output/
│   └── DLTB_with_slope.shp                    ← 带 slope_mean 字段的图斑
├── results_real/blocks/township_<code>/
│   ├── block_compositions.json                 ← block ↔ parcels 映射
│   ├── block_features.json                     ← block 特征向量
│   └── parcel_block_mapping.csv
├── townships.json                              ← {code: 中文名}
└── prepare_data_summary.json                   ← provenance
```

### 3.2 Phase B 产出

```text
<prepared_dir>/tool2/
├── transitions.npz                             ← (s, a, r, s') 元组
└── pairwise.npz                                ← (s, a_better, a_worse) 对
```

### 3.3 Phase C 产出

```text
<prepared_dir>/tool3/
├── ensemble_member0.onnx
├── ensemble_member1.onnx
├── ensemble_member2.onnx
└── train_summary.json
```

### 3.4 Phase D 产出

```text
<out_dir>/
├── mpc_land_use.npy                            ← env 最终 land_use 数组
├── mpc_summary.json                            ← 指标 + 配置
└── mpc_run.log

# 如果指定了 output-shp 参数：
<output_path>/optimized.shp                     ← 4 个新字段：
                                                ←   OPT_DLBM   优化后地类码
                                                ←   OPT_DLMC   优化后地类名
                                                ←   CHG_FLAG   变化标志
                                                ←   ORIG_DLBM  原地类码（回溯用）
```

**CHG_FLAG 取值**：
- `0` — 未变
- `1` — 耕地 → 林地（退耕）
- `2` — 林地 → 耕地（开垦）

---

## 4. 四件套使用指南

### 4.1 Tool 1 / `prepare` — 数据准备

**用途**：把 DLTB + DEM + 可选 XZQ 转为下游可消费的 prepared_dir。

#### ArcGIS UI 参数

| 参数 | 必填 | 默认 | 说明 |
|------|------|------|------|
| DLTB Feature Class | ✓ | — | 三调地类图斑 |
| XZQ Feature Class | — | — | 行政区（仅注入中文 label） |
| DEM Source | ✓ | user-supplied | 当前版本仅支持 user-supplied |
| DEM Raster | ✓ | — | 任意 CRS DEM |
| DLBM field | — | `DLBM` | 地类码字段名 |
| QSDWDM field | — | `QSDWDM` | 权属字段名 |
| Reference Township Layer | — | — | 全国乡镇矢量（注入中文 label） |
| Projected CRS | — | `EPSG:32648` | 见 CRS 对照表 |
| Block min parcels | — | 3 | Paper 3 默认 |
| Block min area (ha) | — | 0.5 | Paper 3 默认 |
| Block max parcels | — | 30 | Paper 3 默认 |
| Output Directory | ✓ | — | prepared_dir |

#### CLI 等价命令

```bash
farmland-mpc prepare \
    --dltb path/to/DLTB.shp \
    --dem  path/to/DEM.tif \
    --out  prepared_dir/ \
    --crs  EPSG:32648
```

> CLI 当前不暴露 block 三个超参数（用 Paper 3 默认值）。如需调整，通过 Python API：
> ```python
> from farmland_mpc.prepare import run
> run(dltb_path=..., dem_path=..., prepared_dir=..., proj_crs=...,
>     block_min_parcels=5, block_min_area_ha=1.0, block_max_parcels=50)
> ```

---

### 4.2 Tool 2 / `sample` — 采样

**用途**：在 Phase A 环境上用随机策略采集训练数据。

#### 参数对照

| ArcGIS UI | CLI | 默认 | 说明 |
|-----------|-----|------|------|
| Prepared Data Directory | `--prepared-dir` | — | Tool 1 输出 |
| Number of transition episodes | `--n-episodes` | 60 | Paper 9 v6 默认 |
| Number of pairwise states | `--n-states` | 1000 | Paper 9 v6 默认 |
| Actions per pairwise state | `--n-actions` | 50 | 上限 = env.n_blocks |
| Random seed | `--seed` | 0 | 复现用 |
| Projected CRS | `--crs` | auto | 与 Tool 1 一致 |

```bash
farmland-mpc sample --prepared-dir prepared_dir/
```

---

### 4.3 Tool 3 / `train` — 训练 ensemble

**用途**：读 Phase B 数据训练 3-member ensemble，导出 ONNX。

#### 参数对照

| ArcGIS UI | CLI | 默认 | 说明 |
|-----------|-----|------|------|
| Prepared Data Directory | `--prepared-dir` | — | 同 Tool 2 |
| Ensemble size | `--n-members` | 3 | Paper 9 v6 默认 |
| Epochs per member | `--epochs` | 30 | |
| Patience | `--patience` | 8 | Early stop（0=关） |
| Contrastive lambda_rank | `--lambda-rank` | 5.0 | **Paper 9 v6 突破点** |
| Ranking margin | `--margin` | 0.1 | hinge margin |
| Batch size | `--batch-size` | 256 | |
| Seed base | `--seed-base` | 0 | Member i 用 seed_base + i×1000 |
| torch_threads | `--torch-threads` | 0 | 0=自动；12 实测最佳 |

```bash
farmland-mpc train --prepared-dir prepared_dir/
```

**lambda_rank = 5.0 是关键**：这是 Paper 9 v6 相比 v5 的核心突破。设为 0.0 退化为纯 MSE world model，性能显著下降。

---

### 4.4 Tool 4 / `plan` — MPC 规划

**用途**：用 Phase C 的 ensemble 做 MPC 规划，输出优化后的 DLTB。

#### 参数对照

| ArcGIS UI | CLI | 默认 | 说明 |
|-----------|-----|------|------|
| Prepared Data Directory | `--prepared-dir` | — | Tool 1 输出 |
| Ensemble Directory | `--ensemble-dir` | — | Tool 3 输出 |
| Output Directory | `--out-dir` | — | 存 summary + log |
| MPC horizon H | `--horizon` / `-H` | 5 | |
| Top-K candidates | `--top-k` / `-K` | 50 | |
| Discount gamma | (UI only) | 0.99 | |
| Continuation policy | `--continuation` | random | random=快; greedy=慢但略好 |
| 1-step scoring | `--scoring` | reward | reward / slope_only |
| n_episodes | `--n-episodes` | 1 | Paper 9 v6 用 5 |
| ONNX threads | `--threads` | 0 | 0=自动 |
| Seed offset | `--seed-offset` | 0 | |
| Projected CRS | `--crs` | auto | |
| Output Optimized DLTB | `--output-shp` | — | 留空跳过 shapefile 输出 |
| Farm DLBM | `--farm-dlbm` | `011` | 互换时代表性耕地编码 |
| Forest DLBM | `--forest-dlbm` | `031` | 互换时代表性林地编码 |
| Reward 权重 (4 项) | (UI only) | 见下 | **见 §6.1 陷阱说明** |

#### Reward 权重默认值

- slope_weight = 4000
- cont_weight = 500
- baimu_weight = 1500
- baimu_bonus = 5

#### 典型命令

```bash
# 单 episode 快速规划，写出优化后 shapefile
farmland-mpc plan \
    --prepared-dir prepared_dir/ \
    --ensemble-dir prepared_dir/tool3/ \
    --out-dir mpc_run1/ \
    --horizon 5 --top-k 50 \
    --output-shp mpc_run1/optimized.shp

# 5-seed 评估（Paper 9 v6 标准配置）
for seed in 0 1 2 3 4; do
  farmland-mpc plan \
      --prepared-dir prepared_dir/ \
      --ensemble-dir prepared_dir/tool3/ \
      --out-dir "mpc_run_seed${seed}/" \
      --horizon 5 --top-k 50 --n-episodes 1 \
      --seed-offset ${seed}
done
```

---

## 5. 标准操作流程

### 5.1 首次部署 — 新区域完整流水线

完整端到端约 1–2 小时（县级 CPU）：

```text
步骤 1 — 准备数据
    收集：DLTB.shp（三调） + DEM.tif（≥30m） + 可选 XZQ.shp

步骤 2 — 选择 CRS
    根据区域经度选择 UTM Zone（见 CRS 对照表）

步骤 3 — Tool 1 / prepare       (~10-15 min)
    Output: prepared_dir/

步骤 4 — Tool 2 / sample        (~15-25 min)
    Output: prepared_dir/tool2/

步骤 5 — Tool 3 / train         (~30-60 min)
    Output: prepared_dir/tool3/

步骤 6 — Tool 5 / Check Dependencies（GUI 用户）
    确认所有依赖在位

步骤 7 — Tool 4 / plan          (~7 min/episode)
    Output: mpc_run1/ + optimized.shp

步骤 8 — 可视化对比
    在 ArcGIS Pro 或 QGIS 中打开 optimized.shp
    按 CHG_FLAG 字段符号化（0=灰、1=红、2=绿）
```

### 5.2 反复规划

跳过 Tool 1–3，直接跑 Tool 4，改超参数对比：

```bash
# 探索不同 horizon
for H in 3 5 8 10; do
  farmland-mpc plan --prepared-dir prepared_dir/ \
      --ensemble-dir prepared_dir/tool3/ \
      --out-dir "run_H${H}/" --horizon ${H}
done
```

---

## 6. 关键陷阱与注意事项

### 6.1 Reward 权重不能在 Tool 4 改

Tool 4 UI 暴露了 4 个 reward 权重（slope_weight 等），但**改它们几乎不会改变 MPC 的决策**：

- `env.step()` 用新权重计算 reward，但**只影响 summary 报告的数字**
- MPC 候选排序用的是 **Tool 3 ensemble 预测的 reward**，ensemble 学的是**训练时**的权重
- 所以改权重后 MPC 候选排序不变 → 行动不变

**要让新权重真正生效**：用新权重**重跑 Tool 2 + Tool 3**。

### 6.2 换区域必须重训 ensemble

ONNX graph 把 `n_blocks` 作为 action embedding 表大小**烘焙进图里**（静态维度）。

| 区域 | n_blocks |
|------|----------|
| 璧山 | 2600 |
| 内江 | ~1500 |

换区域时 ONNX 维度不匹配会直接崩。Tool 4 启动时做 `assert_compatible` 预检，不匹配立即报错。**不能复用其他区域训好的 ensemble**。

### 6.3 CRS 选错的信号

如果 Tool 1 输出的 `slope_mean` 分布异常：
- 均值 > 50° → 大概率 CRS 是地理坐标（度而非米）
- 均值 < 1° → CRS 单位是米但带号错（西部数据用了东部带号）
- 正常丘陵地区：5–15°

### 6.4 跨区域迁移（partial transfer）效果有限

实测：用 Region A 的 ensemble 做 warm-start 训练 Region B，**不优于** Region B from-scratch：
- partial transfer 在 slope 上不胜 from-scratch
- 连通性和 baimu 指标反而退化（std 翻倍）

**结论**：跨区域应直接 from-scratch；partial transfer 仅作 warm-start 加速收敛。

### 6.5 `baimu`（百亩方）是什么

Paper 9 objective 里的一项：面积 ≥ **6.67 公顷**（即 100 亩）的**连通耕地斑块**数量。生产场景里是一个重要的 agronomic metric——规模化经营的下限。

- `baimu_count` — 数量
- `baimu_area` (ha) — 总面积

### 6.6 内存峰值

- Tool 1：~2 GB（DEM 重投影 + 逐图斑 zonal stat）
- Tool 2：~3 GB（n_transition_episodes × n_blocks 状态采样）
- Tool 3：~4 GB 峰值（batch 训练）
- Tool 4：~2 GB（ONNX runtime + env）

16 GB 系统内存足够；8 GB 系统可能在 Tool 3 OOM，建议降 batch_size 至 128。

---

## 7. 故障排查

### 7.1 Tool 1 / prepare

| 错误 | 解决 |
|------|------|
| `Spatial Analyst extension not available` (ArcGIS only) | 启用 SA 扩展，或改用 CLI（纯 Python prepare 不依赖 arcpy） |
| `DLTB missing required field: BSM` | 用 `--bsm-field` 覆盖为实际字段名 |
| `slope_mean` 均值异常 | CRS 选错，见 §6.3 |
| `proj.db version 5 ... expect ≥ 6` warning | 无害，可忽略；或传 WKT 替代 EPSG 编码 |

### 7.2 Tool 2 / sample

| 错误 | 解决 |
|------|------|
| `libpysal: disconnected components / islands` warning | 无害，孤立小地块产生，不影响结果 |
| 第一次 Queen 邻接构建慢（~60 秒/县级） | 正常，会缓存供后续复用 |

### 7.3 Tool 3 / train

| 错误 | 解决 |
|------|------|
| `OOM (out of memory)` | 降 `--batch-size` 到 128 或 64 |
| 训练 loss 不下降 | 检查 `lambda_rank` 是否 5.0（不要设 0） |
| `KMP_DUPLICATE_LIB_OK` warning | 已设环境变量，无害 |

### 7.4 Tool 4 / plan

| 错误 | 解决 |
|------|------|
| `ONNX ensemble was trained with n_blocks=N1, env has n_blocks=N2` | 区域不匹配，见 §6.2 |
| Shapefile 输出报 `output must be an absolute path` | 用绝对路径 `D:/out/optimized.shp` |
| Reward 权重改了没效果 | 见 §6.1（必须重跑 Tool 2+3） |
| 单 episode 跑了 >15 min | 检查 `--threads` 是否锁过低；尝试 `--threads 0`（自动） |

---

## 8. 性能参考

| 区域 | n_blocks | n_parcels | Phase A | Phase B | Phase C | Phase D (1 ep) |
|------|----------|-----------|---------|---------|---------|----------------|
| 合成 4-polygon | 1 | 4 | <1 s | <1 s | ~10 s | ~2 s |
| 县级（典型）| 2000–3000 | 80k–120k | 10–15 min | 15–25 min | 30–60 min | 5–10 min |

**硬件假设**：Intel i7 CPU / 16 GB RAM / 无 GPU。

---

## 9. 算法与方法

### 9.1 对比损失

总 loss = MSE(next_state_pred, next_state_true) + λ · margin_ranking_loss

其中 margin ranking loss 来自 pairwise 对 (a_better, a_worse) 与同 state：

```
L_rank = max(0, m - (R̂(s, a_better) - R̂(s, a_worse)))
```

`λ = 5.0` 是 Paper 9 v6 调出的甜区。

### 9.2 MPC rollout

每步：
1. 取当前 state s_t
2. 列出所有可行 action（候选 blocks 翻转）
3. 用 ensemble 预测每个候选 horizon H 步的累计 discounted return
4. 取 top-K，按 1-step scoring（默认 reward）排序
5. 执行最优 action

continuation policy：
- `random` — H 步内剩余动作随机（快）
- `greedy` — H 步内剩余动作贪心（慢但精度略高）

### 9.3 已发表结果（Paper 9 v6，璧山 5 seeds）

| 指标 | 数值 |
|------|------|
| Slope change | **−1.289% ± 0.079%** |
| ΔContiguity | +0.0160 ± 0.0016 |
| ΔBaimu count | +3.4 ± 1.0 |
| ΔBaimu area | −312 ha (诚实 limitation) |
| 训练成本 | 25 min CPU / seed |

参考：vs in-house MARL slope −0.81% ± 0.09%（1.6× 提升），训练 8–12 GPU-hour / seed。

---

## 10. 引用

如在学术工作中使用本工具箱，请引用底层研究论文（详情待该论文正式发表后更新）：

> Zhou, N. et al. *Model-based AI planning enables county-scale farmland consolidation in fragmented mountain landscapes*. (in preparation, Peking University).

---

*参数细节*：参阅 `LandUseOptimization_P9.pyt` 工具描述或 `farmland-mpc <command> --help`。
*算法细节*：参阅 `farmland_mpc/` 包源码 + Paper 9 主稿。
*问题反馈*：GitHub Issues at https://github.com/zhouning/arcgis-farmland-mpc/issues
