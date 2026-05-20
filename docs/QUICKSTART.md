# 5 分钟快速验证 / Quickstart Verification

> 目的：部署完成后，用最短路径确认整条管道能跑通。

本文档提供三条 5 分钟验证路径，对应三种部署方式。**完成任一路径即可证明部署成功**。

---

## 🟢 路径 B 验证：纯 Python smoke test（最快，2 分钟）

### 前置

完成 [DEPLOYMENT.md 路径 B](DEPLOYMENT.md#路径-b--纯-python-cli)，已 `conda activate farmland-mpc`。

### 执行

```bash
cd <path-to>/arcgis-farmland-mpc
python -m farmland_mpc.tests.smoke_prepare
```

### 预期输出

```text
INFO smoke_prepare: building synthetic DEM at /tmp/.../dem.tif
INFO smoke_prepare: writing 4-polygon DLTB at /tmp/.../dltb.shp
INFO farmland_mpc.prepare: reading DLTB ...
INFO farmland_mpc.prepare: computing Horn 3x3 slope ...
INFO farmland_mpc.prepare: zonal mean per polygon ...
INFO smoke_prepare: slope_mean range: [3.040, 4.116] degrees
INFO smoke_prepare: per-quadrant: {'P001': 3.089, 'P002': 4.116, 'P003': 3.054, 'P004': 3.04}
INFO smoke_prepare: smoke test passed
```

### 验证点

- ✅ 4 个合成多边形 slope 都在合理范围 (0°–90°)
- ✅ `P002`（右上象限有 DEM 凸起）slope 最高
- ✅ `prepare_data_summary.json` 写入成功
- ✅ 重跑覆盖前次输出，无残留

### 失败排查

| 错误 | 原因 | 解决 |
|------|------|------|
| `ImportError: DLL load failed ... _warp` | rasterio DLL 链路问题 | 重新创建 env，确认走的是 `farmland-mpc-pure` 而非 ArcGIS clone |
| `UnicodeDecodeError ... 0xd5` | fiona GBK 崩溃 | 你在 ArcGIS clone 里跑了——退到独立 conda env |
| `ModuleNotFoundError: farmland_mpc` | 未安装 | `pip install -e .` |

---

## 🟡 路径 A 验证：ArcGIS Pro UI（5 分钟）

### 前置

完成 [DEPLOYMENT.md 路径 A](DEPLOYMENT.md#路径-a--arcgis-pro-工具箱)，工具箱已加载。

### 执行

#### 第 1 步：运行 Check Dependencies

```text
1. ArcGIS Pro → Catalog → Toolboxes → LandUseOptimization_P9.pyt
2. 双击 "5. Check Dependencies"
3. 点击 Run
```

### 预期输出（Geoprocessing Messages 面板）

```text
[OK] Python version: 3.13.x
[OK] arcpy: <version>
[OK] Spatial Analyst extension: Available
[OK] torch: 2.12.0+cpu
[OK] onnx: 1.21.0
[OK] onnxruntime: 1.26.0
[OK] gymnasium: 1.3.0
[OK] geopandas: 1.1.3
[OK] libpysal: 4.14.1
[OK] rasterio: 1.4.4
[OK] core/ modules: 11 found
==========================================
All checks passed. Toolbox is ready.
```

### 验证点

- ✅ 11 条 `[OK]` 全部通过
- ✅ 没有 `[MISSING]` 或 `[ERROR]` 行

#### 第 2 步（可选，更彻底）：运行 Tool 4 with synthetic test data

如果你想验证 ONNX runtime 真的能加载 ensemble，但**没有现成的 prepared_dir**：参阅 [USER_GUIDE.md §6](USER_GUIDE.md) 自带数据走完 Tool 1 → 4。最小测试数据集见 §"测试数据下载" 章节。

### 失败排查

| 错误行 | 原因 | 解决 |
|--------|------|------|
| `[MISSING] torch` | 未补装 PyTorch | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| `[MISSING] gymnasium` | 未补装 | `pip install gymnasium` |
| `[ERROR] Spatial Analyst: Not available` | 许可问题 | ArcGIS Pro → Settings → Licensing 启用 SA |
| `[ERROR] torch: OMP Error #15` | KMP 冲突 | 确认 `.pyt` 顶部有 `KMP_DUPLICATE_LIB_OK=TRUE`，重启 ArcGIS Pro |

---

## 🔵 路径 C 验证：Colab notebook（3 分钟，无需本地安装）

### 执行

1. 点击 badge: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zhouning/arcgis-farmland-mpc/blob/main/notebooks/farmland_mpc_colab_demo.ipynb)
2. 登录 Google 账号
3. Runtime → Run all（或逐个 cell 运行）

### 预期输出

依次完成 4 个阶段：
```text
[Cell 1] Cloning repo + pip install -e .  → 完成
[Cell 2] Generating synthetic fixture     → 4 polygons + DEM created
[Cell 3] Phase A: prepare                  → slope_mean populated
[Cell 4] Phase B: sample                   → transitions.npz, pairwise.npz
[Cell 5] Phase C: train (epochs=3, fast)   → 3 ONNX members exported
[Cell 6] Phase D: plan (1 episode)        → mpc_summary.json
[Cell 7] Visualization                     → before/after map
```

### 验证点

- ✅ 所有 cell 无红色报错
- ✅ 最终输出地图显示 land-use 变化（部分耕地→林地或反之）
- ✅ `mpc_summary.json` 中 `slope_change` 字段为负值（坡度下降）

### 失败排查

| 错误 | 原因 | 解决 |
|------|------|------|
| `Quota exceeded` | Colab GPU 配额耗尽 | 切回 CPU runtime；本流程不需要 GPU |
| Cell 6 超时 | 默认配置太重 | 在 Phase D 参数 cell 改 `n_episodes=1, max_steps=10` |

---

## 测试数据获取

仓库未附带真实测试数据（避免敏感数据公开）。你可以：

### 选项 1：使用 smoke_prepare 内置合成数据
路径 B 验证的 4-polygon fixture 已经覆盖了核心管道。

### 选项 2：使用公开数据复现 Paper 9 实测
- DLTB 来源：第三次全国国土调查（向当地自然资源局申请）
- DEM 来源：[Copernicus GLO-30](https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model)（公开，30m）
- 行政区：[全国乡镇矢量](https://github.com/GaryBikini/ChinaAdminDivisonSHP)（社区维护）

### 选项 3：构造你自己的最小区域

任意 50–100 个相邻 DLTB 图斑 + 一块覆盖它们的 DEM（≥30m）即可跑通 Tool 1→4，单次约 5–10 分钟。

---

## 验证完成后

如三条路径中任一通过，部署即视为成功。下一步：

1. 准备你的真实区域数据，参阅 [USER_GUIDE.md §4 数据契约](USER_GUIDE.md#4-输入数据契约)
2. 按 [USER_GUIDE.md §5 标准流程](USER_GUIDE.md#5-四件套使用指南) 跑 Tool 1 → 2 → 3 → 4
3. 县级规模首次端到端约需 1–2 小时（CPU）

---

*故障排查未解决*？查看 [DEPLOYMENT.md 常见问题](DEPLOYMENT.md#常见部署问题) 或在 GitHub Issues 反馈。
