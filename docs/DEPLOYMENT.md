# 部署手册 / Deployment Guide

> farmland-mpc v0.2 · 2026-05

本文档覆盖三条部署路径。选择最适合你的场景：

| 路径 | 适用场景 | 操作系统 | 许可要求 |
|------|----------|----------|----------|
| **A. ArcGIS Pro 工具箱** | 规划员 GUI 操作 | Windows 10/11 | ArcGIS Pro 3.7 + Spatial Analyst |
| **B. 纯 Python CLI** | 科研复现 / 服务器 / CI | Windows / macOS / Linux | 无 |
| **C. Google Colab** | 零安装浏览器演示 | 任意 | Google 账号 |

---

## 路径 A — ArcGIS Pro 工具箱

### A.1 前置条件

| 项目 | 要求 |
|------|------|
| ArcGIS Pro | **3.7**（含 Deep Learning Libraries 安装包） |
| 扩展 | Spatial Analyst + Image Analyst |
| Python 环境 | `arcgispro-py3` 或其 clone（Python 3.13） |
| 磁盘 | 工具箱 ~10 MB + 每区域 prepared_dir ~50-200 MB |
| 内存 | 建议 16 GB+（Tool 2/3 峰值 ~4 GB） |

### A.2 安装步骤

```text
步骤 1 — 复制仓库
    将整个 arcgis-farmland-mpc/ 文件夹复制到目标机器任意位置。
    确保 core/ 和 LandUseOptimization_P9.pyt 同级。

步骤 2 — Clone Python 环境（推荐隔离）
    ArcGIS Pro → Settings → Python → Manage Environments → Clone Default
    记住 clone 路径（例如 D:\Users\<user>\AppData\Local\ESRI\conda\envs\arcgispro-py3-clone3）

步骤 3 — 补装依赖
    关闭 ArcGIS Pro，打开 "Python Command Prompt"（开始菜单 → ArcGIS 文件夹）：

    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install onnx onnxruntime gymnasium

    注意：libpysal / geopandas / scikit-learn 在 ArcGIS Pro 3.7 已内置，无需额外安装。

步骤 4 — 加载工具箱
    ArcGIS Pro → Catalog → Toolboxes → 右键 Add Toolbox
    浏览到 LandUseOptimization_P9.pyt

步骤 5 — 验证
    展开工具箱 → 双击 "5. Check Dependencies"
    所有行显示 [OK] 即部署成功。
```

### A.3 KMP OpenMP 冲突修复

ArcGIS Pro 3.7 内置 `mkl 2024.2.2`，与 PyTorch 捆绑的 `libiomp5md.dll` 冲突。
`.pyt` 文件顶部已包含修复：

```python
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
```

如果你看到 `OMP: Error #15: ... already initialized`，确认此行存在且位于所有 `import torch` 之前。

### A.4 已知限制

- **fiona GBK 崩溃**（仅中文 Windows + 旧 clone 环境）：ArcGIS Pro 3.7 的 `arcgispro-py3-clone3` 已修复此问题（fiona 1.10.1）。如果你使用旧 clone（clone-new2 等），可能遇到 `UnicodeDecodeError`。解决方案：重新 clone 默认环境。
- **Spatial Analyst 必需**：Tool 1 依赖 `arcpy.sa.Slope` + `ZonalStatisticsAsTable`。无此扩展无法运行 Tool 1（Tool 2-4 不需要）。

---

## 路径 B — 纯 Python CLI

### B.1 前置条件

| 项目 | 要求 |
|------|------|
| OS | Windows / macOS / Linux |
| conda | Miniconda 或 Mambaforge（推荐 mamba） |
| Python | 3.11+（environment.yml 锁定 3.11） |
| 磁盘 | env ~3 GB（含 PyTorch CPU） |

### B.2 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/zhouning/arcgis-farmland-mpc.git
cd arcgis-farmland-mpc

# 2. 创建 conda 环境（约 5 分钟）
conda env create -f environment.yml
# 或使用 mamba 加速：
# mamba env create -f environment.yml

# 3. 激活
conda activate farmland-mpc

# 4. 验证
farmland-mpc version
# 输出: farmland-mpc 0.1.0
```

### B.3 手动创建环境（替代方案）

如果 `environment.yml` 解析失败（solver 冲突等），可手动创建：

```bash
conda create -n farmland-mpc --override-channels -c conda-forge -y \
    python=3.11 geopandas rasterio pyogrio fiona shapely numpy scipy \
    networkx libpysal scikit-learn matplotlib tqdm typer pytorch cpuonly \
    onnx onnxruntime gymnasium

conda activate farmland-mpc
pip install -e .
```

### B.4 DLL 链路说明（仅 Windows）

`farmland_mpc/__init__.py` 包含 `_isolate_conda_geostack()` shim：
- 自动注册 `Library/bin` 等 DLL 目录（Python 3.8+ 不再查 PATH）
- 设置 `PROJ_LIB` / `GDAL_DATA` 指向 conda env 内部
- 清除 `GDAL_DRIVER_PATH` 避免加载外部插件

此 shim 在非 conda 环境下为 no-op，无需手动干预。

### B.5 GPU 支持（可选）

默认安装 CPU-only PyTorch。如需 GPU 加速 Tool 3 训练：

```bash
# 替换 pytorch cpuonly 为 GPU 版本
conda install -c conda-forge pytorch pytorch-cuda=12.4 -c nvidia
```

注意：MPC 规划（Tool 4）使用 ONNX Runtime，不依赖 PyTorch GPU。

---

## 路径 C — Google Colab

### C.1 一键打开

点击 badge 直接在浏览器中运行：

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zhouning/arcgis-farmland-mpc/blob/main/notebooks/farmland_mpc_colab_demo.ipynb)

### C.2 Colab 内部流程

Notebook 自动执行：
1. `!pip install -e .` 安装 farmland_mpc 包
2. 生成 4-polygon 合成 DEM fixture
3. 运行 Phase A（prepare）→ Phase B（sample）→ Phase C（train）→ Phase D（plan）
4. 可视化优化前后对比

### C.3 注意事项

- 默认使用 **CPU runtime**（足够，MPC 不需要 GPU）
- 如需加速 Tool 3 训练，切换到 GPU runtime（Runtime → Change runtime type → T4 GPU）
- Colab 免费版有 **24 小时会话上限**；县级规模全链路约 2 小时，不会超时
- 数据不持久化——每次重连需重新运行

---

## 投影 CRS 区域对照表

Tool 1 / Tool 4 需要一个**米制投影 CRS** 做面积和坡度计算。默认 `EPSG:32648`（UTM Zone 48N），仅适用于经度 102°–108°E。

| 区域 | 经度范围 | 推荐 CRS |
|------|----------|-----------|
| 四川、重庆、贵州、云南 | 102°–108°E | `EPSG:32648`（默认） |
| 河南、湖北、湖南 | 108°–114°E | `EPSG:32649` |
| 江苏、浙江、福建、江西 | 114°–120°E | `EPSG:32650` |
| 广东、广西、海南 | 108°–114°E | `EPSG:32649` |
| 山东、河北 | 114°–120°E | `EPSG:32650` |
| 东北（辽宁、吉林、黑龙江）| 120°–132°E | `EPSG:32651` / `32652` |
| 新疆 | 72°–96°E | `EPSG:32643`–`32645` |
| 西藏 | 78°–102°E | `EPSG:32644`–`32647` |
| CGCS2000 3° 带（通用替代）| 全国 | `EPSG:4523`–`EPSG:4554`（按中央经线选） |

**判断方法**：用 DLTB 数据的质心经度 ÷ 6 + 31 = UTM 带号。例如重庆 106.5°E → 106.5/6+31 = 48.75 → Zone 48。

**错误信号**：如果 Tool 1 输出的 `slope_mean` 均值 > 50° 或 < 1°，大概率 CRS 选错。正常丘陵地区均值应在 5–15° 之间。

---

## 部署验证清单

完成部署后，按以下清单逐项确认：

### 路径 A（ArcGIS Pro）

- [ ] Check Dependencies 全部 `[OK]`
- [ ] `import torch; print(torch.__version__)` 在 Python Command Prompt 中正常输出
- [ ] `import gymnasium; import libpysal` 无报错
- [ ] 工具箱在 Catalog 中可见，5 个工具均可展开

### 路径 B（纯 Python CLI）

- [ ] `farmland-mpc version` 输出版本号
- [ ] `python -c "import rasterio; import geopandas; print('OK')"` 无报错
- [ ] `python -m farmland_mpc.tests.smoke_prepare` 输出 `smoke test passed`

### 路径 C（Colab）

- [ ] Notebook 第一个 cell `!pip install -e .` 无报错
- [ ] 第二个 cell 合成数据生成成功
- [ ] 全部 cell 运行完毕无红色报错

---

## 常见部署问题

| 症状 | 原因 | 解决 |
|------|------|------|
| `OMP: Error #15: ... already initialized` | mkl 与 PyTorch libiomp5md 冲突 | 确认 `.pyt` 顶部有 `KMP_DUPLICATE_LIB_OK=TRUE` |
| `ModuleNotFoundError: No module named 'gymnasium'` | 未补装依赖 | `pip install gymnasium` |
| `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xd5` | fiona GBK 崩溃（旧 clone） | 重新 clone 默认环境或使用路径 B |
| `ImportError: DLL load failed while importing _warp` | rasterio DLL 链路断裂 | 使用 ArcGIS Pro 3.7 + Deep Learning Libraries |
| `farmland-mpc: command not found` | 未 `pip install -e .` 或未激活 env | `conda activate farmland-mpc && pip install -e .` |
| Tool 1 报 `Spatial Analyst extension not available` | 许可不含 SA | 购买扩展或改用路径 B（纯 Python prepare 不需要 arcpy） |
| `slope_mean` 均值异常（>50° 或 <1°） | proj_crs 选错 | 参照 CRS 区域对照表修改 |

---

*下一步*：部署验证通过后，参阅 [QUICKSTART.md](QUICKSTART.md) 进行 5 分钟端到端验证，或参阅 [USER_GUIDE.md](USER_GUIDE.md) 了解完整操作流程。
