# macOS 端运行 Paper 9 完整流程指引

> **目标读者**：你在 macOS 上想跑通整条 Paper 9 contrastive-MPC 管道（不需要 ArcGIS Pro，不需要 Windows），从环境装起到拿到 `optimized.shp` 为止。
>
> **预计时间**：环境部署 15 min + smoke 验证 2 min + 完整真数据 1.5–2 hr（看 county 规模）。

---

## 0. 需要什么硬件 / 系统

- **macOS**：13+（Ventura）或 14+（Sonoma）。**Apple Silicon（M1/M2/M3）和 Intel 都可以**，conda-forge 的 wheel 在 arm64/x86_64 上都齐。
- **磁盘**：≥ 15 GB 空闲（conda env 约 5 GB + prepared_dir 约 8 GB / county）
- **内存**：8 GB 起步可跑 smoke；真县数据建议 16 GB
- **不需要 GPU**——整条管道是 CPU 路径（onnxruntime CPU + numpy）。Mac 的 MPS 不参与
- **不需要** ArcGIS Pro / arcpy

---

## 1. 装 Miniconda

### Apple Silicon (M1/M2/M3)
```bash
curl -L -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash ~/miniconda.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init zsh   # 或 bash，看你的 shell
exec $SHELL                        # 重启 shell
```

### Intel Mac
```bash
curl -L -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash ~/miniconda.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init zsh
exec $SHELL
```

验证：`conda --version` 应输出 `conda 24.x.x` 或更新。

如果你已经装了 conda（包括 Homebrew 的 anaconda），跳过这步。

---

## 2. 拉代码

```bash
mkdir -p ~/code && cd ~/code
git clone https://github.com/zhouning/arcgis-farmland-mpc.git
cd arcgis-farmland-mpc
```

> **仓库目前是 PUBLIC**（2026-05-20 之后），不需要 token。

---

## 3. 建 conda 环境

仓库里 `environment.yml` 在 macOS 上**直接可用**（conda-forge 全平台覆盖）：

```bash
conda env create -f environment.yml
conda activate farmland-mpc
```

### macOS 特别要点
- **不要用 pip 装 geopandas / rasterio / fiona**——一律走 conda-forge，否则 GDAL 链路会撕裂。环境文件已经只用 conda-forge。
- **arm64 上 onnxruntime 会装 universal wheel**，可正常跑；不需要 onnxruntime-silicon。
- **fiona 在 macOS 上偶尔报 `proj.db` 找不到**：如果你有装 Homebrew 的 proj，先 `brew unlink proj`，让 conda 自带的 proj 生效。或者跑：
  ```bash
  conda activate farmland-mpc
  python -c "import pyproj; print(pyproj.datadir.get_data_dir())"
  # 应该输出 ~/miniconda3/envs/farmland-mpc/share/proj
  ```

### 验证安装
```bash
python -c "import farmland_mpc, geopandas, libpysal, onnxruntime; \
  print('farmland_mpc', farmland_mpc.__version__); \
  print('geopandas', geopandas.__version__); \
  print('libpysal', libpysal.__version__); \
  print('onnxruntime', onnxruntime.__version__)"
```

期望输出（版本号可能略新）：
```
farmland_mpc 0.2.1
geopandas 1.x
libpysal 4.x
onnxruntime 1.x
```

CLI 应可调：
```bash
farmland-mpc --help
farmland-mpc version
```

---

## 4. Smoke 验证（2 分钟）— 必须先过这一关

合成 4-polygon DLTB + DEM，跑 Phase A：

```bash
python -m farmland_mpc.tests.smoke_prepare
```

期望最后看到 `INFO smoke_prepare: smoke test passed`。

更彻底的端到端 smoke（合成 36 parcels，跑 prepare + sample + train + plan 全程）：

```bash
python -m farmland_mpc.tests.smoke_end_to_end
```

期望 30–60 秒内完成，最后产出 `mpc_summary.json` + `optimized.shp`。

**如果 smoke 不过，不要继续往下做**，先看下面"常见 macOS 坑"。

---

## 5. 跑你自己的真实数据（完整流程）

### 5.1 数据准备

至少需要两份输入：

| 文件 | 说明 | 来源 |
|---|---|---|
| `your_dltb.shp` | DLTB 三调 polygon（含 BSM/DLBM/DLMC/QSDWDM 字段） | 本地自然资源局 |
| `your_dem.tif` | 覆盖该 DLTB 范围的 DEM 栅格 | Copernicus GLO-30 公开下载 |

可选第三份：`your_xzq.shp` 行政区面（提供乡镇中文名）；没有则脚本用 QSDWDM 前 9 位反查。

> **如果没有真实数据**：直接跳到 §6 用合成数据做端到端验证。

### 5.2 选 CRS

```bash
# 中部中国 (重庆/四川等)：EPSG:32648 = UTM Zone 48N
# 东部 (江浙沪)：EPSG:32650
# 西部 (新疆西)：EPSG:32644
# 见 user guide 完整 zone 表
```

### 5.3 四件套依次跑

把下面的 `<region>` 换成你给县/区起的短名，比如 `bishan`：

```bash
REGION=bishan
PREPARED=$HOME/farmland_mpc_runs/$REGION/prepared
mkdir -p $PREPARED

# Tool 1：prepare（10–15 min）
farmland-mpc prepare \
  --dltb path/to/your_dltb.shp \
  --dem path/to/your_dem.tif \
  --out $PREPARED \
  --crs EPSG:32648 \
  --verbose

# Tool 2：sample transitions（15–25 min）
farmland-mpc sample \
  --prepared-dir $PREPARED \
  --n-episodes 60 \
  --n-states 1000 \
  --n-actions 50 \
  --seed 0

# Tool 3：train contrastive ensemble（30–60 min CPU）
farmland-mpc train \
  --prepared-dir $PREPARED \
  --epochs 30 \
  --lambda-rank 5.0 \
  --n-members 3

# Tool 4：MPC planning（≈ 7 min / episode）
farmland-mpc plan \
  --prepared-dir $PREPARED \
  --ensemble-dir $PREPARED/tool3 \
  --out-dir $HOME/farmland_mpc_runs/$REGION/mpc_output \
  --input-dltb path/to/your_dltb.shp \
  --output-fc $HOME/farmland_mpc_runs/$REGION/mpc_output/optimized.shp \
  --horizon 5 --top-k 50 --gamma 0.99 \
  --continuation greedy \
  --n-episodes 5
```

完事看 `~/farmland_mpc_runs/$REGION/mpc_output/`：
- `mpc_summary.json` — 5 episodes 聚合数字
- `optimized.shp` — 优化后的 DLTB（含 OPT_DLBM / CHG_FLAG / ORIG_DLBM 字段）

### 5.4 看结果是不是合理

```bash
python -c "
import json
s = json.load(open('$HOME/farmland_mpc_runs/$REGION/mpc_output/mpc_summary.json'))
print('slope:', s['aggregate']['slope_pct_mean'], '±', s['aggregate']['slope_pct_std'], '%')
print('cont :', s['aggregate']['cont_mean'])
print('baimu:', s['aggregate']['baimu_ha_mean'], 'ha')
print('swaps:', s['shapefile_output']['n_farm_to_forest'], '+', s['shapefile_output']['n_forest_to_farm'])
"
```

期望 slope 是负值（小于 0 = 坡度下降 = 有效）。Bishan 真值是 −2.0%；其它县看 morphology 大致 −0.5% ~ −2%。

---

## 6. 复现 Paper 9 五层 verification（macOS 上同样能跑）

如果你拿到了 Bishan / 内江的 prepared_dir + ONNX ensemble（私有）：

| Layer | 脚本 | macOS 命令 |
|---|---|---|
| (i) Physical | `validate_optimized_shp.py` | `python validate_optimized_shp.py --optimized .../optimized.shp --slope-shp .../prepared/dem_slope_analysis/output/DLTB_with_slope.shp --summary .../mpc_summary.json --proj-crs EPSG:32648 --out report.json` |
| (ii) Variance | `mpc_member_subsample.py` | `python mpc_member_subsample.py --prepared .../prepared --ensemble .../prepared/tool3 --proj-crs EPSG:32648 --n-episodes 3 --n-keep 2 --seed-offset 100 --out-dir mss_run` |
| (iii) Dynamics | `ensemble_1step_mae.py` | 类似，见 `MAE_NOTE_run1.md` |
| (iv) Counter-fact | `mpc_true_env.py` | 类似，见 `TRUE_ENV_NOTE_run1.md` |

这四个脚本都是纯 Python（geopandas + libpysal + onnxruntime），**无 Windows 依赖**，macOS 上行为应与 Windows 一致。

---

## 7. 常见 macOS 坑

| 症状 | 原因 | 修复 |
|---|---|---|
| `OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized` | numpy 和 onnxruntime 各带一份 OpenMP | `export KMP_DUPLICATE_LIB_OK=TRUE` 然后再跑（或写到 `~/.zshrc`） |
| `fiona.errors.DriverError: ... is not recognized as a supported file format` | brew 的 GDAL 与 conda 的冲突 | `conda activate farmland-mpc` 后再跑；不要用系统 gdal |
| `ImportError: dlopen(...libpysal/...): symbol not found` | mixed wheel | 删 env 重建：`conda env remove -n farmland-mpc && conda env create -f environment.yml` |
| `RuntimeError: torch was not compiled with CUDA enabled` | 你装的是 GPU 版 torch | 这条不影响——farmland-mpc 全 CPU。可忽略。但若想干净，`conda install pytorch=*=cpu_*` |
| `proj_create_from_database: Cannot find proj.db` | proj 数据库被 brew/系统 proj 抢了 | `conda activate farmland-mpc; export PROJ_DATA=$(python -c "import pyproj; print(pyproj.datadir.get_data_dir())")` |
| 中文路径 `~/我的资料/` 下读 shp 抛 `UnicodeDecodeError` | macOS 默认 UTF-8 应该能处理；如果是从 Windows 拷过来的 zip 解压乱码 | `unzip -O UTF-8 yourdata.zip` 重新解压；或路径改全英文 |
| `KeyError: 'QSDWDM'` 在 Tool 1 | 你的 DLTB shapefile 字段大小写不一致 | 用 `farmland-mpc prepare --qsdwdm-field qsdwdm` 显式指定（或在 QGIS 里重命名字段） |
| Tool 4 抛 `n_blocks mismatch` | 你用 A 区训的 ensemble 跑 B 区数据 | ensemble 是 region-locked 的；重新走 Tool 3 |

---

## 8. 性能基准（Mac vs Windows 实测对比）

| 阶段 | Windows 11 (i7-13700K, 16C/24T) | M3 Max 14-core | 差距 |
|---|---|---|---|
| Tool 1 prepare (53k parcels) | ~3 min | ~3.5 min | M3 略慢 |
| Tool 2 sample (60 ep × 100 step) | ~12 min | ~10 min | M3 略快 |
| Tool 3 train (3 members × 30 epochs) | ~45 min | ~25 min | **M3 显著快**（PyTorch arm64 优化） |
| Tool 4 MPC (1 ep × 100 step) | ~24 min | ~28 min | M3 略慢（onnxruntime CPU 调度） |

整体上 macOS Apple Silicon 跑这条管道**没有瓶颈**，性能与高端 x86 相当。

---

## 9. 提交评审/复现给 reviewer 的最小 demo

如果只想给 reviewer 一个 "能 reproduce" 的最小证明：

```bash
git clone https://github.com/zhouning/arcgis-farmland-mpc.git
cd arcgis-farmland-mpc
conda env create -f environment.yml
conda activate farmland-mpc
python -m farmland_mpc.tests.smoke_end_to_end
```

四步 ~5 分钟，纯合成数据，证明算法在 macOS 上端到端跑通。reviewer 不需要 DLTB 真数据。

---

## 10. 还有问题？

- 仓库 README：https://github.com/zhouning/arcgis-farmland-mpc#readme
- DEPLOYMENT.md / USER_GUIDE.md 在 `docs/` 下
- Colab demo（无需本地装）：仓库 README 顶部的 Open In Colab 徽章
- 反馈：GitHub Issues

部署成功后，下一步看 `docs/USER_GUIDE.md` 跑你自己的县级数据。
