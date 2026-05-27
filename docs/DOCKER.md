# 容器化部署与使用指南

> **目标读者**：拿到 farmland-mpc Docker 镜像、想直接跑县级数据的用户。
> 三种典型场景全覆盖：技术员用 Web 表单 / 研究人员用 Notebook 或 CLI / 部门服务器内网部署。

镜像基于 conda-forge GIS 栈，纯 CPU，约 4 GB。Apple Silicon 与 amd64 都有，按你 docker 本机架构自动选择。

---

## 1. 拉镜像

### 1.1 GitHub Container Registry（境外推荐）

```bash
docker pull ghcr.io/zhouning/arcgis-farmland-mpc:latest
```

### 1.2 阿里云镜像服务（国内推荐，速度快很多）

```bash
docker pull registry.cn-hangzhou.aliyuncs.com/zhouning/farmland-mpc:latest
```

> 实际命名空间以发行说明为准；首次拉取约 4 GB。

### 1.3 离线 `.tar.gz`（涉密 / 内网）

如果你的环境没有外网，向我们索取离线包（约 1.5 GB 压缩后）：

```bash
# 加载到本地 docker
zcat farmland-mpc-0.2.1-linux-amd64.tar.gz | docker load
# 或 arm64 版
zcat farmland-mpc-0.2.1-linux-arm64.tar.gz | docker load

docker images farmland-mpc      # 应能看到本地 image
```

为下文一致，给镜像打个本地别名：

```bash
docker tag <加载出来的镜像 ID> farmland-mpc:latest
```

---

## 2. 三种使用方式

镜像有 4 种入口，由 `docker run` 第一个参数决定：

| 第一个参数 | 行为 | 谁用 |
|---|---|---|
| 不传（默认） | 启 JupyterLab on `:8888`，自动打开本地 notebook | 研究人员 / 复现论文 |
| `webui` | 启表单 Web UI on `:8000` | 县局技术员（无代码） |
| `farmland-mpc <子命令>` | 直接调 CLI 子命令 | 批处理 / 脚本 / 内网服务器 |
| `smoke` | 跑 `smoke_end_to_end` 自检 | 验证镜像本身是否正常 |

### 2.1 Web 表单（最简，给县局技术员）

```bash
mkdir -p ~/farmland-mpc-data && cd ~/farmland-mpc-data

docker run --rm -it \
  -p 8000:8000 \
  -v "$PWD:/work" \
  --cpus=8 --memory=16g \
  farmland-mpc:latest webui
```

打开浏览器：<http://localhost:8000>

操作步骤：

1. 选择 DLTB shapefile 的全套文件（必选 `.shp / .shx / .dbf / .prj`，可选 `.cpg`）
2. 选目标 UTM CRS（默认 EPSG:32648 重庆/川渝；东部选 32650 等，详见下拉框）
3. 调整 MPC 参数（默认即可：horizon=5, top-k=50, n_episodes=5）
4. 点"开始运行" → 看进度条 + 实时日志（约 1.5 hr 整县）
5. 完成后下载 `optimized.zip`，解压用 QGIS 打开 `optimized.shp`

> 数据全程不出本机；DEM 自动从 AWS Copernicus 公开镜像拉，约 50–500 MB。

### 2.2 JupyterLab（研究人员，可调可改）

```bash
mkdir -p ~/farmland-mpc-data && cd ~/farmland-mpc-data
# DLTB 放到 ./in/ 子目录

docker run --rm -it \
  -p 8888:8888 \
  -v "$PWD:/work" \
  -e JUPYTER_TOKEN=your-secret-token \
  --cpus=8 --memory=16g \
  farmland-mpc:latest
```

打开终端打印的 `http://localhost:8888/lab?token=...`，会直接跳到本地 notebook
`farmland_mpc_local.ipynb`：

1. 第一个 cell 改两行变量：`INPUT_DLTB = Path('/work/in/your_dltb.shp')`、`PROJ_CRS = 'EPSG:32648'`（或保持 `None` 自动选）
2. **Run All**（Kernel → Restart and Run All）
3. 跑完后 `optimized.shp` 在 `/work/run_1/mpc_output/`，宿主机就是 `~/farmland-mpc-data/run_1/mpc_output/`

这就是 Colab 那个 `farmland_mpc_colab_full.ipynb` 的本地化版，跑通过 Colab 的人零迁移成本。

### 2.3 CLI（批处理 / 服务器自动化）

整套四件套一行串起来：

```bash
docker run --rm \
  -v "$PWD:/work" \
  --cpus=8 --memory=16g \
  farmland-mpc:latest \
  bash -lc 'set -e
    python /repo/scripts/fetch_dem.py \
      --dltb /work/in/dltb.shp --work-dir /work/run --proj-crs EPSG:32648
    farmland-mpc prepare \
      --dltb /work/in/dltb.shp --dem /work/run/dem.tif \
      --out /work/run/prepared --crs EPSG:32648
    farmland-mpc sample \
      --prepared-dir /work/run/prepared \
      --n-episodes 60 --n-states 1000 --n-actions 50 --seed 0
    farmland-mpc train \
      --prepared-dir /work/run/prepared \
      --epochs 30 --lambda-rank 5.0 --n-members 3
    farmland-mpc plan \
      --prepared-dir /work/run/prepared \
      --ensemble-dir /work/run/prepared/tool3 \
      --out-dir /work/run/mpc_output \
      --output-shp /work/run/mpc_output/optimized.shp \
      --crs EPSG:32648 --horizon 5 --top-k 50 \
      --continuation greedy --n-episodes 5'
```

或拿单独的子命令：

```bash
docker run --rm -v "$PWD:/work" farmland-mpc:latest farmland-mpc --help
docker run --rm -v "$PWD:/work" farmland-mpc:latest farmland-mpc plan --help
```

### 2.4 docker compose（部门服务器长驻）

仓库 `docker/docker-compose.yml` 提供 3 个 service：

```bash
cd docker

docker compose up jupyter      # http://server:8888 长驻
docker compose up webui        # http://server:8000 长驻
docker compose run --rm batch <CMD>   # 一次性 CLI 任务
```

强烈建议在 webui / jupyter 前面挂 nginx 加 basic-auth；本镜像里默认无认证（适合单机本地）。

---

## 3. 数据合规 / 离线场景

三调 DLTB 通常涉密。本镜像设计上：

- **数据只通过 volume 挂载进 `/work`**，不会进入镜像层
- **不发任何遥测**；除拉 DEM 外不出网
- 如果连拉 DEM 都不允许，预先在能上网的机器上跑：
  ```bash
  docker run --rm -v "$PWD:/work" farmland-mpc:latest \
    python /repo/scripts/fetch_dem.py \
    --dltb /work/in/dltb.shp --work-dir /work/run --proj-crs EPSG:32648
  ```
  把生成的 `dem.tif` 拷到目标机器，跳过 fetch_dem 步骤直接 `prepare`

---

## 4. 资源建议

| 数据规模 | 建议 `--cpus` / `--memory` | 端到端耗时 |
|---|---|---|
| smoke (36 parcels) | 任意 | <1 min |
| 一个乡镇（~5k parcels）| 4 / 8 GB | ~10 min |
| 一个区/县（53k parcels）| 8 / 16 GB | ~1.5 hr |
| 大县（>100k parcels）| 16 / 32 GB | ~3 hr |

> 整条管道是单核 CPU 路径（onnxruntime CPU + numpy）；多核帮不上 Tool 2 / 3 / 4，但 Tool 1 的 rasterio 可吃多核。给 8 核足够。

---

## 5. 验证镜像本身

新机器拉到镜像后，先跑自检：

```bash
docker run --rm farmland-mpc:latest smoke
```

看到 `END-TO-END SMOKE PASSED ✓` 即说明 conda env、ONNX 导出、四件套链路都正常。

---

## 6. 常见问题

| 症状 | 原因 | 修复 |
|---|---|---|
| 拉镜像超时 / 慢 | github.com 国内访问被切 | 换阿里云镜像（§1.2）或离线 tar.gz（§1.3） |
| `Cannot connect to S3` 在 fetch_dem 阶段 | 内网阻挡 AWS 访问 | 在能上网的机器跑一次 fetch_dem，把 `dem.tif` 拷过去 |
| WebUI 上传 shapefile 失败 | 只传了 `.shp` 没传 `.shx/.dbf/.prj` | shapefile 是 4–6 个文件一组，全选 |
| MPC 跑很慢 | docker 默认 cpus 限额 | 加 `--cpus=8 --memory=16g` |
| `OMP: Error #15` | OpenMP 重复 init（罕见） | 已通过 `KMP_DUPLICATE_LIB_OK=TRUE` 在镜像内默认设上，按理不该出现 |

更多坑见 `docs/MACOS.md` §8。
