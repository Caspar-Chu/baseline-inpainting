# Baseline Suite

遥感图像遮挡去除论文的 **baseline 增补实验框架**。根据审稿意见，在原文 Section 4.3–4.5 的对比实验中，补充 LaMa、MAT、RePaint、DDRM 等近年方法的恢复结果、性能指标与运行时间。

## 背景

原文任务：对被遮挡/破坏的遥感图像进行补全。遮挡方式包括：

- **Random mask**：按缺失率随机置零像素
- **Block mask**：块状连续遮挡（模拟云覆盖等）

输入为遮挡后的图像，输出为恢复后的完整图像。需在原文标黄的表格和图中，为每组实验新增 4 列 baseline 结果。

## 快速开始

```bash
# 1. 克隆本仓库
git clone <your-repo-url> baseline && cd baseline

# 2. 安装依赖（Python >= 3.12，推荐本机 Mac 使用 uv）
uv sync

# 3. 查看可用实验与 baseline
uv run baseline-suite list

# 4. 生成 mask 缓存并做流水线测试
uv run baseline-suite make-masks missing_patterns
uv run baseline-suite run missing_patterns -b identity

# 5. 接入 LaMa 后单图冒烟（权重首次运行自动下载）
uv run python scripts/smoke_lama.py --experiment missing_patterns --case random
```

> **GPU 说明**：LaMa / MAT 可在 CPU 上冒烟；RePaint / DDRM 推理需要 **CUDA GPU**（推荐 24GB 显存，如 RTX 4090）。服务器部署见 [docs/server_rental_template.md](docs/server_rental_template.md)。

## Baseline 一览

| 名称 | 状态 | 预训练域 | 尺寸处理 | 外部仓库 |
|------|------|----------|----------|----------|
| `identity` | 内置 | — | 无推理，用于流水线测试 | — |
| `lama` | 已接入 | Places (`big-lama`) | 任意分辨率 | `external/lama`（可选，wrapper 使用 `simple-lama-inpainting`） |
| `mat` | 已接入 | Places-512 | pad 至 512 倍数后 crop 回原尺寸 | `external/mat` |
| `repaint` | 已接入 | Places2-256 | resize 至 256² 后还原 | `external/repaint` |
| `ddrm` | 已接入 | ImageNet-256 | resize 至 256² 后还原 | `external/ddrm` |

统一接口见 [扩展 Baseline](#扩展-baseline)。Mask 语义与各方法的转换关系见 [docs/baseline_integration_checklist.md](docs/baseline_integration_checklist.md)。

## 项目结构

```
baseline/
├── baseline增补计划/          # 需求文档、原文、数据素材
├── configs/
│   ├── experiments.yaml       # 4 组实验配置
│   ├── repaint_places256.yml  # RePaint 推理默认参数
│   ├── repaint_places256_fast.yml
│   └── ddrm_imagenet256.yml   # DDRM 推理默认参数
├── docs/
│   ├── baseline_integration_checklist.md  # 各 baseline 对接细节
│   ├── git_sync.md                      # 本机 ↔ 服务器同步
│   └── server_rental_template.md        # GPU 服务器租借与部署
├── scripts/
│   ├── smoke_lama.py          # 单图冒烟测试
│   ├── smoke_mat.py
│   ├── smoke_repaint.py
│   ├── smoke_ddrm.py
│   └── generate_paper_tables.py  # 汇总论文表格
├── src/baseline_suite/        # 核心 Python 包
│   ├── masks.py               # mask 生成
│   ├── io.py                  # 图像读写
│   ├── metrics.py             # PSNR / SSIM / LPIPS / 计时
│   ├── experiments.py         # 实验样本构建
│   ├── runner.py              # 批量运行
│   ├── registry.py            # baseline 注册
│   ├── cli.py                 # baseline-suite 命令行
│   └── baselines/             # 各方法 wrapper
├── data/masks/                # 生成的 mask 缓存
├── results/                   # 恢复图与 metrics.csv
└── external/                  # 第三方 baseline 仓库（不纳入 git）
```

## 环境安装

本项目使用 [uv](https://github.com/astral-sh/uv) 管理依赖，要求 **Python >= 3.12**。

```bash
cd baseline
uv sync
```

主要依赖：`torch`、`opencv-python-headless`、`scikit-image`、`lpips`、`simple-lama-inpainting`、`pandas`、`pyyaml`。

### 服务器部署注意

在 AutoDL 等 GPU 服务器上，若平台已预装 PyTorch/CUDA，**不要**盲目 `uv sync` 覆盖环境。推荐按 [docs/git_sync.md](docs/git_sync.md) 与 [docs/server_rental_template.md](docs/server_rental_template.md) 使用 conda 镜像自带的 PyTorch，仅安装本项目额外依赖。

## 第三方 Baseline 仓库与权重

`external/` 目录下的第三方仓库和模型权重**不纳入 git**，需在每台机器上单独克隆/下载。

### LaMa

```bash
git clone https://github.com/advimman/lama external/lama   # 可选
```

权重在首次运行 `lama` baseline 时由 `simple-lama-inpainting` 自动下载（`big-lama.pt`，约 196MB）。

### MAT

```bash
git clone https://github.com/fenglinglwb/MAT external/mat
mkdir -p external/mat/pretrained
# Places-512 权重（约 661MB）
curl -L -o external/mat/pretrained/Places_512_FullData.pkl \
  "https://huggingface.co/Icar/mat_places512_full/resolve/main/Places_512_FullData.pkl"
```

MAT 要求输入边长为 **512 的整数倍**；框架会自动 pad 并在推理后裁回原尺寸。Mask 约定与项目一致：`1` = 保留，`0` = 缺失。

### RePaint

```bash
git clone https://github.com/andreas128/RePaint external/repaint
cd external/repaint && bash download.sh
```

默认使用 Places2-256 权重：`external/repaint/data/pretrained/places256_300000.pt`。推理配置见 `configs/repaint_places256.yml`；快速冒烟可用 `--fast` 切换 `repaint_places256_fast.yml`。

### DDRM

```bash
git clone https://github.com/BGUCompSci/DDRM external/ddrm
# 按官方 README 将 checkpoint 放到：
# external/ddrm/exp/logs/imagenet/256x256_diffusion_uncond.pt
```

默认 20 步扩散、`sigma_0=0`（无噪声 inpainting），配置见 `configs/ddrm_imagenet256.yml`。**需要 CUDA**。

### 冒烟测试

```bash
# LaMa
uv run python scripts/smoke_lama.py --experiment missing_patterns --case random
uv run baseline-suite run missing_patterns -b lama

# MAT
uv run python scripts/smoke_mat.py --experiment missing_patterns --case random
uv run baseline-suite run missing_patterns -b mat

# RePaint（GPU；--fast 仅用于快速验证，不计入正式指标）
uv run python scripts/smoke_repaint.py --experiment missing_patterns --case random --fast
uv run baseline-suite run missing_patterns -b repaint

# DDRM（GPU）
uv run python scripts/smoke_ddrm.py --experiment missing_patterns --case random
uv run baseline-suite run missing_patterns -b ddrm
```

## 实验约定

与原文及 `baseline增补计划/所需数据素材/*/备注.rtf` 保持一致：

| 项目 | 约定 |
|------|------|
| Mask 矩阵 | `1` = 保留（已知像素），`0` = 缺失 |
| 遮挡图像 | `masked = original ⊙ mask`（逐元素乘积） |
| Random mask | 全 1 矩阵，按缺失率随机置 0 |
| Block mask | 从 PNG 转换：黑色 → 1，白色 → 0 |
| 指标 | PSNR↑、SSIM↑、LPIPS↓、运行时间（秒） |

## 四组实验

配置见 `configs/experiments.yaml`：

| CLI 实验名 | 论文章节 | 说明 |
|------------|----------|------|
| `random_pixel_masks` | 4.3.1 | 30% / 50% / 70% 随机缺失，`leida.png`，20 次试验取平均 |
| `block_masks` | 4.3.2 | 300×300 块状遮挡（Image-2 / Image-4） |
| `missing_patterns` | 4.4 | 900×900，30% 缺失，对比 random vs block |
| `scenes_resolutions` | 4.5 | 多场景原分辨率（P0101 / P0410 / P1196） |

> **注意**：若 `block_masks` 素材中缺少 Image-2/4 的 RGB 原图，请在 `configs/experiments.yaml` 的 `cases[].image` 中指定正确路径。

## 使用方法

### 查看实验与 baseline 列表

```bash
uv run baseline-suite list
```

### 仅生成 mask（不跑模型）

```bash
uv run baseline-suite make-masks random_pixel_masks
uv run baseline-suite make-masks block_masks
```

生成的 mask 保存在 `data/masks/<experiment>/`，包含 `original.png`、`masked.png`、`mask_vis.png`、`keep_mask.npy`。

### 运行实验

```bash
# 流水线测试
uv run baseline-suite run missing_patterns -b identity

# 指定多个 baseline
uv run baseline-suite run scenes_resolutions -b lama mat

# 跑齐全部 4 个深度 baseline
uv run baseline-suite run missing_patterns -b lama mat repaint ddrm

# 对含多次 trial 的实验（如 4.3.1）打印平均指标
uv run baseline-suite run random_pixel_masks -b lama --aggregate
```

### 指定配置文件

```bash
uv run baseline-suite --config path/to/experiments.yaml run missing_patterns -b identity
```

### 生成论文表格

四组实验全部跑完后，从 `results/` 汇总为与原文格式对齐的 Markdown 表格：

```bash
uv run python scripts/generate_paper_tables.py
# 输出：results/paper_baseline_tables.md
```

该脚本会生成 Table 1–4 的完整版（含原文方法列）以及 **supplement-only** 表格（仅 4 个新增 baseline 列，对应论文标黄位置）。

## 输出说明

每次运行在 `results/<experiment>/` 下生成：

```
results/missing_patterns/
├── metrics.csv          # 所有 case × baseline 的指标
├── summary.json         # 运行摘要
├── random/
│   └── lama/
│       └── restored.png
└── block/
    └── lama/
        └── restored.png
```

`metrics.csv` 字段：`experiment`, `case`, `baseline`, `trial`, `psnr`, `ssim`, `lpips`, `time_sec`。

若 baseline 缺少外部仓库或权重，对应行会记录 `status` 和 `error` 信息。

## 扩展 Baseline

1. 在 `src/baseline_suite/baselines/` 下实现 `BaseInpainter` 子类
2. 在 `src/baseline_suite/registry.py` 中注册
3. 在 `configs/experiments.yaml` 的 `baselines` 列表中加入名称

详细对接步骤（mask 转换、尺寸限制、推理命令）见 **[docs/baseline_integration_checklist.md](docs/baseline_integration_checklist.md)**。

统一接口：

```python
def inpaint(self, masked_image, keep_mask, *, reference=None) -> InpaintResult:
    """
    masked_image: RGB float32, [0, 1]，已乘以 keep_mask
    keep_mask:    2D float32, 1=保留, 0=缺失
    reference:    原图 GT（部分方法需要未遮挡的原图输入）
    """
```

扩散类方法通常使用 hole mask（`1` = 缺失），可通过基类的 `hole_mask(keep_mask)` 转换。

## 数据素材

素材位于 `baseline增补计划/所需数据素材/`：

| 目录 | 用途 |
|------|------|
| `random-mask素材/` | 4.3.1、4.4（`leida.png`） |
| `block-mask素材/` | 4.3.2、4.4 的 block mask |
| `scene素材/` | 4.5 场景对比（`P0101.png` 等） |

## 推荐工作流

1. `uv sync` 安装依赖
2. `uv run baseline-suite make-masks <experiment>` 检查 mask 是否正确
3. 按上文说明克隆 `external/` 仓库并下载权重
4. 依次对各 baseline 跑 `scripts/smoke_*.py` 单图冒烟
5. 批量运行四组实验：`uv run baseline-suite run <experiment> -b lama mat repaint ddrm`
6. `uv run python scripts/generate_paper_tables.py` 生成论文表格
7. 从 `results/` 提取定性对比图，填入论文标黄位置

## 相关文档

| 文档 | 内容 |
|------|------|
| [docs/baseline_integration_checklist.md](docs/baseline_integration_checklist.md) | 各 baseline 的 mask 语义、尺寸、常见坑 |
| [docs/git_sync.md](docs/git_sync.md) | 本机与 AutoDL 服务器之间的代码/数据同步 |
| [docs/server_rental_template.md](docs/server_rental_template.md) | GPU 实例选型、环境配置、批量实验命令 |
