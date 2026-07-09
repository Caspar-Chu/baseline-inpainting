# Baseline Suite

遥感图像遮挡去除论文的 **baseline 增补实验框架**。根据审稿意见，在原文 Section 4.3–4.5 的对比实验中，补充 LaMa、MAT、RePaint、DDRM 等近年方法的恢复结果、性能指标与运行时间。

## 背景

原文任务：对被遮挡/破坏的遥感图像进行补全。遮挡方式包括：

- **Random mask**：按缺失率随机置零像素
- **Block mask**：块状连续遮挡（模拟云覆盖等）

输入为遮挡后的图像，输出为恢复后的完整图像。需在原文标黄的表格和图中，为每组实验新增 4 列 baseline 结果。

## 项目结构

```
project1/
├── baseline增补计划/          # 需求文档、原文 PDF、数据素材
├── configs/
│   └── experiments.yaml       # 4 组实验配置
├── src/baseline_suite/        # 核心 Python 包
│   ├── masks.py               # mask 生成
│   ├── io.py                  # 图像读写
│   ├── metrics.py             # PSNR / SSIM / LPIPS
│   ├── experiments.py         # 实验样本构建
│   ├── runner.py              # 批量运行
│   ├── registry.py            # baseline 注册
│   └── baselines/             # 各方法 wrapper
├── data/masks/                # 生成的 mask 缓存
├── results/                   # 恢复图与 metrics.csv
└── external/                  # 第三方 baseline 仓库
```

## 环境安装

本项目使用 [uv](https://github.com/astral-sh/uv) 管理依赖，要求 Python >= 3.12。

```bash
cd project1
uv sync
```

## 第三方 Baseline 仓库

将开源方法克隆到 `external/` 目录：

### LaMa（已接入）

```bash
git clone https://github.com/advimman/lama external/lama
```

权重在首次运行 `lama` baseline 时由 `simple-lama-inpainting` 自动下载（`big-lama.pt`，约 196MB）。

### MAT（已接入）

```bash
git clone https://github.com/fenglinglwb/MAT external/mat
mkdir -p external/mat/pretrained
# Places-512 权重（约 661MB），可从 HuggingFace 镜像下载：
curl -L -o external/mat/pretrained/Places_512_FullData.pkl \
  "https://huggingface.co/Icar/mat_places512_full/resolve/main/Places_512_FullData.pkl"
```

MAT 要求输入尺寸为 512 的整数倍；框架会自动 pad 并在推理后裁回原尺寸。Mask 约定与项目一致：`1`=保留，`0`=缺失。

### 其他 baseline（待接入）

```bash
git clone https://github.com/andreas128/RePaint external/repaint
git clone https://github.com/BGUCompSci/DDRM external/ddrm
```

RePaint / DDRM 还需按官方说明下载预训练权重。

### 冒烟测试

```bash
uv run python scripts/smoke_lama.py --experiment missing_patterns --case random
uv run baseline-suite run missing_patterns -b lama

uv run python scripts/smoke_mat.py --experiment missing_patterns --case random
uv run baseline-suite run missing_patterns -b mat
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

> **注意**：`block_masks` 素材目录目前 mainly 提供 mask 图。若原文 Image-2/4 的 RGB 原图不在素材中，请在 `configs/experiments.yaml` 的 `cases[].image` 中指定正确路径（当前 fallback 为 `leida.png`）。

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
# 用 identity baseline 做流水线测试
uv run baseline-suite run missing_patterns -b identity

# 指定多个 baseline
uv run baseline-suite run scenes_resolutions -b lama mat

# 对含多次 trial 的实验（如 4.3.1）打印平均指标
uv run baseline-suite run random_pixel_masks -b identity --aggregate
```

### 指定配置文件

```bash
uv run baseline-suite --config path/to/experiments.yaml run missing_patterns -b identity
```

## 输出说明

每次运行在 `results/<experiment>/` 下生成：

```
results/missing_patterns/
├── metrics.csv          # 所有 case × baseline 的指标
├── summary.json         # 运行摘要
├── random/
│   └── identity/
│       └── restored.png
└── block/
    └── identity/
        └── restored.png
```

`metrics.csv` 字段示例：`experiment`, `case`, `baseline`, `trial`, `psnr`, `ssim`, `lpips`, `time_sec`。

若 baseline 尚未实现或缺少外部仓库，对应行会记录 `status` 和 `error` 信息。

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
    """
```

扩散类方法通常使用 hole mask（`1=缺失`），可通过基类的 `hole_mask(keep_mask)` 转换。

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
3. 克隆 `external/lama` 并接入 wrapper，单图冒烟测试
4. 依次跑通 4 组实验 × 4 个 baseline
5. 从 `results/` 汇总表格与可视化图，填入论文标黄位置
