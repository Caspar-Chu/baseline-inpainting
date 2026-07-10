# 服务器部署与批量实验指南

**定位**：租 GPU、把项目部署到 AutoDL 等服务器、跑批量实验。  
本机开发见 [README](../README.md)；Git 同步见 [git_sync.md](git_sync.md)；baseline 技术细节见 [baseline_integration_checklist.md](baseline_integration_checklist.md)。

---

## 当前实例（已租用）

| 项 | 配置 | 对本项目的意义 |
|----|------|----------------|
| **GPU** | RTX 40 系 × 1，显存 **24GB** | 满足 RePaint / DDRM 推理；与文档推荐的 4090 同级 |
| **CPU** | AMD，**14 核** | 数据预处理、mask 生成绰绰有余 |
| **内存** | **64GB** | 高于推荐下限（32GB），四 baseline 并行无压力 |
| **系统盘** | **150GB** | 足够存放代码 + 4 个 `external/` 仓库 + 全部权重 + `results/` |
| **网络** | 同地区实例共享 **2Gb** 带宽 | 大文件（MAT 661MB 权重）建议从 Mac **scp** 上传，比服务器下载快 |

**结论**：该配置**完全满足**本项目四组实验 × 四个 baseline 的批量运行需求，无需升配。

上线后先执行环境验证（§3.3），再按 §5 顺序冒烟 → 批量跑。

---

## 1. 实例选型（参考）

| 项 | 推荐 |
|----|------|
| 平台 | AutoDL（国内短期实验首选） |
| GPU | 1× RTX 4090 24GB（备选 3090 / A5000） |
| 内存 | 32GB+ |
| 系统盘 | 100GB+（建议 150GB） |
| OS | Ubuntu 22.04 LTS |
| 镜像 | `cuda128_torch291_py312`（CUDA 12.8 + PyTorch 2.9.1 + Python 3.12） |

下单时关闭自动续费；安全组仅放行 SSH（22）。

> **关键**：使用镜像**自带 PyTorch**，**不要**在服务器上 `uv sync`（会装 Python 3.14 + cu130 torch，与驱动不兼容）。

### 磁盘空间预估（150GB 实例）

| 内容 | 约占用 |
|------|--------|
| 项目代码 + 实验素材 | < 1GB |
| `external/lama` + `big-lama.pt` | ~0.5GB |
| `external/mat` + Places 权重 | ~0.7GB |
| `external/repaint` + Places2 权重 | ~1–2GB |
| `external/ddrm` + ImageNet checkpoint | ~2–3GB |
| `results/`（四组实验全部输出） | < 5GB |
| **合计** | **约 10–15GB**，远低于 150GB 上限 |

---

## 2. 部署方式选择

| 方式 | 适用场景 |
|------|----------|
| **tar + scp**（§3，推荐） | 国内服务器 GitHub 慢、本机已有 `external/` 权重 |
| **Git clone**（§3 备选） | 已配置 Gitee 镜像，见 [git_sync.md](git_sync.md) |

---

## 3. Mac → 服务器部署（主流程）

### 3.1 Mac 打包

```bash
cd /Users/chuanxin/Diffusion

tar czf baseline-deploy.tar.gz \
  --exclude='baseline/.venv' \
  --exclude='baseline/.git' \
  --exclude='baseline/*.pdf' \
  --exclude='baseline/baseline增补计划.zip' \
  --exclude='baseline/results/*' \
  baseline
```

会包含代码、`configs/`、实验素材、本机已有的 `external/lama` / `external/mat`（含权重）。`results/` 和 PDF 不包含。

### 3.2 上传与解压

```bash
scp -P <端口> /Users/chuanxin/Diffusion/baseline-deploy.tar.gz root@<主机>:/root/

# 服务器
cd /root && tar xzf baseline-deploy.tar.gz && cd baseline
ls baseline增补计划/所需数据素材 && ls external/
```

### 3.3 验证 GPU

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

必须输出 `True`。若失败，重装镜像 torch：

```bash
pip install --force-reinstall torch==2.9.1 torchvision \
  --index-url https://download.pytorch.org/whl/cu128
```

### 3.4 安装项目依赖

```bash
cd /root/baseline

pip install easydict imageio-ffmpeg lpips opencv-python-headless \
  pandas pillow pyspng pyyaml requests scikit-image simple-lama-inpainting \
  timm tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install -e . --no-deps    # 务必 --no-deps，避免覆盖 torch
baseline-suite list
```

服务器上用 **`python`** 直接运行，不要用 `uv run`。

---

## 4. 第三方仓库与权重

`external/` 不纳入 git。本机与服务器操作相同。

### LaMa

```bash
git clone https://github.com/advimman/lama external/lama   # 可选
```

权重首次运行 `lama` 时由 `simple-lama-inpainting` 自动下载（约 196MB）。

### MAT

```bash
git clone https://github.com/fenglinglwb/MAT external/mat
mkdir -p external/mat/pretrained
curl -L -o external/mat/pretrained/Places_512_FullData.pkl \
  "https://huggingface.co/Icar/mat_places512_full/resolve/main/Places_512_FullData.pkl"
```

国内服务器可用 HF 镜像：`export HF_ENDPOINT=https://hf-mirror.com`，或从 Mac scp 该文件（约 661MB，比服务器下载快）。

### RePaint

```bash
git clone https://github.com/andreas128/RePaint external/repaint
cd external/repaint && bash download.sh
# → data/pretrained/places256_300000.pt
```

### DDRM

```bash
git clone https://github.com/BGUCompSci/DDRM external/ddrm
# checkpoint → external/ddrm/exp/logs/imagenet/256x256_diffusion_uncond.pt
```

各方法 mask 语义与尺寸处理见 [baseline_integration_checklist.md](baseline_integration_checklist.md)。

---

## 5. 冒烟与批量实验

### 冒烟（先验证再批量）

```bash
cd /root/baseline
python scripts/smoke_lama.py --experiment missing_patterns --case random
python scripts/smoke_mat.py   --experiment missing_patterns --case random
python scripts/smoke_repaint.py --experiment missing_patterns --case random --fast   # --fast 仅冒烟
python scripts/smoke_ddrm.py    --experiment missing_patterns --case random
```

### 批量运行

```bash
baseline-suite run missing_patterns -b lama mat repaint ddrm
baseline-suite run random_pixel_masks -b lama mat repaint ddrm --aggregate
baseline-suite run block_masks -b lama mat repaint ddrm
baseline-suite run scenes_resolutions -b lama mat repaint ddrm
```

### 长任务（tmux）

```bash
tmux new -s baseline
baseline-suite run random_pixel_masks -b lama mat repaint ddrm
# Ctrl+b, d 断开；tmux attach -t baseline 恢复
```

### 成本控制

- 调试阶段只跑 smoke + `missing_patterns`
- `random_pixel_masks`（20 trials × 3 率）最耗时，确认冒烟通过后再跑
- 不用时立即关机

---

## 6. 数据路径检查

```bash
ls baseline增补计划/所需数据素材/random-mask素材/leida.png
ls baseline增补计划/所需数据素材/block-mask素材/
ls baseline增补计划/所需数据素材/scene素材/P0101.png
```

路径定义在 `configs/experiments.yaml`。

---

## 7. 禁止事项

| 不要做 | 原因 |
|--------|------|
| 服务器 `uv sync` | Python 3.14 + cu130 torch 与 CUDA 12.8 不兼容 |
| `pip install -e .` 不带 `--no-deps` | 覆盖镜像自带 torch |
| 依赖服务器 `git clone github.com` | 国内易超时，优先 Mac 打包或 Gitee |
| 上传 PDF 论文 | 实验不需要 |

---

## 8. 结果拉回本机

```bash
# Mac 本地
scp -P <端口> -r root@<主机>:/root/baseline/results/ \
  /Users/chuanxin/Diffusion/baseline/results/

# 或 rsync（见 git_sync.md §四）
```

拉回后在本机生成论文表格：

```bash
uv run python scripts/generate_paper_tables.py
```

---

## 9. 故障排查

| 症状 | 处理 |
|------|------|
| `torch.cuda.is_available()` 为 False | `nvidia-smi` 检查驱动；误跑过 `uv sync` 则重装 torch（§3.3）或重开实例 |
| MAT 权重缺失 | 检查 `external/mat/pretrained/Places_512_FullData.pkl` 约 661MB |
| RePaint checkpoint 缺失 | `cd external/repaint && bash download.sh` |
| 路径错误 | 确认在 `/root/baseline`，检查 `configs/experiments.yaml` |
| mask 方向反了 | 对比 `data/masks/.../mask_vis.png`，见 [检查表 §0](baseline_integration_checklist.md#0-全局约定) |

---

## 10. 交付前检查清单

- [ ] `nvidia-smi` 正常，`torch.cuda.is_available()` 为 `True`
- [ ] `baseline-suite list` 可运行
- [ ] `external/` 仓库与权重就绪（LaMa / MAT / RePaint / DDRM）
- [ ] 四个 `smoke_*.py` 均通过
- [ ] 四组实验 `metrics.csv` 已生成
- [ ] `results/` 已拉回本机，`paper_baseline_tables.md` 已生成

---

## 附录：服务器一键块（上传解压后执行）

```bash
cd /root/baseline
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
pip install easydict imageio-ffmpeg lpips opencv-python-headless \
  pandas pillow pyspng pyyaml requests scikit-image simple-lama-inpainting \
  timm tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e . --no-deps
baseline-suite list
python scripts/smoke_lama.py --experiment missing_patterns --case random
python scripts/smoke_mat.py   --experiment missing_patterns --case random
```
