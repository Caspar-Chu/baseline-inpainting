# 服务器租借配置模板（Baseline Inpainting 项目）

本模板用于当前项目 `project1`：补充 LaMa / MAT / RePaint / DDRM baseline，并完成 Section 4.3.1–4.5 对比实验。

---

## 1. 推荐平台与机型

### 平台优先级（国内）

1. **AutoDL（首推）**：上手快，按小时计费，适合短期实验。
2. **腾讯云 / 阿里云**：稳定性好，适合长期固定资源。
3. **海外平台（RunPod / Lambda）**：仅在网络和支付条件合适时考虑。

### 推荐实例规格

- **GPU**：1 x RTX 4090 24GB（首选）  
  备选：RTX 3090 24GB / A5000 24GB
- **vCPU**：8 核及以上
- **内存**：32GB 及以上
- **系统盘**：100GB 及以上（建议 150GB）
- **OS**：Ubuntu 22.04 LTS

---

## 2. 下单配置清单（可直接照填）

### 必填项

- 地域：离你网络近、库存稳定的区域
- 镜像：`Ubuntu 22.04 + CUDA + PyTorch`（平台现成深度学习镜像）
- GPU 数量：1
- GPU 型号：4090 24GB（无则 3090 24GB）
- 云盘：100GB+
- 公网：开启（便于 SSH 与下载模型）

### 可选项

- 自动续费：**关闭**（防止忘关机）
- 安全组：只放行 `22`（SSH）与必要端口
- 预装 Jupyter：可不开（本项目以 CLI 为主）

---

## 3. 开机后环境初始化（一次性）

> 下面命令可整段执行。

```bash
sudo apt update
sudo apt install -y git curl wget build-essential tmux htop unzip
nvidia-smi
```

安装 `uv`（项目约定）：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
uv --version
```

---

## 4. 项目拉取与目录约定

```bash
git clone <YOUR_REPO_URL> ~/project1
cd ~/project1
```

目录以当前项目为准：

- 代码根目录：`~/project1`
- 第三方仓库目录：`~/project1/external/`
- 实验结果目录：`~/project1/results/`

---

## 5. 依赖安装（主环境）

```bash
cd ~/project1
uv sync
uv run baseline-suite list
```

---

## 6. baseline 仓库与权重准备

### 6.1 拉取 external 仓库

```bash
mkdir -p ~/project1/external
git clone https://github.com/advimman/lama ~/project1/external/lama
git clone https://github.com/fenglinglwb/MAT ~/project1/external/mat
git clone https://github.com/andreas128/RePaint ~/project1/external/repaint
git clone https://github.com/BGUCompSci/DDRM ~/project1/external/ddrm
```

### 6.2 MAT 权重（已在项目中验证）

```bash
mkdir -p ~/project1/external/mat/pretrained
curl -L -o ~/project1/external/mat/pretrained/Places_512_FullData.pkl \
  "https://huggingface.co/Icar/mat_places512_full/resolve/main/Places_512_FullData.pkl"
```

### 6.3 LaMa 权重

- 首次运行时由 `simple-lama-inpainting` 自动下载 `big-lama.pt`。

### 6.4 RePaint / DDRM 权重

- 按各自官方 README 下载到 `external/repaint` 与 `external/ddrm` 对应目录。

---

## 7. 数据集路径检查（必须）

当前配置读取：

- `baseline增补计划/所需数据素材/random-mask素材/leida.png`
- `baseline增补计划/所需数据素材/block-mask素材/...`
- `baseline增补计划/所需数据素材/scene素材/P0101.png` 等

检查命令：

```bash
ls -la ~/project1/baseline增补计划/所需数据素材
```

---

## 8. 冒烟测试（先验证再批量跑）

```bash
cd ~/project1
uv run python scripts/smoke_lama.py --experiment missing_patterns --case random
uv run python scripts/smoke_mat.py --experiment missing_patterns --case random
```

---

## 9. 与当前进度对齐（LaMa + MAT 的 missing_patterns）

```bash
cd ~/project1
uv run baseline-suite run missing_patterns -b lama mat
```

结果查看：

- `results/missing_patterns/metrics.csv`
- `results/missing_patterns/random/lama/restored.png`
- `results/missing_patterns/random/mat/restored.png`
- `results/missing_patterns/block/lama/restored.png`
- `results/missing_patterns/block/mat/restored.png`

---

## 10. 长任务运行模板（tmux）

```bash
tmux new -s baseline
cd ~/project1
uv run baseline-suite run random_pixel_masks -b lama mat
# 断开会话：Ctrl+b, d
# 恢复会话：
tmux attach -t baseline
```

---

## 11. 成本控制模板

- 调试阶段只跑 smoke 和 `missing_patterns`
- 批量阶段集中跑完 `random_pixel_masks` 后及时关机
- 不用时立即释放实例（避免空转计费）

---

## 12. 故障排查速查

### CUDA 不可用

```bash
nvidia-smi
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### 权重缺失

- 检查 `external/mat/pretrained/Places_512_FullData.pkl` 是否存在
- 检查文件大小是否约 661MB

### 路径错误

- 确认当前目录是 `~/project1`
- 确认 `configs/experiments.yaml` 中路径与服务器文件一致

---

## 13. 交付前检查清单

- [ ] `uv sync` 成功
- [ ] `baseline-suite list` 可运行
- [ ] LaMa smoke 成功
- [ ] MAT smoke 成功
- [ ] `missing_patterns` 完整输出成功
- [ ] `results/missing_patterns/metrics.csv` 含 `lama` 与 `mat`
- [ ] 结果图已生成并可打开

---

## 14. 一键执行块（最小可用）

```bash
cd ~/project1
uv sync
uv run python scripts/smoke_lama.py --experiment missing_patterns --case random
uv run python scripts/smoke_mat.py --experiment missing_patterns --case random
uv run baseline-suite run missing_patterns -b lama mat
```

