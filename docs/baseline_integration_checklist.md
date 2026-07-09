# Baseline 对接检查表

本文档将 **本项目约定** 与 **4 个 baseline 官方实现** 对齐，用于接入 `baseline_suite` 前的逐项核对。

---

## 0. 全局约定（先统一再对接）

### 本项目（原文实验）

| 项目 | 约定 |
|------|------|
| `keep_mask` | `1` = 保留（已知像素），`0` = 缺失 |
| 遮挡图 | `masked = original ⊙ keep_mask` |
| 评估 | 与 **原图 GT** 比 PSNR / SSIM / LPIPS |
| 计时 | 单次推理 wall-clock（秒），记录硬件 |

### Mask 方向转换速查

设本项目的 `keep_mask`（1=保留，0=缺失）：

| 方法 | 对方 mask 语义 | 转换公式 |
|------|---------------|----------|
| **LaMa** | 非零 = 待修复区域（hole） | `hole = (keep_mask == 0)` → PNG 白=255 |
| **MAT** | `0` = hole，`1` = 保留 | `mat = keep_mask`（0~1 或 ×255） |
| **RePaint** | `255` = 已知，`0` = 待生成 | `repaint = keep_mask × 255` |
| **DDRM** | 观测 `y = Hx`，`H` 为 mask 投影 | 用 `deg=inp`，`sigma_0=0`（无噪声） |

> **接入前必做**：用同一张图、同一个 mask，目视对比「遮挡区域」是否一致。

---

## 1. LaMa

**论文精读**：Abstract、Figure 1、实验部分（了解大 mask 能力即可）  
**代码精读**：GitHub README、`bin/predict.py`、`configs/prediction/default.yaml`

### 仓库与权重

| 项 | 内容 |
|----|------|
| Repo | https://github.com/advimman/lama |
| 克隆路径 | `external/lama` |
| 推荐权重 | `big-lama`（Places 预训练，通用场景首选） |
| 环境 | 按官方 README；需 `TORCH_HOME`、`PYTHONPATH=$(pwd)` |

### 输入 / 输出

| 项 | 要求 |
|----|------|
| 输入图像 | RGB，`png`/`jpg`，与 mask 同目录 |
| Mask 文件命名 | `{图像名}_mask001.png`（与图像 basename 对应） |
| Mask 像素值 | **非零 = hole**（`predict.py`: `mask > 0`） |
| 输出 | `outdir/` 下修复后 RGB 图 |
| 分辨率 | 支持任意分辨率；可选 `refine=True` 提升质量（更慢） |

### 推理命令

```bash
cd external/lama
export TORCH_HOME=$(pwd) PYTHONPATH=$(pwd)

python3 bin/predict.py \
  model.path=$(pwd)/big-lama \
  indir=/path/to/input_dir \
  outdir=/path/to/output_dir
```

### 对接本项目

- [ ] 从 `keep_mask` 生成 LaMa mask：`hole_uint8 = (1 - keep_mask) * 255`
- [ ] 按命名规则写入临时目录：`{name}.png` + `{name}_mask000.png`
- [ ] 原图用 **未遮挡的 GT** 还是 **masked 图**？→ LaMa 通常输入 **原图 + mask**（非 masked 乘积图）
- [ ] 在 `lama.py` wrapper 中封装目录准备 + subprocess 调用
- [ ] 跑 `missing_patterns` 单 case 冒烟

### 对应实验组

| 实验 | 注意点 |
|------|--------|
| 4.3.1 random | 20 trials × 3 缺失率；mask 每次不同 |
| 4.3.2 block | 300×300；mask 从 block 素材转换 |
| 4.4 | 900×900；random / block 各 1 次 |
| 4.5 | 原分辨率；遥感图与 Places 分布差异大，如实报告 |

### 常见坑

- mask 文件名不符合 `{img}_maskXXX` 规则 → 找不到配对
- mask 方向反了 → 修复已知区、破坏缺失区
- 误传 `masked_image` 而非原图 → 结果偏差

### 预期耗时

- 单张 512²：约 **1–5 s**（GPU）
- 900² / 原分辨率：更长；`refine=True` 显著增加时间

---

## 2. MAT

**论文精读**：Abstract、Introduction 贡献、实验设置  
**代码精读**：`generate_image.py`、README Quick Test

### 仓库与权重

| 项 | 内容 |
|----|------|
| Repo | https://github.com/fenglinglwb/MAT |
| 克隆路径 | `external/mat` |
| 权重 | OneDrive（README 链接）→ `pretrained/Places365.pkl` 或 `CelebA-HQ.pkl` |
| 遥感图建议 | 优先 **Places365** 权重（非人脸） |

### 输入 / 输出

| 项 | 要求 |
|----|------|
| 输入 | RGB 图像目录 `--dpath` |
| Mask | 可选 `--mpath`；灰度图 `/255` 后 **0=hole，1=保留** |
| 尺寸 | **边长必须是 512 的倍数**；不足则 pad，**mask 用 0 pad** |
| 输出 | `--outdir` 下与输入同名的修复图 |

### 推理命令

```bash
cd external/mat
python generate_image.py \
  --network pretrained/Places365.pkl \
  --dpath /path/to/images \
  --mpath /path/to/masks \
  --outdir /path/to/output
```

### 对接本项目

- [ ] `mat_mask = keep_mask`（float 0~1 或 uint8 0/255 经 `/255`）
- [ ] 实现 **pad 到 512 倍数**，推理后 **crop 回原尺寸**
- [ ] 图像与 mask 文件名一一对应、数量相同
- [ ] 确认输入是 GT 原图 + mask（非 masked 乘积图）

### 对应实验组

| 实验 | 注意点 |
|------|--------|
| 4.3.2 (300×300) | 需 pad 到 512×512，结果 crop 回 300×300 再算指标 |
| 4.4 (900×900) | 900 已是 512 的倍数？→ 900 不是！需 pad 到 1024 |
| 4.5 | 各场景尺寸不同，逐个处理 pad/crop |

### 常见坑

- 尺寸非 512 倍数 → 直接报错
- pad 后忘记 crop → 指标对不齐
- 用 CelebA 权重跑遥感 → 效果可能很差（仍可作为 baseline）

### 预期耗时

- 512²：约 **1–3 s**/张（GPU）

---

## 3. RePaint

**论文精读**：Abstract、**Section 3 Method**（反向扩散 + resampling）  
**代码精读**：README、`confs/*.yml`、`test.py`

### 仓库与权重

| 项 | 内容 |
|----|------|
| Repo | https://github.com/andreas128/RePaint |
| 克隆路径 | `external/repaint` |
| 权重 | `download.sh` 下载（ImageNet / CelebA-HQ / Places2） |
| 遥感建议 | 复制 `confs/test_inet256_*.yml` 或 `test_p256_*.yml` 改路径 |

### 输入 / 输出

| 项 | 要求 |
|----|------|
| GT 路径 | config 中 `gt_path` |
| Mask 路径 | config 中 `mask_path` |
| Mask 像素值 | **255 = 已知区域，0 = 待生成** |
| 典型分辨率 | 256×256（与预训练 DDPM 一致） |
| 输出 | `log/.../inpainted/` |

### 关键配置（`schedule_jump_params`）

| 参数 | 含义 | 速度/质量权衡 |
|------|------|---------------|
| `t_T` | 总扩散步数 | 减小 → 更快，质量可能下降 |
| `jump_length` | 每次跳跃步长 | — |
| `jump_n_sample` | 重采样次数 | 减小 → 更快 |

### 推理命令

```bash
cd external/repaint
bash download.sh   # 首次
python test.py --conf_path confs/face_example.yml
```

自定义数据：复制 config，修改 `gt_path`、`mask_path`、`image_size`。

### 对接本项目

- [ ] `repaint_mask = keep_mask * 255`（uint8）
- [ ] 非 256 尺寸：resize/pad 到 256，推理后 resize 回原尺寸（与论文设定对齐需在文中说明）
- [ ] 记录 `t_T`、`jump_n_sample` 和 GPU 型号（影响 Table 3/4 的时间对比）
- [ ] 扩散模型随机性：固定 seed，必要时多次采样取最好或平均（与原文 20 trials 策略区分）

### 对应实验组

| 实验 | 注意点 |
|------|--------|
| 4.3.1 | 70% 缺失率对大 mask 友好，但 256 限制需处理 |
| 4.4 | 需记录 **Clock time**；RePaint 通常最慢 |
| 全部 | 预训练在自然图像上，遥感为 out-of-distribution |

### 常见坑

- mask 方向与 LaMa 相同（已知=255），但与 MAT 相反（MAT 保留=1）
- 忘记 `download.sh` → 缺 checkpoint
- 直接跑 900² 不 resize → 与预训练不匹配

### 预期耗时

- 256²，默认 schedule：**数十秒～数分钟**/张（GPU）
- 减 `t_T` / `jump_n_sample` 可加速，需在论文中注明设置

---

## 4. DDRM

**论文精读**：Abstract、inpainting 段落（\(y=Hx\)、\(H=\mathrm{diag}\) mask）、实验表（PSNR/SSIM/NFEs）  
**代码精读**：README、`main.py`、`configs/*.yml`、`deg=inp` 相关逻辑

### 仓库与权重

| 项 | 内容 |
|----|------|
| Repo | https://github.com/bahjat-kawar/ddrm |
| 克隆路径 | `external/ddrm` |
| 权重 | `exp/logs/imagenet/` 下 diffusion checkpoint（见 README 目录结构） |
| 环境 | PyTorch 1.8–1.10（较老；可能需要独立 conda 环境） |

### 输入 / 输出

| 项 | 要求 |
|----|------|
| 退化类型 | `--deg inp`（inpainting） |
| 噪声 | `--sigma_0 0.0`（无噪声，与原文一致） |
| 步数 | `--timesteps 20`（论文示例；影响 NFEs 与时间） |
| 超参 | `--eta 0.85 --etaB 1`（默认） |
| 输出 | `image_samples/{folder}/` 下 `orig_*.png`、`y0_*.png`、`*_-1.png`（恢复） |

### 推理命令

```bash
cd external/ddrm
python main.py --ni \
  --config imagenet_256.yml \
  --doc imagenet \
  --timesteps 20 \
  --eta 0.85 --etaB 1 \
  --deg inp \
  --sigma_0 0.0 \
  -i my_inpaint_run
```

### 对接本项目

- [ ] 确认 DDRM 如何传入**自定义 mask**（需读 `main.py` 中 `inp` 分支；可能需改 config 或放图到 `exp/datasets/`）
- [ ] 图像尺寸通常 **256×256**（ImageNet 预训练）
- [ ] 记录 `timesteps`（= NFEs 相关）和运行时间
- [ ] 评估时用 `*_-1.png` 与 GT 对齐尺寸后算指标

### 对应实验组

| 实验 | 注意点 |
|------|--------|
| 4.3.1 | 论文 inpainting 示例为 50% random drop；你方 30/50/70% 需自定义 mask |
| 4.4 | 同时报告 time；DDRM 比 RePaint 快（论文称约 5×） |
| 全部 | 代码年代较早，接入成本可能最高 |

### 常见坑

- PyTorch 版本不兼容
- `exp/` 目录结构与 checkpoint 路径繁琐
- 自定义 mask 接口不直观，需读源码

### 预期耗时

- 256²，20 steps：约 **数秒～十几秒**/张（GPU，依实现而定）

---

## 5. GLTF-Net（本轮可选，低优先级）

| 项 | 说明 |
|----|------|
| 任务 | **多时相**厚云去除，非单图 inpainting |
| 数据 | 需要同一区域多个时间点影像 |
| 与当前框架 | 与 random/block mask 单图设定 **不一致** |
| 建议 | 确认导师是否纳入本轮；若做需单独设计实验管线 |

---

## 6. 按实验组的统一执行清单

每跑完一个 baseline × 一个实验组，勾选：

### 4.3.1 Random Pixel Masks

- [ ] 使用 `leida.png`
- [ ] 缺失率 30% / 50% / 70%
- [ ] 每个率 **20 次 trial**，不同 seed
- [ ] 对 20 次结果取 **PSNR/SSIM/LPIPS 平均**
- [ ] 选 1 个缺失率保存定性图（对齐 Figure 5–7 布局）

### 4.3.2 Block Masks

- [ ] Image-2 / Image-4，300×300
- [ ] Block mask 来自 `block-mask素材/`
- [ ] 补全原图路径（若素材仅有 mask）
- [ ] 对齐 Table 2、Figure 8–9

### 4.4 Missing Patterns

- [ ] 900×900，`leida.png` resize
- [ ] 固定 30%：random + block 各 1 组
- [ ] 记录 **time_sec**（Table 3）
- [ ] 同一硬件、同一 batch 设置

### 4.5 Scenes & Resolutions

- [ ] P0101 / P0410 / P1196，**原分辨率**
- [ ] 30% random mask，seed 固定
- [ ] 记录 time + 三指标（Table 4）

---

## 7. 推荐接入顺序与冒烟测试

```
Step 1  LaMa     → leida.png, 30% random, 1 trial
Step 2  MAT      → 同上，验证 pad/crop
Step 3  RePaint  → 256 resize，验证 mask 255/0
Step 4  DDRM     → deg=inp, sigma_0=0, 验证自定义 mask
Step 5  批量     → uv run baseline-suite run <experiment> -b lama mat repaint ddrm
```

### 单方法冒烟通过标准

- [ ] 输出图视觉合理（缺失区有内容，已知区未破坏）
- [ ] PSNR/SSIM 数量级与 identity baseline 有显著差异
- [ ] mask 方向自检：与 `data/masks/.../mask_vis.png` 一致
- [ ] 运行时间已记录

---

## 8. 论文写作时需注明的信息

在修订稿中建议统一说明（避免审稿人质疑公平性）：

1. 各 deep baseline 使用 **官方预训练权重**，未在遥感数据上微调  
2. 输入尺寸处理方式（pad/resize 到 256 或 512）  
3. 随机方法（RePaint/DDRM）的 **seed / 步数** 设置  
4. 计时硬件（GPU 型号、CUDA 版本）  
5. Mask 语义与各方法官方实现的对齐方式  

---

## 9. 快速对照：谁用什么 mask

```
本项目 keep_mask:     白(1)=保留    黑(0)=缺失
         ↓ 转换
LaMa hole mask:       白(255)=缺失  黑(0)=保留
MAT mask:             白(1)=保留    黑(0)=缺失   ← 与 keep_mask 同向
RePaint mask:         白(255)=保留  黑(0)=缺失   ← 与 keep_mask 同向（×255）
DDRM:                 通过 H 投影，缺失像素在 y 中不可见
```

建议在 `src/baseline_suite/baselines/` 各 wrapper 里写单元测试：

```python
# 伪代码：中心 50×50 区域缺失
keep = np.ones((100, 100))
keep[25:75, 25:75] = 0
# 转换后目视保存，确认 hole 在中心
```
