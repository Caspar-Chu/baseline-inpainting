# Baseline 对接检查表

**定位**：各 baseline 的 mask 语义、官方 API 与 wrapper 实现的对照参考。部署与权重下载见 [server_rental_template.md](server_rental_template.md)；CLI 用法见 [README](../README.md)。

四个 baseline（LaMa / MAT / RePaint / DDRM）已在 `src/baseline_suite/baselines/` 接入。本文档用于排查问题、写论文说明、或新增 baseline 时对照。

---

## 0. 全局约定

| 项目 | 约定 |
|------|------|
| `keep_mask` | `1` = 保留（已知像素），`0` = 缺失 |
| 遮挡图 | `masked = original ⊙ keep_mask` |
| 评估 | 与 **原图 GT** 比 PSNR / SSIM / LPIPS |
| 计时 | 单次推理 wall-clock（秒），记录硬件 |

### Mask 转换速查

| 方法 | 对方 mask 语义 | 从 `keep_mask` 转换 |
|------|---------------|---------------------|
| **LaMa** | 非零 = hole | `hole_uint8 = (1 - keep_mask) * 255` |
| **MAT** | `0` = hole，`1` = 保留 | `mat_mask = keep_mask`（同向） |
| **RePaint** | `255` = 已知，`0` = 待生成 | `repaint_mask = keep_mask * 255` |
| **DDRM** | 观测 `y = Hx`，缺失像素不可见 | `deg=inp`，`sigma_0=0` |

```
本项目 keep_mask:   白(1)=保留    黑(0)=缺失
LaMa hole mask:     白(255)=缺失  黑(0)=保留
MAT / RePaint:      白=保留       黑=缺失（RePaint ×255）
```

> 接入或排错时：用同一张图、同一 mask，对比 `data/masks/.../mask_vis.png` 与各方法输入是否一致。

---

## 1. LaMa

**Wrapper**：`baselines/lama.py`（基于 `simple-lama-inpainting`，权重自动下载）

| 项 | 内容 |
|----|------|
| 官方 Repo | https://github.com/advimman/lama |
| 权重 | `big-lama`（Places，约 196MB） |
| 输入 | **原图 GT** + hole mask（非 masked 乘积图） |
| 分辨率 | 任意；官方可选 `refine=True`（更慢） |

**常见坑**：mask 方向反了；误传 `masked_image` 而非原图；官方 CLI 要求 `{img}_maskXXX.png` 命名（wrapper 已处理）。

**实验注意**：4.5 遥感图与 Places 分布差异大，如实报告；900² / 原分辨率耗时显著增加。

---

## 2. MAT

**Wrapper**：`baselines/mat.py`

| 项 | 内容 |
|----|------|
| 官方 Repo | https://github.com/fenglinglwb/MAT |
| 权重 | `external/mat/pretrained/Places_512_FullData.pkl`（约 661MB） |
| Mask | `0` = hole，`1` = 保留（与 `keep_mask` 同向） |
| 尺寸 | **边长须为 512 倍数**；wrapper 自动 pad（mask 用 0）并 crop 回原尺寸 |

**常见坑**：300×300 需 pad 到 512；900×900 需 pad 到 1024；pad 后忘记 crop 会导致指标错位；CelebA 权重不适合遥感。

**实验注意**：

| 实验 | 尺寸处理 |
|------|----------|
| 4.3.2 (300²) | pad → 512² → crop 回 300² |
| 4.4 (900²) | pad → 1024² → crop 回 900² |
| 4.5 | 各场景尺寸不同，逐个 pad/crop |

---

## 3. RePaint

**Wrapper**：`baselines/repaint.py`；配置 `configs/repaint_places256.yml`（正式）/ `repaint_places256_fast.yml`（快速冒烟）

| 项 | 内容 |
|----|------|
| 官方 Repo | https://github.com/andreas128/RePaint |
| 权重 | `external/repaint/data/pretrained/places256_300000.pt` |
| Mask | `255` = 已知，`0` = 待生成 |
| 分辨率 | 256×256（wrapper resize 后还原） |
| GPU | **必需** |

**关键超参**（`schedule_jump_params`）：`t_T`（总步数）、`jump_n_sample`（重采样次数）——影响 Table 3/4 的时间，论文中须注明。

**常见坑**：忘记 `download.sh`；直接跑 900² 不 resize；扩散随机性——wrapper 固定 seed=42。

**实验注意**：4.4 需记录 clock time，RePaint 通常最慢；`--fast` 仅用于冒烟，不计入正式指标。

---

## 4. DDRM

**Wrapper**：`baselines/ddrm.py`；配置 `configs/ddrm_imagenet256.yml`

| 项 | 内容 |
|----|------|
| 官方 Repo | https://github.com/BGUCompSci/DDRM |
| 权重 | `external/ddrm/exp/logs/imagenet/256x256_diffusion_uncond.pt` |
| 退化 | `deg=inp`，`sigma_0=0`（无噪声） |
| 步数 | 默认 20（`timesteps`） |
| 分辨率 | 256×256（wrapper resize 后还原） |
| GPU | **必需** |

**常见坑**：checkpoint 路径繁琐；官方代码较老，环境与 PyTorch 版本需与镜像匹配。

---

## 5. 按实验组的执行清单

每跑完一个 baseline × 实验组，勾选：

### 4.3.1 `random_pixel_masks`

- [ ] `leida.png`；缺失率 30% / 50% / 70%
- [ ] 每个率 20 trials，不同 seed
- [ ] 对 20 次取 PSNR/SSIM/LPIPS 均值（`--aggregate`）
- [ ] 保存定性图对齐 Figure 5–7

### 4.3.2 `block_masks`

- [ ] Image-2 / Image-4，300×300
- [ ] Block mask 来自 `block-mask素材/`
- [ ] 对齐 Table 2、Figure 8–9

### 4.4 `missing_patterns`

- [ ] 900×900，`leida.png` resize；30% random + block 各 1 组
- [ ] 记录 `time_sec`（Table 3）；同一硬件

### 4.5 `scenes_resolutions`

- [ ] P0101 / P0410 / P1196 原分辨率；30% random，seed=42
- [ ] 记录 time + 三指标（Table 4）

---

## 6. 冒烟通过标准

```bash
uv run python scripts/smoke_<method>.py --experiment missing_patterns --case random
```

- [ ] 缺失区有合理填充，已知区未被破坏
- [ ] PSNR/SSIM 与 `identity` 有显著差异
- [ ] mask 方向与 `data/masks/.../mask_vis.png` 一致
- [ ] `time_sec` 已记录

推荐顺序：LaMa → MAT → RePaint → DDRM → 批量 `baseline-suite run`。

---

## 7. 论文写作须注明

1. 各 deep baseline 使用 **官方预训练权重**，未在遥感数据上微调
2. 输入尺寸处理（MAT pad 512 倍数；RePaint/DDRM resize 256）
3. RePaint/DDRM 的 **seed / 扩散步数** 设置
4. 计时硬件（GPU 型号、CUDA 版本）
5. Mask 语义对齐方式（见本文 §0）

---

## 附录：GLTF-Net（本轮不纳入）

多时相厚云去除，与当前单图 random/block mask 设定不一致；若导师要求纳入需单独设计管线。
