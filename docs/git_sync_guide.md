# Git 同步指南（本机 ↔ AutoDL 服务器）

本仓库用 Git 同步**代码与实验配置**；`external/` 第三方仓库和大权重在每台机器上单独克隆/下载。

---

## 一、本机（Mac）一次性设置

### 1. 确认已有首次提交

```bash
cd /Users/chuanxin/Diffusion/project1
git status
git log -1
```

### 2. 在 Gitee 或 GitHub 创建**私有**空仓库

- 国内 + AutoDL：推荐 **Gitee**（`gitee.com`），拉取更快
- 仓库名示例：`baseline-inpainting`（不要勾选 README 初始化，保持空仓库）

### 3. 关联远程并推送

**Gitee 示例：**

```bash
git remote add origin https://gitee.com/<你的用户名>/baseline-inpainting.git
git branch -M main
git push -u origin main
```

**GitHub 示例：**

```bash
git remote add origin https://github.com/<你的用户名>/baseline-inpainting.git
git branch -M main
git push -u origin main
```

首次 push 会提示输入账号密码或 Token。

---

## 二、服务器（AutoDL）首次拉取

SSH 登录服务器后：

```bash
cd /root
git clone https://gitee.com/<你的用户名>/baseline-inpainting.git project1
cd project1
```

### 拉取第三方 baseline 仓库

```bash
git clone https://github.com/advimman/lama external/lama
git clone https://github.com/fenglinglwb/MAT external/mat
git clone https://github.com/andreas128/RePaint external/repaint
git clone https://github.com/BGUCompSci/DDRM external/ddrm
```

### 下载 MAT 权重（不进 Git）

```bash
mkdir -p external/mat/pretrained
curl -L -o external/mat/pretrained/Places_512_FullData.pkl \
  "https://huggingface.co/Icar/mat_places512_full/resolve/main/Places_512_FullData.pkl"
```

### 安装依赖并验证

```bash
pip install -U uv -i https://pypi.tuna.tsinghua.edu.cn/simple
uv sync
uv run baseline-suite list
```

---

## 三、日常同步流程

### 在本机改代码后 → 同步到服务器

**本机：**

```bash
cd /Users/chuanxin/Diffusion/project1
git add .
git commit -m "描述你的修改"
git push
```

**服务器：**

```bash
cd /root/project1
git pull
```

### 在服务器改代码后 → 同步回本机

**服务器：**

```bash
cd /root/project1
git add .
git commit -m "描述修改"
git push
```

**本机：**

```bash
cd /Users/chuanxin/Diffusion/project1
git pull
```

---

## 四、实验结果怎么同步

`results/` 已在 `.gitignore` 中，**不会**进 Git（避免仓库过大）。

服务器跑完后，在本机执行（拉回结果）：

```bash
rsync -avz -e "ssh -p <端口>" \
  root@<主机>:/root/project1/results/ \
  /Users/chuanxin/Diffusion/project1/results/
```

---

## 五、不要提交的内容

| 内容 | 原因 | 服务器上如何获得 |
|------|------|------------------|
| `external/lama` 等 | 体积大、自带 `.git` | `git clone` |
| `*.pkl` / `big-lama.pt` | 数百 MB 权重 | 脚本下载或首次运行自动下 |
| `results/` | 实验输出 | `rsync` 拉回 |
| `.venv` | 环境 | `uv sync` |

---

## 六、常见问题

**`git pull` 冲突**  
先 `git status` 看冲突文件，手动改完后 `git add` + `git commit`，再 push。

**服务器没有配置 Git 账号**  

```bash
git config --global user.email "you@example.com"
git config --global user.name "Your Name"
```

推送私有仓可用 HTTPS + Token，或配置 SSH key。

**本机与服务器分支不一致**  
两边都保持在 `main`，推送前先 `git pull` 再 `git push`。
