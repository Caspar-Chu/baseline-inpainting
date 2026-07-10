# Git 同步指南（本机 ↔ 服务器）

**定位**：用 Git 在 Mac 与 GPU 服务器之间同步**代码与配置**。  
不涵盖：服务器环境搭建 → [server_rental_template.md](server_rental_template.md)；baseline 技术细节 → [baseline_integration_checklist.md](baseline_integration_checklist.md)。

本仓库 Git 同步**代码与实验配置**；`external/` 第三方仓库、模型权重、`results/` 输出均不纳入版本控制。

---

## 一、本机一次性设置

```bash
cd /Users/chuanxin/Diffusion/baseline
git status && git log -1
```

在 Gitee 或 GitHub 创建**私有空仓库**（国内 + AutoDL 推荐 Gitee）：

```bash
git remote add origin https://gitee.com/<用户名>/baseline-inpainting.git
git branch -M main
git push -u origin main
```

---

## 二、服务器首次拉取

```bash
cd /root
git clone https://gitee.com/<用户名>/baseline-inpainting.git baseline
cd baseline
```

拉取后按 [server_rental_template.md](server_rental_template.md) 配置 Python 环境与 `external/` 权重。**不要在服务器上执行 `uv sync`**。

---

## 三、日常同步

### 本机 → 服务器

```bash
# 本机
cd /Users/chuanxin/Diffusion/baseline
git add . && git commit -m "描述修改" && git push

# 服务器
cd /root/baseline && git pull
```

### 服务器 → 本机

反向操作：服务器 `commit` + `push`，本机 `git pull`。

两边保持在 `main` 分支；推送前先 `git pull` 避免冲突。

---

## 四、实验结果同步

`results/` 在 `.gitignore` 中，**不进 Git**。服务器跑完后在本机拉回：

```bash
rsync -avz -e "ssh -p <端口>" \
  root@<主机>:/root/baseline/results/ \
  /Users/chuanxin/Diffusion/baseline/results/
```

或用 `scp -P <端口> -r root@<主机>:/root/baseline/results/ ...`（见 [服务器指南 §8](server_rental_template.md#8-结果拉回本机)）。

---

## 五、不提交的内容

| 路径/类型 | 原因 | 服务器上如何获得 |
|-----------|------|------------------|
| `external/*` | 体积大、自带 `.git` | 克隆或 Mac scp，见 [服务器指南 §4](server_rental_template.md#4-第三方仓库与权重) |
| `*.pkl` / `*.pt` | 数百 MB 权重 | 下载或首次运行自动下 |
| `results/` | 实验输出 | `rsync` / `scp` 拉回 |
| `.venv` | 虚拟环境 | 本机 `uv sync`；服务器 pip 安装 |
| `*.pdf` | 论文体积大 | 仅保留本机 |

---

## 六、常见问题

**`git pull` 冲突**  
`git status` 查看冲突文件，手动解决后 `git add` + `git commit`。

**服务器未配置 Git 身份**

```bash
git config --global user.email "you@example.com"
git config --global user.name "Your Name"
```

私有仓推送用 HTTPS + Token 或 SSH key。

**替代方案：不用 Git**  
若 GitHub/Gitee 访问不便，可用 tar + scp 整包上传，流程见 [server_rental_template.md §3](server_rental_template.md#3-mac--服务器部署主流程)。
