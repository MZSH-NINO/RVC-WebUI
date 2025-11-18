# RVC-WebUI 一键部署工具

## 🚀 快速开始

### 双击运行 `一键部署.bat` 即可自动完成部署！

脚本会引导您完成：
1. 选择显卡类型（NVIDIA / AMD / CPU）
2. 自动安装所有依赖
3. 下载必需的模型文件

---

## 📁 文件说明

- **一键部署.bat** - 交互式部署脚本（主要文件）
- **download_ffmpeg.py** - ffmpeg 自动下载工具
- **部署使用说明.md** - 详细部署文档
- **README.md** - 本文件

---

## 🎯 部署流程

### 1. 运行部署脚本

双击 `一键部署.bat`，按照提示操作：

```
步骤 1/2: 选择您的显卡类型
  [1] NVIDIA 显卡 (推荐)
  [2] AMD / Intel 显卡
  [3] 仅使用 CPU
  [0] 退出安装
```

### 2. 确认配置

系统会显示检测到的配置，确认后自动开始安装。

### 3. 等待完成

安装过程约 10-30 分钟，请耐心等待。

---

## ✅ 安装完成后

### 快速启动

双击项目根目录的 **`一键启动.bat`**，系统会：
- 自动激活 Conda 环境
- 启动 WebUI
- 自动打开浏览器访问 http://127.0.0.1:7865

### 手动启动

```bash
# 1. 激活环境
conda activate RVC-WebUI

# 2. 启动 WebUI
python infer-web.py
```

---

## 📋 系统要求

### 必需

- **操作系统**：Windows 10/11 64位
- **内存**：≥8GB RAM
- **磁盘**：≥10GB 可用空间
- **网络**：稳定的互联网连接
- **Conda**：已安装 Miniconda 或 Anaconda

### 推荐（GPU 加速）

- **NVIDIA 显卡**：RTX 2060 或更高（6GB+ 显存）
- **AMD 显卡**：RX 5000 系列或更高
- **Intel 显卡**：Arc 系列

---

## ❓ 常见问题

### Q: 提示"未检测到 Conda 环境"

**A:** 请先安装 Miniconda：
- 下载地址：https://docs.conda.io/en/latest/miniconda.html
- 安装后重新运行脚本

### Q: 模型下载很慢或失败

**A:** 可以稍后手动下载：
```bash
conda activate RVC-WebUI
python tools\download_models.py
```

### Q: AMD 显卡性能如何？

**A:** AMD/Intel 显卡使用 DirectML 加速：
- 性能约为 NVIDIA 显卡的 30-70%
- 仍远快于 CPU 模式
- 推荐使用 NVIDIA 获得最佳体验

### 更多问题

查看 **`部署使用说明.md`** 获取详细解答。

---

## 📧 技术支持

- **GitHub**：https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
- **文档**：查看项目 Wiki 和 Issues

---

**祝您使用愉快！🎉**
