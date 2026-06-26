# Data Download Guide

所有数据集下载命令。

## 1. 原始训练数据集 (4个源)

```bash
# PixMo-Points (~1.85M行, 主要数据源)
hf download allenai/pixmo-points --repo-type dataset --local-dir ./raw_data/pixmo_points

# RoboPoint (~667K行)
hf download wentao-yuan/robopoint-data --repo-type dataset --local-dir ./raw_data/robopoint

# RefSpatial (~1.8K行)
hf download JingkunAn/RefSpatial --repo-type dataset --local-dir ./raw_data/refspatial

# Where2Place (~100行)
hf download FlagEval/Where2Place --repo-type dataset --local-dir ./raw_data/where2place

# 国内加速 (可选):
# export HF_ENDPOINT=https://hf-mirror.com
```

## 2. PointArena评估数据

```bash
hf download PointArena/pointarena-data --repo-type dataset --local-dir ./eval_data

# 包含:
#   val_pointarena.jsonl  (982评估样本)
#   images/affordance/    (~40张)
#   images/counting/      (~40张)
#   images/reasoning/     (~40张)
#   images/spatial/       (~40张)
#   images/steerable/     (200张)
#   masks/{5 categories}/ (ground truth掩码)
```

## 3. 模型权重

```bash
# Molmo2-8B (transformers自动下载, ~16GB):
#   首次使用时自动从 allenai/Molmo2-8B 下载

# Qwen3-8B (用于本地数据管线):
hf download Qwen/Qwen3-8B --local-dir ./models/Qwen3-8B

# MolmoPoint-8B (替代backbone, 可选):
#   首次使用时自动从 allenai/MolmoPoint-8B 下载

# SAM3 ONNX (用于Steerable管线):
#   手动下载放置到 ./models/sam3.onnx
```

## 4. 处理后的数据 (由管线生成, 不需下载)

运行数据管线后自动生成:
- `data/cache/pointarena_rewritten_training_summary_steerable_d.json` (168MB)
- `pointarena_extract/` (Gemini产出)
- `clean_3types_local/` (Qwen产出)
- `make_steerable1/sam_clean/outputs/` (Steerable产出)

## 目录结构建议

```
PointArena/
├── raw_data/              # 原始HF数据集
│   ├── pixmo_points/
│   ├── robopoint/
│   ├── refspatial/
│   └── where2place/
├── eval_data/             # PointArena评估数据
│   ├── val_pointarena.jsonl
│   ├── images/
│   └── masks/
├── models/                # 模型权重
│   ├── Molmo2-8B/        # 自动下载
│   ├── Qwen3-8B/         # 手动下载
│   └── sam3.onnx         # 手动下载
├── processed_data/        # 管线产出
├── checkpoints/           # 训练检查点
│   └── checkpoint.pt     # 3.8GB
└── outputs/               # 训练输出
```

## 下载工具安装

```bash
# 安装 HuggingFace CLI 和 datasets (Python 3.10+):
pip install --user huggingface_hub datasets

# 确保 ~/.local/bin 在 PATH 中:
export PATH="$HOME/.local/bin:$PATH"

# 验证安装:
hf --version

# 或使用 python3 -m 方式:
python3 -m pip install --user huggingface_hub datasets

# 如果 pip 未安装, 先安装 pip:
# curl -s https://bootstrap.pypa.io/get-pip.py | python3

# 国内加速 (可选):
# export HF_ENDPOINT=https://hf-mirror.com

# 或使用 git-lfs 手动下载:
git lfs install
git clone https://huggingface.co/datasets/allenai/pixmo-points
```

**注意:**
- `hf` 已废弃且不再工作, 请使用 `hf` 命令代替。
- 如果 `hf` 不在 PATH 中, 使用 `python3 -m huggingface_hub` 或添加 `~/.local/bin` 到 PATH。
- `download_raw_datasets.py` 依赖 `datasets` 包。
- `download_pointarena_eval.py` 依赖 `huggingface_hub` 包。

## 安装下载工具

```bash
pip install huggingface_hub datasets
# 如果 hf 命令不可用，使用:
# python3 -m huggingface_hub download ...
```
