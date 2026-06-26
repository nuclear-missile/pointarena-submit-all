# PointArena Submit All — Complete Training Pipeline

本仓库包含CVPR 2026 Workshop论文 **"PointArena: Data Synthesis, AttnRes Steerability, and ABC Point Correction for Vision-Language Pointing"** 的全部可运行代码。

## 论文方法概览

我们使用LoRA微调Molmo2-8B VLM，系统包含三个创新点：

| 创新点 | 内容 | 目录 |
|--------|------|------|
| **1. Agent-Driven Data Synthesis** | Gemini API + Qwen3-8B本地 + Steerable确定性生成器 | `1_gemini_pipeline/` `2_qwen_pipeline/` `3_steerable_pipeline/` |
| **2. AttnRes for Steerability** | 门控注意力残差模块，每4层注入跨块历史信息 | `4_model_training/arch/arch_attnres.py` |
| **3. ABC Point Correction** | 三支点坐标编码 (A=文本, B=PointMLP+ViT, C=CoordMap CNN) | `4_model_training/arch/arch_point_injection.py` |

**最终结果**: PointArena benchmark **77.2%** (routed ensemble of 3 experts)

---

## 目录结构

```
submit_all/
├── README.md                    # 本文件
├── 0_data_download/             # 数据下载指南
├── 1_gemini_pipeline/           # 创新点1a: Gemini API数据管线
├── 2_qwen_pipeline/             # 创新点1b: Qwen3-8B本地数据管线
├── 3_steerable_pipeline/        # 创新点1c: Steerable数据管线
├── 4_model_training/            # 创新点2+3: AttnRes + ABC模型训练
├── 5_checkpoints_and_data/      # 检查点和数据路径参考
└── tests/                       # 测试代码 (46个测试文件)
```

---

## 环境配置

### 系统要求
| 组件 | 最低配置 | 推荐配置 |
|------|---------|----------|
| GPU | 1× A100 40GB | 8× A100 40GB |
| VRAM | 32GB (训练) / 16GB (Qwen推理) | 40GB+ |
| CUDA | 12.1+ | 12.8 |
| Python | 3.10+ | 3.12 |
| 磁盘 | 500GB | 1TB+ (数据集约200GB) |

### 安装
```bash
# 创建环境
conda create -n pointarena python=3.12 -y && conda activate pointarena

# PyTorch (CUDA 12.8)
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128

# 核心依赖
pip install transformers==4.57.0 "peft>=0.10.0" safetensors numpy Pillow einops tqdm pyyaml accelerate

# 各管线额外依赖
pip install openai                    # Gemini管线
pip install requests                  # Qwen管线 (+ vllm可选)
pip install opencv-python-headless shapely onnxruntime  # Steerable管线
pip install nvidia-ml-py              # 模型训练
```

---

## 创新点1: Agent-Driven Data Synthesis

### 1a. Gemini API数据管线 (`1_gemini_pipeline/`)

使用Gemini Flash API进行4阶段数据处理 (Gatekeeping → Multipoint检测 → 5分类 → 改写)。

**要下载的文件**:
```bash
# 原始数据集通过HuggingFace下载:
hf download allenai/pixmo-points --repo-type dataset --local-dir ./raw_data/pixmo_points
hf download wentao-yuan/robopoint-data --repo-type dataset --local-dir ./raw_data/robopoint
hf download JingkunAn/RefSpatial --repo-type dataset --local-dir ./raw_data/refspatial
hf download FlagEval/Where2Place --repo-type dataset --local-dir ./raw_data/where2place
```

**要跑的脚本**:
```bash
cd 1_gemini_pipeline/
export VECTORENGINE_API_KEY="your-api-key"

# 1. 运行4阶段管线处理所有数据源
python run_streaming.py \
    --input-dirs ../raw_data/pixmo_points ../raw_data/robopoint \
    --output-root ../processed_data/gemini \
    --target-per-category 5000 \
    --model gemini-3-flash-preview-thinking

# 2. 汇总输出
python summarize_outputs.py --roots ../processed_data/gemini

# 3. (可选) 评估分类准确率
python evaluate_pointarena.py --disable-thinking --num-samples 20
```

**论文数据**: 37,498条Gemini处理 → 24,415条可训练样本

---

### 1b. Qwen3-8B本地数据管线 (`2_qwen_pipeline/`)

使用本地Qwen3-8B (vLLM serving) 进行同样的4阶段处理，免费无API费用。

**要下载的文件**:
```bash
# Qwen3-8B模型 (如果未下载):
hf download Qwen/Qwen3-8B --local-dir ./models/Qwen3-8B
```

**要跑的脚本**:
```bash
cd 2_qwen_pipeline/

# 1. 启动vLLM服务
bash launch_qwen3_8b_vllm.sh
python healthcheck_vllm.py  # 验证服务正常

# 2. 运行3类型管线 (Affordance, Object Reference, Reasoning)
python run_streaming_local.py \
    --input-dirs ../raw_data \
    --output-root ../processed_data/qwen_3types \
    --target-categories Affordance "Object Reference" Reasoning \
    --target-per-category 2000 \
    --num-workers 24

# 3. 运行Counting管线
python run_streaming_local.py \
    --input-dirs ../raw_data \
    --output-root ../processed_data/qwen_counting \
    --target-categories Counting \
    --target-count 20000 \
    --counting-mode \
    --num-workers 24

# 4. 停止vLLM
bash stop_qwen3_8b_vllm.sh
```

**与Gemini管线的区别**: 本地免费、3类型而非5类型、含规则引擎后处理

---

### 1c. Steerable数据管线 (`3_steerable_pipeline/`)

基于SAM3掩码验证的确定性生成器，700个锚点相对方向模板。无需LLM API调用。

**要下载的文件**:
```bash
# PixMo-Points数据集 (含图像):
hf download allenai/pixmo-points --repo-type dataset --local-dir ./raw_data/pixmo_points

# SAM3 ONNX模型 (手动下载放置):
# 放置路径: ./models/sam3.onnx
```

**要跑的脚本**:
```bash
cd 3_steerable_pipeline/

# 1. 构建和验证700模板
python build_templates.py --output template_library_raw.json
python filter_generated_templates.py --input template_library_raw.json --output template_library.json

# 2. 运行Steerable数据生成管线
python run_pipeline.py \
    --config config.yaml \
    --input-labels ../raw_data/pixmo_points \
    --output ../processed_data/steerable \
    --sam3-model ../models/sam3.onnx \
    --template-library template_library.json \
    --max-samples 10000
```

**输出**: `train_steerable.jsonl` — 带蓝色锚点渲染图像的锚点相对指向任务

---

## 创新点2+3: AttnRes + ABC模型训练 (`4_model_training/`)

### 模型架构

| 实验 | 方法 | 关键模块 | 准确率 |
|------|------|---------|--------|
| A | 文本坐标修正 | 无(基线) | 72.3% |
| B | PointMLP + ViT融合 | PointCoordinateMLP, ViTFeatureExtractor | 75.5% |
| C | CoordMap CNN | B + CoordMapEncoder (多尺度热图) | **77.0%** |
| AttnRes | 门控注意力残差 | AttnRes blocks (每4层) | 63.0% (Steerability) |

### 要下载的文件

```bash
# 基础模型 (自动下载):
# - allenai/Molmo2-8B (~16GB) — 标准backbone
# - allenai/Molmo2-8B-attnres (自定义) — AttnRes backbone

# PointArena评估数据:
hf download PointArena/pointarena-data --repo-type dataset --local-dir ./eval_data

# 处理后的训练数据 (由上方的管线生成):
# - data/cache/pointarena_rewritten_training_summary_steerable_d.json
# - clean_3types_local/qwen3_8b_three_types_*_nodup_reverse/
# - make_steerable1/sam_clean/outputs/mix10000_v2/07_final/accepted_samples.jsonl

# 训练检查点 (位于服务器本地):
#   ~/PointArena/checkpoints/checkpoint.pt (3.8GB)
#   或从提交包获取: ~/PointArena/submit/pointarena/checkpoints/checkpoint.pt
```

### 要跑的脚本

```bash
cd 4_model_training/
export PYTHONPATH=$PWD:$PYTHONPATH

# 实验A: 文本坐标修正基线
python train/train_exp_a.py \
    --model_path allenai/Molmo2-8B \
    --summary_path ../data/cache/pointarena_rewritten_training_summary_steerable_d.json \
    --output_dir ../outputs/exp_a \
    --max_steps 20000 --lr 2e-4

# 实验B: PointMLP + ViT
python train/train_exp_b.py --exp B \
    --model_path allenai/Molmo2-8B \
    --output_dir ../outputs/exp_b \
    --noise_sigma 100.0

# 实验C: CoordMap CNN (最佳)
python train/train_exp_c.py --exp C \
    --model_path allenai/Molmo2-8B \
    --output_dir ../outputs/exp_c

# Steerable + AttnRes训练
python train/train_steerable_attnres.py \
    --model_path allenai/Molmo2-8B-attnres \
    --attnres_enabled --attnres_layers_per_block 4 \
    --output_dir ../outputs/attnres

# 评估 (使用最终路由模型)
python eval/eval_submission.py \
    --data-dir ../eval_data \
    --checkpoint ../checkpoints/checkpoint.pt \
    --gpu 0 --seed 42
```

---

## 最终提交结果

| 类别 | 专家 | Backbone | 检查点 | 准确率 |
|------|------|---------|--------|--------|
| Affordance | C | Molmo2-8B | C_v8/ckpt-2000 | **93.94%** |
| Counting | C | Molmo2-8B | C_v8/ckpt-2000 | **70.41%** |
| Reasoning | B | Molmo2-8B | B_v7/ckpt-2000 | **78.24%** |
| Spatial Relation | C | Molmo2-8B | C_v8/ckpt-2000 | **82.56%** |
| Steerability | steer | Molmo2-8B-attnres | AttnRes_n2/ckpt-14000 | **63.00%** |

**Overall: 758/982 = 77.189%** (seed=42, 确定性推理)

---

## 测试

```bash
cd tests/
bash run_all_tests.sh    # 运行全部46个测试
# 或单独运行:
python test_20_molmo2_attnres.py    # AttnRes架构测试
python test_17_guide4_task_logic.py # Steerable管线测试
python test_19_steerable_d_training_dataset.py  # 训练数据集测试
```

---

## 论文结果对照

| Evidence | Config | Aff. | Cnt. | Rea. | Spa. | Ste. | **All** |
|----------|--------|------|------|------|------|------|---------|
| Eval | Zero-shot | 85.9 | 73.0 | 77.2 | 76.9 | 50.5 | 72.7 |
| Data | Pipe-A | 93.9 | 67.3 | **82.9** | 83.1 | 49.5 | 75.3 |
| ABC | ABC-C | **94.4** | 69.9 | 79.3 | **82.6** | 62.5 | **77.0** |
| Route | Routed | 93.9 | **70.4** | 78.2 | **82.6** | **63.0** | **77.2** |

---

## 数据溯源

| 数据集 | HuggingFace路径 | 行数 | 用途 |
|--------|----------------|------|------|
| PixMo-Points | `allenai/pixmo-points` | 1,855,313 | 主要指向数据 |
| RoboPoint | `wentao-yuan/robopoint-data` | 666,578 | 机器人空间数据 |
| RefSpatial | `JingkunAn/RefSpatial` | 1,864 | 引用空间表达 |
| Where2Place | `FlagEval/Where2Place` | 100 | 物体放置推理 |
| PointArena Eval | `PointArena/pointarena-data` | 982 | 官方benchmark |

## Troubleshooting

| 问题 | 解决方法 |
|------|---------|
| CUDA OOM | 减小batch size, 确保gradient_checkpointing开启 |
| Gemini API超时 | 检查API key, 增加--request-timeout |
| Qwen vLLM启动失败 | `nvidia-smi`检查GPU内存, 换GPU |
| HF下载慢 | `export HF_ENDPOINT=https://hf-mirror.com` |
| 导入错误 | `export PYTHONPATH=$PWD:$PYTHONPATH` |
