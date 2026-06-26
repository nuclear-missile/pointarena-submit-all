# Model Training: AttnRes + ABC (创新点2+3)

AttnRes (门控注意力残差) + ABC三支点坐标编码。完整训练和评估流程。

## 运行环境要求

所有训练和评估脚本需要 GPU + PyTorch 环境。基础 Python 语法检查可通过 `py_compile` 完成，
但实际运行需要以下依赖：

```bash
# 必需依赖:
pip install torch transformers peft safetensors numpy Pillow einops accelerate tqdm pyyaml nvidia-ml-py
```

## 要下载的文件

```bash
# 1. 基础模型 (transformers自动下载):
#    - allenai/Molmo2-8B (~16GB) — 标准backbone
#    - Molmo2-8B-attnres (自定义, 有AttnRes块的backbone)

# 2. PointArena评估数据:
hf download PointArena/pointarena-data --repo-type dataset --local-dir ./eval_data

# 3. 处理后的训练数据 (由数据管线生成):
#    - data/cache/pointarena_rewritten_training_summary_steerable_d.json (168MB)
#    - clean_3types_local/qwen3_8b_three_types_*_nodup_reverse/ (Qwen产出)
#    - make_steerable1/sam_clean/outputs/mix10000_v2/07_final/accepted_samples.jsonl (Steerable产出)

# 4. 训练检查点:
#    位于服务器: ~/PointArena/checkpoints/checkpoint.pt (3.8GB)
#    或: ~/PointArena/submit/pointarena/checkpoints/checkpoint.pt
```

## 快速检查 (无需GPU)

```bash
cd 4_model_training/
export PYTHONPATH=$PWD:$PYTHONPATH

# 编译检查所有模块
python3 -m py_compile arch/arch_attnres.py && echo "OK"
python3 -m py_compile data/data_common.py && echo "OK"
python3 -m py_compile train/utils_seed.py && echo "OK"
python3 -m py_compile eval/reorder_val.py && echo "OK"
```

## 要跑的脚本

### 实验A: 文本坐标修正 (基线)
```bash
cd 4_model_training/
export PYTHONPATH=$PWD:$PYTHONPATH

python train/train_exp_a.py \
    --model_path allenai/Molmo2-8B \
    --summary_path ../data/cache/pointarena_rewritten_training_summary_steerable_d.json \
    --output_dir ../outputs/exp_a \
    --max_steps 20000 \
    --lr 2e-4 \
    --context_len 2048
# 预期: ~72.3% overall
```

### 实验B: PointMLP + ViT
```bash
python train/train_exp_b.py --exp B \
    --model_path allenai/Molmo2-8B \
    --summary_path ../data/cache/pointarena_rewritten_training_summary_steerable_d.json \
    --output_dir ../outputs/exp_b \
    --noise_sigma 100.0 \
    --lr 1e-4
# 预期: ~75.5% overall
```

### 实验C: CoordMap CNN (最佳)
```bash
python train/train_exp_c.py --exp C \
    --model_path allenai/Molmo2-8B \
    --summary_path ../data/cache/pointarena_rewritten_training_summary_steerable_d.json \
    --output_dir ../outputs/exp_c \
    --noise_sigma 100.0 \
    --lr 1e-4
# 预期: ~77.0% overall
```

### Steerable + AttnRes训练
```bash
python train/train_steerable_attnres.py \
    --model_path allenai/Molmo2-8B-attnres \
    --summary_path ../data/cache/pointarena_rewritten_training_summary_steerable_d.json \
    --attnres_enabled \
    --attnres_layers_per_block 4 \
    --clean3_roots_json [../clean_3types_local/qwen3_8b_three_types_2000_each_nodup_reverse, ../clean_3types_local/qwen3_8b_three_types_add2000_nodup_reverse] \
    --sam_clean_jsonl ../make_steerable1/sam_clean/outputs/mix10000_v2/07_final/accepted_samples.jsonl \
    --output_dir ../outputs/attnres
```

### 评估 (最终路由提交)
```bash
python eval/eval_submission.py \
    --data-dir ../eval_data \
    --checkpoint ../checkpoints/checkpoint.pt \
    --gpu 0 --seed 42
# 预期: 758/982 = 77.189%
```

## 架构详解

### AttnRes (注意力残差)

    H_i = H_i + tanh(alpha) * AttnRes(H_i; H_{i-3:i-1})

- 每4个transformer层注入一次
- Gate alpha 初始化为0 (训练开始时为恒等映射)
- 跨块softmax注意力聚合历史block状态
- **仅用于Steerability和Counting** (消融实验表明对非steerable类别有害)
- 实现在 `arch/arch_attnres.py`

### ABC三支点编码
| 支 | 方法 | 关键模块 | 准确率 |
|----|------|---------|--------|
| A | 文本坐标追加 | 无 | 72.3% |
| B | PointMLP + ViT | PointCoordinateMLP (2->4096) + ViTFeatureExtractor (层[8,12,16]) | 75.5% |
| C | CoordMap CNN | B + CoordMapEncoder (100x100x4热图CNN -> 16 tokens) | **77.0%** |
- 视觉编码驱动精度提升; 额外修正轮次改善稳定性而非峰值
- 实现在 `arch/arch_point_injection.py`

### 类别路由 (最终提交)
| 类别 | 专家 | Backbone | 检查点 | 准确率 |
|------|------|---------|--------|--------|
| Affordance | C | Standard Molmo2 | C_v8/ckpt-2000 | 93.94% |
| Counting | C | Standard Molmo2 | C_v8/ckpt-2000 | 70.41% |
| Reasoning | B | Standard Molmo2 | B_v7/ckpt-2000 | 78.24% |
| Spatial | C | Standard Molmo2 | C_v8/ckpt-2000 | 82.56% |
| Steerability | steer | AttnRes Molmo2 | AttnRes_n2/ckpt-14000 | 63.00% |

## 共享超参数
| 参数 | 值 |
|------|-----|
| LoRA rank | 64, alpha=128, dropout=0.05 |
| 学习率 | 2e-4 (A), 1e-4 (B/C) |
| Batch | 1 micro x 16 GA |
| 最大步数 | 20,000 |
| 上下文长度 | 2048, max_crops=12 |
| LR调度 | Cosine, 100步warmup |
| 精度 | bf16 + AMP |

## 目录结构
```
4_model_training/
+-- arch/           # 架构模块 (AttnRes, ABC, IterPos, CoordMap)
+-- data/           # 数据集类和数据整理器
+-- train/          # 训练脚本 (实验A/B/C, Steerable, 训练循环)
+-- eval/           # 提交模型和评估入口
```
