# Checkpoints and Data Reference

## 训练检查点

### 最终提交检查点
- **路径**: `~/PointArena/checkpoints/checkpoint.pt` (3.8 GB)
- **内容**: 所有3个专家的LoRA适配器 + 架构子模块 + 元数据
- **格式**: PyTorch checkpoint dict
```python
{
    "lora_adapters": {"C": {...}, "B": {...}, "steer": {...}, "counting": {...}},
    "lora_configs": {"C": {...}, "B": {...}, "steer": {...}, "counting": {...}},
    "arch_modules": {...},
    "meta": {...}
}
```

### 各专家检查点来源
| 专家 | 训练脚本 | 训练数据 | 最佳步数 |
|------|---------|---------|---------|
| C (Affordance/Counting/Spatial) | `run_lora_train_iter_pos_BC.py --exp C` | Steerable-D (12,680行) | 2,000 |
| B (Reasoning) | `run_lora_train_iter_pos_BC.py --exp B` | Steerable-D (12,680行) | 2,000 |
| Steer (Steerability) | Steerability sweep (AttnRes_n2) | Steerable-D + steer倾斜 | 14,000 |

### 训练输出目录 (在filtered_data_ft/下)
| 目录 | 实验 |
|------|------|
| `output_D_lr2e4_ga16/` | Steerable-D基线 |
| `output_AttnRes_D_lr2e4_ga16/` | AttnRes基线 |
| `output_D_steerability_lr2e4_ga16/` | Steerability sweep (Default + AttnRes, n=1/2/4) |
| `output_steerability_AttnRes_B/` | Steerability + Anchor PointMLP |
| `output_counting_sweep/` | Counting sweep (Default + AttnRes, n=2/3) |

---

## 模型权重

### 基础模型 (transformers自动下载)
| 模型 | HuggingFace路径 | 大小 | 用途 |
|------|----------------|------|------|
| Molmo2-8B | `allenai/Molmo2-8B` | ~16GB | 标准backbone (专家B, C) |
| MolmoPoint-8B | `allenai/MolmoPoint-8B` | ~16GB | 替代backbone变体 |

### 管线专用模型
| 模型 | 路径/下载方式 | 大小 | 用途 |
|------|-------------|------|------|
| Qwen3-8B | `hf download Qwen/Qwen3-8B` | ~16GB | Qwen本地数据管线 |
| SAM3 ONNX | 手动放置 | ~2GB | Steerable掩码生成 |

---

## 训练数据

### 原始数据集
```bash
hf download allenai/pixmo-points --repo-type dataset --local-dir ./raw_data/pixmo_points
hf download wentao-yuan/robopoint-data --repo-type dataset --local-dir ./raw_data/robopoint
hf download JingkunAn/RefSpatial --repo-type dataset --local-dir ./raw_data/refspatial
hf download FlagEval/Where2Place --repo-type dataset --local-dir ./raw_data/where2place
```

### 评估数据
```bash
hf download PointArena/pointarena-data --repo-type dataset --local-dir ./eval_data
```

### 处理后的训练数据 (由管线生成)
| 文件 | 生成管线 | 典型路径 |
|------|---------|---------|
| `summary_steerable_d.json` (168MB) | Gemini+Qwen+Steerable合并 | `data/cache/pointarena_rewritten_training_summary_steerable_d.json` |
| Gemini产出 | Gemini管线 | `pointarena_extract/mix4_*/*.json` |
| Qwen 3类型产出 | Qwen管线 | `clean_3types_local/qwen3_8b_three_types_*_nodup_reverse/` |
| Steerable产出 | Steerable管线 | `make_steerable1/sam_clean/outputs/mix10000_v2/07_final/accepted_samples.jsonl` |

---

## 数据流

```
原始HF数据集 (2.5M行)
  → 清洗 (reclean_from_raw_full.py)
  → Gemini管线 → 24,415可训练 (5类)
  → Qwen管线 → 3类+Counting补充数据
  → Steerable管线 → 锚点相对方向数据
  → Summary Cache合并 → summary_steerable_d.json
  → 平衡采样 → 训练集 (3,815 / 12,680 / 17,500行)
  → 训练循环 → LoRA适配器
  → 检查点合并 → checkpoint.pt (3.8GB)
  → 类别路由评估 → 77.2%
```

## GPU需求
| 阶段 | 最低VRAM | 推荐GPU |
|------|---------|---------|
| Gemini管线 | 无 (API) | CPU only |
| Qwen管线 | 16GB | 1× RTX 4090 |
| Steerable管线 | 8GB | 1× RTX 3090 |
| 训练 | 32GB | 1× A100 40GB |
| 评估 | 32GB | 1× A100 40GB |
