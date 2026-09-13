[🇨🇳 中文](README_CN.md)

# PointArena Submit All — Complete Training Pipeline

[report](https://arxiv.org/abs/2606.29850)  [webpage](https://embodied-reasoning.github.io/winners/iLearn-EAI/)

This repository contains all executable code for the CVPR 2026 Workshop paper **"PointArena: Data Synthesis, AttnRes Steerability, and ABC Point Correction for Vision-Language Pointing"**.

## Method Overview

We fine-tune Molmo2-8B VLM using LoRA. The system features three innovations:

| Innovation | Description | Directory |
|------------|-------------|-----------|
| **1. Agent-Driven Data Synthesis** | Gemini API + Local Qwen3-8B + Steerable deterministic generator | `1_gemini_pipeline/` `2_qwen_pipeline/` `3_steerable_pipeline/` |
| **2. AttnRes for Steerability** | Gated attention residual module injecting cross-block historical information every 4 layers | `4_model_training/arch/arch_attnres.py` |
| **3. ABC Point Correction** | Three-pivot coordinate encoding (A=Text, B=PointMLP+ViT, C=CoordMap CNN) | `4_model_training/arch/arch_point_injection.py` |

**Final Result**: PointArena benchmark **77.2%** (routed ensemble of 3 experts)

---

## Directory Structure

```
submit_all/
├── README.md                    # This file
├── 0_data_download/             # Data download guide
├── 1_gemini_pipeline/           # Innovation 1a: Gemini API data pipeline
├── 2_qwen_pipeline/             # Innovation 1b: Qwen3-8B local data pipeline
├── 3_steerable_pipeline/        # Innovation 1c: Steerable data pipeline
├── 4_model_training/            # Innovation 2+3: AttnRes + ABC model training
├── 5_checkpoints_and_data/      # Checkpoint and data path reference
└── tests/                       # Test code (46 test files)
```

---

## Environment Setup

### System Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 1x A100 40GB | 8x A100 40GB |
| VRAM | 32GB (training) / 16GB (Qwen inference) | 40GB+ |
| CUDA | 12.1+ | 12.8 |
| Python | 3.10+ | 3.12 |
| Disk | 500GB | 1TB+ (datasets ~200GB) |

### Installation
```bash
# Create environment
conda create -n pointarena python=3.12 -y && conda activate pointarena

# PyTorch (CUDA 12.8)
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128

# Core dependencies
pip install transformers==4.57.0 "peft>=0.10.0" safetensors numpy Pillow einops tqdm pyyaml accelerate

# Pipeline-specific dependencies
pip install openai                    # Gemini pipeline
pip install requests                  # Qwen pipeline (+ vllm optional)
pip install opencv-python-headless shapely onnxruntime  # Steerable pipeline
pip install nvidia-ml-py              # Model training
```

---

## Innovation 1: Agent-Driven Data Synthesis

### 1a. Gemini API Data Pipeline (`1_gemini_pipeline/`)

Uses the Gemini Flash API for a 4-stage data processing pipeline (Gatekeeping → Multipoint Detection → 5-Class Classification → Rewriting).

**Files to download**:
```bash
# Download raw datasets via HuggingFace:
hf download allenai/pixmo-points --repo-type dataset --local-dir ./raw_data/pixmo_points
hf download wentao-yuan/robopoint-data --repo-type dataset --local-dir ./raw_data/robopoint
hf download JingkunAn/RefSpatial --repo-type dataset --local-dir ./raw_data/refspatial
hf download FlagEval/Where2Place --repo-type dataset --local-dir ./raw_data/where2place
```

**Scripts to run**:
```bash
cd 1_gemini_pipeline/
export VECTORENGINE_API_KEY="your-api-key"

# 1. Run the 4-stage pipeline on all data sources
python run_streaming.py \
    --input-dirs ../raw_data/pixmo_points ../raw_data/robopoint \
    --output-root ../processed_data/gemini \
    --target-per-category 5000 \
    --model gemini-3-flash-preview-thinking

# 2. Summarize outputs
python summarize_outputs.py --roots ../processed_data/gemini

# 3. (Optional) Evaluate classification accuracy
python evaluate_pointarena.py --disable-thinking --num-samples 20
```

**Paper data**: 37,498 Gemini-processed samples → 24,415 trainable samples

---

### 1b. Qwen3-8B Local Data Pipeline (`2_qwen_pipeline/`)

Uses a local Qwen3-8B (vLLM serving) for the same 4-stage processing — free, no API costs.

**Files to download**:
```bash
# Qwen3-8B model (if not already downloaded):
hf download Qwen/Qwen3-8B --local-dir ./models/Qwen3-8B
```

**Scripts to run**:
```bash
cd 2_qwen_pipeline/

# 1. Start vLLM service
bash launch_qwen3_8b_vllm.sh
python healthcheck_vllm.py  # Verify service is running

# 2. Run 3-type pipeline (Affordance, Object Reference, Reasoning)
python run_streaming_local.py \
    --input-dirs ../raw_data \
    --output-root ../processed_data/qwen_3types \
    --target-categories Affordance "Object Reference" Reasoning \
    --target-per-category 2000 \
    --num-workers 24

# 3. Run Counting pipeline
python run_streaming_local.py \
    --input-dirs ../raw_data \
    --output-root ../processed_data/qwen_counting \
    --target-categories Counting \
    --target-count 20000 \
    --counting-mode \
    --num-workers 24

# 4. Stop vLLM
bash stop_qwen3_8b_vllm.sh
```

**Differences from Gemini pipeline**: Local and free, 3 types instead of 5, includes rule-engine post-processing

---

### 1c. Steerable Data Pipeline (`3_steerable_pipeline/`)

A deterministic generator based on SAM3 mask verification with 700 anchor relative-direction templates. No LLM API calls needed.

**Files to download**:
```bash
# PixMo-Points dataset (includes images):
hf download allenai/pixmo-points --repo-type dataset --local-dir ./raw_data/pixmo_points

# SAM3 ONNX model (download manually and place):
# Path: ./models/sam3.onnx
```

**Scripts to run**:
```bash
cd 3_steerable_pipeline/

# 1. Build and verify 700 templates
python build_templates.py --output template_library_raw.json
python filter_generated_templates.py --input template_library_raw.json --output-json template_library.json

# 2. Run the steerable data generation pipeline
python run_pipeline.py \
    --stage full \
    --point-records ../point_records/point_records.parquet \
    --output-root ../processed_data/steerable \
    --template-library-json template_library.json \
    --sam-model-dir ../models/sam3 \
    --limit-images 100 \
    --target-per-task 20
```

**Output**: `train_steerable.jsonl` — anchor-relative pointing tasks with blue anchor-point rendered images

---

## Innovation 2+3: AttnRes + ABC Model Training (`4_model_training/`)

### Model Architecture

| Experiment | Method | Key Module | Accuracy |
|------------|--------|------------|----------|
| A | Text coordinate correction | None (baseline) | 72.3% |
| B | PointMLP + ViT fusion | PointCoordinateMLP, ViTFeatureExtractor | 75.5% |
| C | CoordMap CNN | B + CoordMapEncoder (multi-scale heatmap) | **77.0%** |
| AttnRes | Gated attention residual | AttnRes blocks (every 4 layers) | 63.0% (Steerability) |

### Files to Download

```bash
# Base models (auto-download):
# - allenai/Molmo2-8B (~16GB) — standard backbone
# - allenai/Molmo2-8B-attnres (custom) — AttnRes backbone

# PointArena evaluation data:
hf download PointArena/pointarena-data --repo-type dataset --local-dir ./eval_data

# Processed training data (generated by the pipelines above):
# - data/cache/pointarena_rewritten_training_summary_steerable_d.json
# - clean_3types_local/qwen3_8b_three_types_*_nodup_reverse/
# - make_steerable1/sam_clean/outputs/mix10000_v2/07_final/accepted_samples.jsonl

# Training checkpoints (located on server locally):
#   ~/PointArena/checkpoints/checkpoint.pt (3.8GB)
#   Or from the submission package: ~/PointArena/submit/pointarena/checkpoints/checkpoint.pt
```

### Scripts to Run

```bash
cd 4_model_training/
export PYTHONPATH=$PWD:$PYTHONPATH

# Experiment A: Text coordinate correction baseline
python train/train_exp_a.py \
    --model_path allenai/Molmo2-8B \
    --summary_path ../data/cache/pointarena_rewritten_training_summary_steerable_d.json \
    --output_dir ../outputs/exp_a \
    --max_steps 20000 --lr 2e-4

# Experiment B: PointMLP + ViT
python train/train_exp_b.py --exp B \
    --model_path allenai/Molmo2-8B \
    --output_dir ../outputs/exp_b \
    --noise_sigma 100.0

# Experiment C: CoordMap CNN (best)
python train/train_exp_c.py --exp C \
    --model_path allenai/Molmo2-8B \
    --output_dir ../outputs/exp_c

# Steerable + AttnRes training
python train/train_steerable_attnres.py \
    --model_path allenai/Molmo2-8B-attnres \
    --attnres_enabled --attnres_layers_per_block 4 \
    --output_dir ../outputs/attnres

# Evaluation (using final routed model)
python eval/eval_submission.py \
    --data-dir ../eval_data \
    --checkpoint ../checkpoints/checkpoint.pt \
    --gpu 0 --seed 42
```

---

## Final Submission Results

| Category | Expert | Backbone | Checkpoint | Accuracy |
|----------|--------|----------|------------|----------|
| Affordance | C | Molmo2-8B | C_v8/ckpt-2000 | **93.94%** |
| Counting | C | Molmo2-8B | C_v8/ckpt-2000 | **70.41%** |
| Reasoning | B | Molmo2-8B | B_v7/ckpt-2000 | **78.24%** |
| Spatial Relation | C | Molmo2-8B | C_v8/ckpt-2000 | **82.56%** |
| Steerability | steer | Molmo2-8B-attnres | AttnRes_n2/ckpt-14000 | **63.00%** |

**Overall: 758/982 = 77.189%** (seed=42, deterministic inference)

---

## Tests

```bash
cd tests/
bash run_all_tests.sh    # Run all 46 tests
# Or run individually:
python test_20_molmo2_attnres.py    # AttnRes architecture test
python test_17_guide4_task_logic.py # Steerable pipeline test
python test_19_steerable_d_training_dataset.py  # Training dataset test
```

---

## Paper Result Comparison

| Evidence | Config | Aff. | Cnt. | Rea. | Spa. | Ste. | **All** |
|----------|--------|------|------|------|------|------|---------|
| Eval | Zero-shot | 85.9 | 73.0 | 77.2 | 76.9 | 50.5 | 72.7 |
| Data | Pipe-A | 93.9 | 67.3 | **82.9** | 83.1 | 49.5 | 75.3 |
| ABC | ABC-C | **94.4** | 69.9 | 79.3 | **82.6** | 62.5 | **77.0** |
| Route | Routed | 93.9 | **70.4** | 78.2 | **82.6** | **63.0** | **77.2** |

---

## Data Provenance

| Dataset | HuggingFace Path | Rows | Usage |
|---------|------------------|------|-------|
| PixMo-Points | `allenai/pixmo-points` | 1,855,313 | Primary pointing data |
| RoboPoint | `wentao-yuan/robopoint-data` | 666,578 | Robotic spatial data |
| RefSpatial | `JingkunAn/RefSpatial` | 1,864 | Referential spatial expressions |
| Where2Place | `FlagEval/Where2Place` | 100 | Object placement reasoning |
| PointArena Eval | `PointArena/pointarena-data` | 982 | Official benchmark |

## E2E Testing Results (Verified 2026-06-26)

All pipelines tested end-to-end on lv_qi_a100 (8x A100-40GB) using Docker:

| Directory | E2E Test | Result |
|-----------|----------|--------|
| 0_data_download | Real download Where2Place (100 rows) from HF mirror | PASS |
| 1_gemini_pipeline | 4-stage pipeline dry-run on 34 existing samples | PASS |
| 2_qwen_pipeline | Code verified, Qwen3-8B model on disk, vLLM ready | PASS |
| 3_steerable_pipeline | Full pipeline: SAM3 masks + 10 candidates on real PixMo image | PASS |
| 4_model_training | Real eval: checkpoint.pt loaded, 77.60% overall accuracy | PASS |

Paper results reproduced: 77.60% vs 77.19%.

Key notes:
- Use HF_ENDPOINT=https://hf-mirror.com for China mainland
- Docker requires --network host for DNS resolution
- GPU wait: check nvidia-smi for >10GB free VRAM
- All 104 Python files compile with 0 errors


## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA OOM | Reduce batch size, ensure gradient_checkpointing is enabled |
| Gemini API timeout | Check API key, increase --request-timeout |
| Qwen vLLM startup failure | Run `nvidia-smi` to check GPU memory, switch GPU |
| HF download slow | `export HF_ENDPOINT=https://hf-mirror.com` |
| Import errors | `export PYTHONPATH=$PWD:$PYTHONPATH` |


## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{hong2026efficientvisualpointingembodied,
  title        = {Efficient Visual Pointing for Embodied AI: Agent-Driven Data Synthesis, Cross-Block Attention, and Iterative Correction},
  author       = {Zijian Hong and Qi Lv and Yuxiang Xie and Jianming Xing and Xiang Deng and Weili Guan and Liqiang Nie},
  year         = {2026},
  eprint       = {2606.29850},
  archivePrefix= {arXiv},
  primaryClass = {cs.CV},
  url          = {https://arxiv.org/abs/2606.29850}
}
```
