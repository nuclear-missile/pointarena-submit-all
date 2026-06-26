[🇨🇳 中文](README_CN.md)

# Qwen3-8B Local Data Pipeline (Innovation 1b)

This pipeline uses a **local Qwen3-8B model served via vLLM** to perform the same 4-stage data processing as the Gemini pipeline, but **completely free with no API costs**. It focuses on 3 categories (Affordance, Object Reference, Reasoning) plus a separate Counting category, and applies a rule-based post-processing engine to correct misclassifications based on linguistic patterns.

> **Note**: Requires Python 3.10+. Install dependencies: `pip install -r requirements.txt`

## Environment Requirements

Ensure the following Python dependencies are installed before running:

```bash
pip install openai>=1.0.0 tqdm>=4.60.0 requests>=2.28.0 Pillow>=9.0.0 numpy>=1.21.0
```

If `pip` is not available on your system, install it first:
```bash
curl -sS https://bootstrap.pypa.io/get-pip.py | python3
```

**Important:** Scripts in this directory depend on the `clean_3types_local` module, located at the project root `~/PointArena/clean_3types_local/`. The scripts automatically add the parent directory to the Python search path.

## Files to Download

```bash
# 1. Qwen3-8B model:
hf download Qwen/Qwen3-8B --local-dir ./models/Qwen3-8B

# 2. Raw training datasets (if not already downloaded):
hf download allenai/pixmo-points --repo-type dataset --local-dir ./raw_data/pixmo_points
hf download wentao-yuan/robopoint-data --repo-type dataset --local-dir ./raw_data/robopoint
hf download JingkunAn/RefSpatial --repo-type dataset --local-dir ./raw_data/refspatial
```

## Scripts to Run

### 1. Start Qwen3-8B vLLM Service
```bash
cd 2_qwen_pipeline/
bash launch_qwen3_8b_vllm.sh
# Verify the service:
python healthcheck_vllm.py
# Expected output: "Qwen3-8B server healthy at http://127.0.0.1:8018"
```

### 2. 3-Type Pipeline (Affordance / Object Reference / Reasoning)
```bash
python run_streaming_local.py \
    --input-dirs ../raw_data/pixmo_points ../raw_data/robopoint \
                  ../raw_data/refspatial \
    --output-root ../processed_data/qwen_3types \
    --target-categories Affordance "Object Reference" Reasoning \
    --target-per-category 2000 \
    --num-workers 24
```

### 3. Counting Pipeline
```bash
# Method A: Streaming pipeline (requires Qwen LLM)
python run_streaming_local.py \
    --input-dirs ../raw_data/pixmo_points ../raw_data/robopoint \
    --output-root ../processed_data/qwen_counting \
    --target-categories Counting \
    --target-count 20000 \
    --counting-mode \
    --num-workers 24

# Method B: Direct build (no LLM needed, pure rule-based)
python build_counting_direct.py \
    --input ../raw_data \
    --output ../processed_data/counting_direct.jsonl
```

### 4. Stop vLLM
```bash
bash stop_qwen3_8b_vllm.sh
```

## Pipeline Stages (Same as Gemini)

```
Input (JSONL, containing query + points + image_path)
   |
   |-- Stage 1: CLEAN -> Validate validity, extract core_target
   |-- Stage 2: MULTIPOINT -> Detect multi-point and counting tasks
   |-- Stage 3: CLASSIFY -> 3-class classification (or forced Counting)
   +-- Stage 4: REWRITE -> "Point to ..." format, with JSON fallback retry
```

### Post-Processing: Rule Engine (`three_type_rules.py`)

Pattern-based rule corrections applied on top of Qwen classification outputs to improve accuracy:

| Pattern | Forced Category |
|---------|-----------------|
| "book about ..." | Reasoning |
| "tool for ..." | Affordance |
| "current point" / "existing point" | Object Reference |

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen3-8B |
| vLLM URL | `http://127.0.0.1:8018/v1` |
| Temperature | 0 (deterministic) |
| Max Context | 8192 tokens |
| Thinking | disabled |
| Parallel Workers | 24 |
| Per-sample Latency | ~1 second |

## Comparison with Gemini Pipeline

| Aspect | Qwen Pipeline | Gemini Pipeline |
|--------|---------------|-----------------|
| Model | Qwen3-8B (local) | Gemini Flash (API) |
| Cost | Free | API call fees |
| Classification | 3 classes + Counting | 5 classes |
| Input | Text only | Text only |
| Post-processing | Rule engine | None |
| Speed | ~1 sec/sample | ~2-3 sec/sample |

## Output Format (Same as Gemini)

Each sample produces one JSON file with a full 4-stage audit trail:
```json
{
  "id": "pixmo_points::0001__abc123",
  "status": "completed",
  "category": "Affordance",
  "final_query": "Point to the coffee cup.",
  "stages": {
    "cleaning": {"keep": true, "core_target": "coffee cup"},
    "multipoint": {"multi_point": false},
    "classification": {"category": "Affordance"},
    "rewrite": {"final_query": "Point to the coffee cup."}
  },
  "session_messages": [...]
}
```

## Output Data Locations
- 3-type results: `clean_3types_local/qwen3_8b_three_types_2000_each_nodup_reverse/`
- 3-type additional: `clean_3types_local/qwen3_8b_three_types_add2000_nodup_reverse/`
- Counting results: `clean_counting_local/` (corresponding output directory)