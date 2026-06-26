[🇨🇳 中文](README_CN.md)

# Gemini API Data Pipeline (Innovation 1a)

This pipeline leverages the **Gemini Flash API** (via [vectorengine.ai](https://api.vectorengine.ai)) to perform a **4-stage data cleaning, classification, and rewriting** process on raw pointing data. It processes samples from multiple datasets (PixMo-Points, RoboPoint, RefSpatial, Where2Place) through a sequential multi-stage transformer that filters invalid samples, detects multi-point requirements, classifies into 5 categories, and standardizes all queries into a consistent format.

> **Note**: Requires Python 3.10+. Install dependencies: `pip install -r requirements.txt`

## Files to Download

```bash
# 1. Raw PointArena training datasets:
hf download allenai/pixmo-points --repo-type dataset --local-dir ./raw_data/pixmo_points
hf download wentao-yuan/robopoint-data --repo-type dataset --local-dir ./raw_data/robopoint
hf download JingkunAn/RefSpatial --repo-type dataset --local-dir ./raw_data/refspatial
hf download FlagEval/Where2Place --repo-type dataset --local-dir ./raw_data/where2place

# 2. API Key: Obtain from https://api.vectorengine.ai
```

## Scripts to Run

### 1. Main Data Processing Pipeline (Core)
```bash
cd 1_gemini_pipeline/
export VECTORENGINE_API_KEY="your-api-key"

# Run the 4-stage pipeline
python run_streaming.py \
    --sources pixmo_points,robopoint,refspatial,where2place \
    --output-root ../processed_data/gemini \
    --model gemini-3-flash-preview-thinking
```

### 2. Single-Source Processing (Small Batch)
```bash
# Process a single source with a sample limit
python process_mix4.py \
    --limit-per-source 100 \
    --output-root ./output_mix4 \
    --model gemini-3-flash-preview-thinking
```

### 3. Summarize Outputs
```bash
python summarize_outputs.py --output-root ../processed_data/gemini
```

### 4. Classification Accuracy Evaluation
```bash
python evaluate_pointarena.py --disable-thinking --samples-per-class 20
# Expected: ~91% 5-class classification accuracy
```

### 5. Full Pipeline Evaluation
```bash
python evaluate_pointarena_full_pipeline.py \
    --input ../eval_data/val_pointarena.jsonl \
    --output ./pipeline_eval_results
```

### 6. API Filtering (Cache-First)
```bash
# Cache-prioritized API filtering
python filter_with_api.py --input ../data/samples.jsonl --output ./filtered.jsonl
```

## Pipeline Stages

```
Input (JSONL, containing query + points + image_path)
   |
   |-- Stage 1: CLEAN (Gatekeeping)
   |    Determines if the sample is a valid pointing task, extracts core_target
   |    Output: {"keep": bool, "core_target": str}
   |
   |-- Stage 2: MULTIPOINT (Multi-Point Detection)
   |    Detects if multiple points are needed (all/every/several)
   |    Output: {"multi_point": bool}
   |
   |-- Stage 3: CLASSIFY (5-Class Classification)
   |    Categories: Affordance / Counting / Object Reference / Reasoning / Spatial Relation
   |    Output: {"category": str}
   |
   +-- Stage 4: REWRITE (Query Standardization)
        Standardizes to "Point to ..." format
        Output: "Point to the coffee cup."
```

## Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| API URL | `https://api.vectorengine.ai/v1/chat/completions` | OpenAI-compatible endpoint |
| Model | `gemini-3-flash-preview-thinking` | Default model |
| Temperature | 0 | Deterministic output |
| Thinking | disabled | thinkingBudget=0 |
| Max Retries | 6 | Exponential backoff |
| JSON Repair | Up to 2 attempts | In-session repair |

## Output Format

Each sample produces one JSON file:
```json
{
  "id": "pixmo_points::0001__abc123",
  "status": "completed" | "filtered_out",
  "stages": {
    "cleaning": {"keep": true, "core_target": "coffee cup"},
    "multipoint": {"multi_point": false},
    "classification": {"category": "Affordance"},
    "rewrite": {"final_query": "Point to the coffee cup."}
  },
  "session_messages": [...],
  "category": "Affordance"
}
```


## Docker Testing Notes

- Use --network host when running in Docker containers that need pip install (default bridge lacks DNS resolution).
- filter_with_api.py is self-contained and does not require an external scripts/paths.py module.
- All scripts support --help for CLI argument details.
- API key failures are expected when no key is set; the scripts handle this gracefully.


## Paper Data Statistics
- Total processed: 37,498 samples
- Trainable samples: 24,415
- 5-class distribution: Counting 13,547 > Object Reference 4,586 > Reasoning 2,715 > Affordance 1,912 > Spatial Relation 804