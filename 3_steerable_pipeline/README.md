[🇨🇳 中文](README_CN.md)

# Steerable Data Pipeline (Innovation 1c)

A **deterministic** steerable data generator based on SAM3 mask verification. It uses 700 anchor relative-direction templates and requires **no LLM API calls** during generation.

## Overview

This pipeline produces high-quality steerable pointing data by:

- Extracting object inventories from raw annotations
- Generating candidate object pairs with geometric constraints
- Validating visibility via SAM3 segmentation masks
- Filling natural-language templates from a curated 700-template library
- Applying multi-layer validation (text scoring, rule checks, optional VLM review)

The result is a dataset where each sample asks the model to point from a blue anchor point toward a target object using relative-direction language (e.g., "to the right of", "above").

---

## Files to Download

```bash
# 1. PixMo-Points dataset (images + annotations):
hf download allenai/pixmo-points --repo-type dataset --local-dir ./raw_data/pixmo_points

# 2. SAM3 ONNX model (for mask generation and path validation):
#    Place manually at ./models/sam3.onnx
```

## Dependencies

Install the required Python packages before running the pipeline:

```bash
pip install torch transformers numpy Pillow opencv-python pyyaml onnxruntime onnx
```

> Note: `pip` must be installed on the server first (`sudo apt-get install python3-pip`).

---

## Scripts to Run

### 1. Build and Validate the Template Library

```bash
cd 3_steerable_pipeline/

# Generate 700 templates (7 groups x 100 each)
python build_templates.py --output template_library_raw.json

# Filter and validate templates (remove disallowed words, malformed entries, etc.)
python filter_generated_templates.py \
    --input template_library_raw.json \
    --output-json template_library.json \
    --report filter_report.json
```

### 2. Run the Steerable Data Generation Pipeline

```bash
python run_pipeline.py \
    --config config.yaml \
    --input-labels ../raw_data/pixmo_points \
    --output ../processed_data/steerable \
    --sam3-model ../models/sam3.onnx \
    --template-library template_library.json \
    --max-samples 10000
```

### 3. Build the Anchor Guidance Document

```bash
python build_anchor_guide.py --output anchor_guide.md
```

---

## Pipeline Flow

```
Raw Images + Annotations
   |
   +-- 1. Object Inventory (label cleaning, same-category grouping, noun extraction)
   |
   +-- 2. Candidate Pair Generation (geometry checks: min distance, direction range)
   |
   +-- 3. SAM3 Mask Generation (one segmentation mask per object)
   |    +-- Path Validation: 3-pixel-wide corridor from anchor to target
   |    +-- Anchor Exclusion: target mask dilated by 5px, ensure anchor is NOT on target
   |    +-- Direction Tolerance: sample anchors within +/-10 degrees of reference direction
   |
   +-- 4. Template Filling (700-template library placeholder substitution)
   |
   +-- 5. LLM Paraphrasing (optional, for naturalness improvement)
   |
   +-- 6. Multi-Layer Validation
   |    +-- Text Scoring (grammar, naturalness, specificity, relation fidelity)
   |    +-- Rule Checks (disallowed words, template rigidity detection)
   |    +-- VLM Review (visual inspection of high-risk samples)
   |
   +-- 7. Final Export (train_steerable.jsonl + rendered anchor images)
```

---

## 700 Template Library (7 Groups x 100)

| Template Group       | Count | Example                                                   |
|----------------------|-------|-----------------------------------------------------------|
| `straight_axis`      | 100   | "Point to the {label} {axis} the blue point."             |
| `single_diagonal`    | 100   | "From the blue point, move {diag} until you reach the {label}." |
| `two_step_axis`      | 100   | "Move {first} then {second} from the blue point to the {label}." |
| `nearest_among_objects` | 100 | "Point to the {label} nearest to the blue point."         |
| `other_object_axis`  | 100   | "From the {anchor}, point to the {label} to its {axis}."  |
| `other_object_diagonal` | 100 | "From the {anchor}, move {diag} to the {label}."          |
| `axis_constrained_nearest` | 100 | "Among the {label}s to the {axis}, point to the nearest." |

---

## Configuration (config.yaml)

```yaml
coord:
  assume_pct_0_100: true
  clip_soft_min: -1.0
  clip_soft_max: 101.0
  hard_drop_min: -1.0
  hard_drop_max: 101.0

pair:
  min_dx: 0.06                # Minimum horizontal separation
  min_dy: 0.06                # Minimum vertical separation
  min_dist: 0.03              # Minimum anchor-target distance
  overlap_drop_dist: 0.01
  uniqueness_margin: 0.03
  max_axis_offshoot: 0.25
  near_duplicate_dist: 0.02

filter:
  max_ambiguity_score: 0.35
  keep_qualities: ["high", "medium"]

render:
  point_radius_ratio: 0.015   # Blue point radius
  point_color: [0, 102, 255]
  point_outline_color: [255, 255, 255]
  target_color: [255, 64, 64]
  target_outline_color: [255, 255, 255]
  banner_height: 110

demo:
  target_pairs: 100
  top_images_to_scan: 1200
  max_relation_per_type: 20
```

---

## Output Format

`train_steerable.jsonl` schema:

```json
{
  "id": "steerable::<pair_id>",
  "image": "/rendered_anchor_images/xxx.jpg",
  "task_type": "pointing_steerable",
  "messages": [
    {"role": "user", "content": "<image>\nThe blue point marks an existing location. Point to the object to the right of it."},
    {"role": "assistant", "content": "<point x=\"0.268\" y=\"0.553\"/>"}
  ],
  "meta": {
    "relation_type": "right_of",
    "anchor_point_norm": [0.3, 0.5],
    "target_point_norm": [0.55, 0.5]
  }
}
```

The blue anchor point is rendered directly on the image (blue circle, radius = min(H,W) * 0.015, with a white outline).

---

## Relationship to AttnRes Training

Steerable data requires the model to maintain awareness of the anchor point while following directional instructions. AttnRes provides cross-block attention residual connections, enabling the model to revisit anchor context across deeper layers. The combination of steerable data and AttnRes improves steerability from 50.5% to **63.0%**.
