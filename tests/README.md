[🇨🇳 中文](README_CN.md)

# Tests

46 test files covering data processing, model architecture, training, and evaluation for the PointArena project.

---

## Overview

The test suite is organized into the following categories:

- **Environment & Resource Detection** -- verify system setup and dependencies
- **Data Processing** -- validate normalization, cleaning, deduplication, and schema for all input datasets
- **LLM Processing** -- test API prompting, parsing, filtering, and context length handling
- **Training** -- smoke tests, dataset loading, and training loop verification (requires GPU/PyTorch)
- **Architecture** -- AttnRes block injection, SinglePos coordinate encoding, FP8 mixed precision
- **Steerable Pipeline** -- task logic and geometric constraint validation
- **Evaluation** -- adapter loading, checkpoint ranking
- **Data Integrity** -- audit outputs, coordinate QA, clean artifact verification

---

## How to Run

### Prerequisites

```bash
# Required dependencies (some tests require GPU/PyTorch):
#   torch, transformers, peft, datasets, cv2, numpy, pandas,
#   PIL, pydantic, pyarrow, pytest, scipy, safetensors, einops

# Set the Python path (required):
export PYTHONPATH=$PWD/..:$PYTHONPATH

# Activate environment if using conda:
# conda activate pointarena
```

### Run All Tests

```bash
cd tests/
bash run_all_tests.sh
```

### Run Tests by Module

**Environment Checks (No GPU Required)**

```bash
python test_00_detect_resources.py
python test_00_dirs.py
python test_01_imports.py
```

**Data Processing Tests (Require src modules, no GPU)**

```bash
python test_01_schema.py
python test_02_normalize_pixmo.py
python test_03_normalize_robopoint.py
python test_04_normalize_where2place.py
python test_05_normalize_refspatial.py
python test_05_rule_filter.py
python test_06_build_unified_dataset.py
python test_06_mask_to_points.py
python test_07_dedup.py
python test_09_final_select.py
```

**LLM Processing Tests**

```bash
python test_04_extract_obj_free.py
python test_05_api_screen.py
python test_07_prompt_and_parse.py
python test_08_api_filter_contract.py
python test_08_context_length.py
```

**Training Tests (Require GPU/PyTorch)**

```bash
python test_09_train_smoke.py
python test_10_export_molmo2_format.py
python test_12_torch_dataset_smoke.py
python test_15_pointarena_rewritten_dataset.py
python test_19_steerable_d_training_dataset.py
```

**Architecture Tests (Require GPU/PyTorch)**

```bash
python test_20_molmo2_attnres.py        # AttnRes architecture verification
python test_21_loss_source_stats.py
python test_22_singlepos.py
python test_23_fp8_mixed.py
```

**Steerable Pipeline Tests**

```bash
python test_17_guide4_task_logic.py     # Guide4 task logic
```

**Evaluation Tests**

```bash
python test_10_eval_adapter.py
python test_11_checkpoint_ranking.py
```

**Data Integrity Tests**

```bash
python test_13_clean_audit_outputs.py
python test_14_coordinate_qa_dataset.py
python test_15_full_clean_artifacts.py
python test_16_query_coord_normalize.py
```

---

## Key Tests Table

| Test File                          | What It Tests                                                  | Dependencies                          |
|------------------------------------|---------------------------------------------------------------|---------------------------------------|
| `test_20_molmo2_attnres.py`        | AttnRes block injection, forward pass, gate initialization, gradient flow | GPU/PyTorch + PointArena project      |
| `test_22_singlepos.py`             | SinglePos coordinate encoding                                 | GPU/PyTorch + PointArena project (src) |
| `test_23_fp8_mixed.py`             | FP8 mixed precision training                                  | GPU/PyTorch + PointArena project (src) |
| `test_17_guide4_task_logic.py`     | Steerable task candidate generation, geometric constraint validation | No GPU, PointArena project (src)       |
| `test_19_steerable_d_training_dataset.py` | Steerable-D dataset loading, class balancing, anchor embedding | GPU/PyTorch + PointArena project (src) |
| `test_01_schema.py`                | Data schema validation                                        | PointArena project (src)               |
| `test_15_pointarena_rewritten_dataset.py` | Rewritten dataset loading, 5-category sampling          | GPU/PyTorch + PointArena project (src) |
| `test_09_train_smoke.py`           | Training smoke test (quick training loop verification)        | GPU/PyTorch + PointArena project (src) |
| `test_01_imports.py`               | Check base dependencies (cv2, datasets, numpy, etc.)          | No GPU                                |
| `test_00_detect_resources.py`      | Detect GPU/CPU resources                                      | No GPU                                |

---

## Supporting Files

| File                    | Description                                                   |
|-------------------------|---------------------------------------------------------------|
| `conftest.py`           | Pytest configuration, auto-adds project root to sys.path      |
| `env.sh`                | Environment variable setup script (sourced by run_all_tests.sh) |
| `run_all_tests.sh`      | Batch test runner                                             |

---

## Important Notes

- All tests that import `src.*` modules require the full PointArena project directory structure (including the `src/` directory)
- Tests without `src.*` imports can be run independently from the `submit_all/` directory
- Tests requiring PyTorch must be executed in a GPU-enabled environment
