[🇬🇧 English](README.md)

# Tests

46个测试文件覆盖数据处理、模型架构、训练和评估。

## 运行环境要求

```bash
# 需要安装以下依赖（部分测试需要GPU/PyTorch环境）:
#   torch, transformers, peft, datasets, cv2, numpy, pandas,
#   PIL, pydantic, pyarrow, pytest, scipy, safetensors, einops

# 设置Python路径（必需）:
export PYTHONPATH=$PWD/..:$PYTHONPATH

# 激活环境（如有conda）:
# conda activate pointarena
```

## 运行测试

```bash
cd tests/

# 运行全部测试:
bash run_all_tests.sh

# 或按模块运行:
# 环境检查（无需GPU）
python test_00_detect_resources.py
python test_00_dirs.py
python test_01_imports.py

# 数据处理测试（需要src模块，无需GPU）
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

# LLM处理测试
python test_04_extract_obj_free.py
python test_05_api_screen.py
python test_07_prompt_and_parse.py
python test_08_api_filter_contract.py
python test_08_context_length.py

# 训练测试（需要GPU/PyTorch）
python test_09_train_smoke.py
python test_10_export_molmo2_format.py
python test_12_torch_dataset_smoke.py
python test_15_pointarena_rewritten_dataset.py
python test_19_steerable_d_training_dataset.py

# 架构测试（需要GPU/PyTorch）
python test_20_molmo2_attnres.py        # AttnRes架构验证
python test_21_loss_source_stats.py
python test_22_singlepos.py
python test_23_fp8_mixed.py

# Steerable管线测试
python test_17_guide4_task_logic.py     # Guide4任务逻辑

# 评估测试
python test_10_eval_adapter.py
python test_11_checkpoint_ranking.py

# 数据完整性测试
python test_13_clean_audit_outputs.py
python test_14_coordinate_qa_dataset.py
python test_15_full_clean_artifacts.py
python test_16_query_coord_normalize.py
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `conftest.py` | pytest配置，自动添加项目根目录到sys.path |
| `env.sh` | 环境变量设置脚本（被run_all_tests.sh引用） |
| `run_all_tests.sh` | 批量运行测试脚本 |

## 关键测试说明

| 测试文件 | 测试内容 | 依赖 |
|---------|---------|------|
| `test_20_molmo2_attnres.py` | AttnRes块注入、前向传播、gate初始化、梯度流 | GPU/PyTorch + PointArena项目 |
| `test_22_singlepos.py` | SinglePos坐标编码 | GPU/PyTorch + PointArena项目(src) |
| `test_23_fp8_mixed.py` | FP8混合精度训练 | GPU/PyTorch + PointArena项目(src) |
| `test_17_guide4_task_logic.py` | Steerable任务候选生成、几何约束验证 | 无需GPU，需PointArena项目(src) |
| `test_19_steerable_d_training_dataset.py` | Steerable-D数据集加载、类别平衡、锚点嵌入 | GPU/PyTorch + PointArena项目(src) |
| `test_01_schema.py` | 数据模式验证 | PointArena项目(src) |
| `test_15_pointarena_rewritten_dataset.py` | 改写数据集加载、5类别采样 | GPU/PyTorch + PointArena项目(src) |
| `test_09_train_smoke.py` | 训练冒烟测试 (快速验证训练循环) | GPU/PyTorch + PointArena项目(src) |
| `test_01_imports.py` | 检查基础依赖（cv2, datasets, numpy等） | 无需GPU |
| `test_00_detect_resources.py` | 检测GPU/CPU资源 | 无需GPU |

### 注意
- 所有导入 `src.*` 的测试需要完整的 PointArena 项目目录结构（含 `src/` 目录）
- 不含 `src.*` 导入的测试可在 `submit_all/` 目录下独立运行
- 需要 PyTorch 的测试需在有 GPU 的环境中运行