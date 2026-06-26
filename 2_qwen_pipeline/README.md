# Qwen3-8B Local Data Pipeline (创新点1b)

## 环境要求

运行前请确保以下Python依赖已安装:



如果系统没有 ，先安装:
Defaulting to user installation because normal site-packages is not writeable
Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
Collecting pip
  Using cached pip-26.1.2-py3-none-any.whl (1.8 MB)
Installing collected packages: pip
  Attempting uninstall: pip
    Found existing installation: pip 26.1.2
    Uninstalling pip-26.1.2:
      Successfully uninstalled pip-26.1.2
Successfully installed pip-26.1.2

**注意:** 本目录的脚本依赖  模块，该模块位于项目根目录 。运行时请确保该目录存在于Python搜索路径中（脚本已自动添加上级目录至路径）。


使用本地Qwen3-8B via vLLM进行4阶段数据处理。**免费，无API费用**。

## 环境要求

运行前请确保以下Python依赖已安装:

```bash
pip install openai>=1.0.0 tqdm>=4.60.0 requests>=2.28.0 Pillow>=9.0.0 numpy>=1.21.0
```

如果系统没有 `pip`，先安装:
```bash
curl -sS https://bootstrap.pypa.io/get-pip.py | python3
```

**注意:** 本目录的脚本依赖 `clean_3types_local` 模块，该模块位于项目根目录 `~/PointArena/clean_3types_local/`。运行时请确保该目录存在于Python搜索路径中（脚本已自动添加上级目录至路径）。

## 要下载的文件

```bash
# 1. Qwen3-8B模型:
hf download Qwen/Qwen3-8B --local-dir ./models/Qwen3-8B

# 2. 原始训练数据集 (如果尚未下载):
hf download allenai/pixmo-points --repo-type dataset --local-dir ./raw_data/pixmo_points
hf download wentao-yuan/robopoint-data --repo-type dataset --local-dir ./raw_data/robopoint
hf download JingkunAn/RefSpatial --repo-type dataset --local-dir ./raw_data/refspatial
```

## 要跑的脚本

### 1. 启动Qwen3-8B vLLM服务
```bash
cd 2_qwen_pipeline/
bash launch_qwen3_8b_vllm.sh
# 验证服务:
python healthcheck_vllm.py
# 预期输出: "Qwen3-8B server healthy at http://127.0.0.1:8018"
```

### 2. 3类型管线 (Affordance / Object Reference / Reasoning)
```bash
python run_streaming_local.py \
    --input-dirs ../raw_data/pixmo_points ../raw_data/robopoint \
                  ../raw_data/refspatial \
    --output-root ../processed_data/qwen_3types \
    --target-categories Affordance "Object Reference" Reasoning \
    --target-per-category 2000 \
    --num-workers 24
```

### 3. Counting管线
```bash
# 方式A: 流式管线 (需要Qwen LLM)
python run_streaming_local.py \
    --input-dirs ../raw_data/pixmo_points ../raw_data/robopoint \
    --output-root ../processed_data/qwen_counting \
    --target-categories Counting \
    --target-count 20000 \
    --counting-mode \
    --num-workers 24

# 方式B: 直接构建 (不需要LLM, 纯规则)
python build_counting_direct.py \
    --input ../raw_data \
    --output ../processed_data/counting_direct.jsonl
```

### 4. 停止vLLM
```bash
bash stop_qwen3_8b_vllm.sh
```

## 管线阶段 (与Gemini相同)

```
输入 (JSONL, 含query+points+image_path)
   │
   ├─ 阶段1: CLEAN → 判断有效性, 提取core_target
   ├─ 阶段2: MULTIPOINT → 检测多点和计数类任务
   ├─ 阶段3: CLASSIFY → 3分类 (或强制Counting)
   └─ 阶段4: REWRITE → "Point to ..." 格式, 含JSON回退重试
```

### 后处理: 规则引擎 (`three_type_rules.py`)
在Qwen分类基础上应用基于模式的规则纠正误分类:
- "book about" → 强制归类为Reasoning
- "tool for" → 强制归类为Affordance
- "current point" / "existing point" → 强制归类为Object Reference

## 配置

| 参数 | 值 |
|------|-----|
| 模型 | Qwen3-8B |
| vLLM URL | `http://127.0.0.1:8018/v1` |
| Temperature | 0 (确定性) |
| 最大上下文 | 8192 tokens |
| Thinking | disabled |
| 并行workers | 24 |
| 单样本耗时 | ~1秒 |

## 与Gemini管线的对比

| 方面 | Qwen管线 | Gemini管线 |
|------|---------|-----------|
| 模型 | Qwen3-8B (本地) | Gemini Flash (API) |
| 成本 | 免费 | API调用费用 |
| 分类数 | 3类 + Counting | 5类 |
| 输入 | 纯文本 | 纯文本 |
| 后处理 | 规则引擎 | 无 |
| 速度 | ~1秒/样本 | ~2-3秒/样本 |

## 输出格式 (与Gemini管线一致)

每个样本一个JSON文件，含完整的4阶段审计追踪:
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

## 产出数据位置
- 3类型: `clean_3types_local/qwen3_8b_three_types_2000_each_nodup_reverse/`
- 3类型追加: `clean_3types_local/qwen3_8b_three_types_add2000_nodup_reverse/`
- Counting: `clean_counting_local/` (对应输出目录)
