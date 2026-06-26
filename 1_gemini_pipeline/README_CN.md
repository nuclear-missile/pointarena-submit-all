[🇬🇧 English](README.md)

# Gemini API Data Pipeline (创新点1a)

使用Gemini Flash API (via vectorengine.ai) 进行 **4阶段数据清洗-分类-改写**。

> **注意**: 需要 Python 3.10+。安装依赖: `pip install -r requirements.txt`

## 要下载的文件

```bash
# 1. 原始PointArena训练数据集:
hf download allenai/pixmo-points --repo-type dataset --local-dir ./raw_data/pixmo_points
hf download wentao-yuan/robopoint-data --repo-type dataset --local-dir ./raw_data/robopoint
hf download JingkunAn/RefSpatial --repo-type dataset --local-dir ./raw_data/refspatial
hf download FlagEval/Where2Place --repo-type dataset --local-dir ./raw_data/where2place

# 2. API Key: 从 https://api.vectorengine.ai 获取
```

## 要跑的脚本

### 1. 数据处理管线 (核心)
```bash
cd 1_gemini_pipeline/
export VECTORENGINE_API_KEY="your-api-key"

# 运行4阶段管线
python run_streaming.py \
    --sources pixmo_points,robopoint,refspatial,where2place \
    --output-root ../processed_data/gemini \
    --model gemini-3-flash-preview-thinking
```

### 2. 单独处理 (小批量)
```bash
# 单源处理
python process_mix4.py \
    --limit-per-source 100 \
    --output-root ./output_mix4 \
    --model gemini-3-flash-preview-thinking
```

### 3. 汇总输出
```bash
python summarize_outputs.py --output-root ../processed_data/gemini
```

### 4. 分类准确率评估
```bash
python evaluate_pointarena.py --disable-thinking --samples-per-class 20
# 预期: ~91% 5分类准确率
```

### 5. 全管线评估
```bash
python evaluate_pointarena_full_pipeline.py \
    --input ../eval_data/val_pointarena.jsonl \
    --output ./pipeline_eval_results
```

### 6. API过滤 (缓存优先)
```bash
# 使用缓存优先的API过滤
python filter_with_api.py --input ../data/samples.jsonl --output ./filtered.jsonl
```

## 管线阶段

```
输入 (JSONL, 含query+points+image_path)
   |
   |-- 阶段1: CLEAN (Gatekeeping)
   |   判断是否有效指向任务, 提取core_target
   |   输出: {"keep": bool, "core_target": str}
   |
   |-- 阶段2: MULTIPOINT (多点检测)
   |   检测是否需要多个点 (all/every/several)
   |   输出: {"multi_point": bool}
   |
   |-- 阶段3: CLASSIFY (5分类)
   |   分类: Affordance / Counting / Object Reference / Reasoning / Spatial Relation
   |   输出: {"category": str}
   |
   +-- 阶段4: REWRITE (改写)
       标准化为 "Point to ..." 格式
       输出: "Point to the coffee cup."
```

## 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| API URL | `https://api.vectorengine.ai/v1/chat/completions` | OpenAI兼容端点 |
| 模型 | `gemini-3-flash-preview-thinking` | 默认模型 |
| Temperature | 0 | 确定性输出 |
| Thinking | disabled | thinkingBudget=0 |
| 最大重试 | 6次 | 指数退避 |
| JSON修复 | 最多2次 | 会话内修复 |

## 输出格式

每个样本一个JSON文件:
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


## Docker 测试说明

- Docker 容器中 pip install 需要 --network host 标志（默认桥接网络无 DNS 解析）。
- filter_with_api.py 为独立脚本，无需外部 scripts/paths.py 模块。
- 所有脚本均支持 --help 查看命令行参数说明。
- 未设置 API key 时的报错为预期行为，脚本会优雅处理。


## 论文数据规模
- 处理: 37,498条
- 可训练: 24,415条
- 5类别分布: Counting 13,547 > Object Reference 4,586 > Reasoning 2,715 > Affordance 1,912 > Spatial Relation 804