[🇬🇧 English](README.md)

# Steerable Data Pipeline (创新点1c)

基于SAM3掩码验证的**确定性**Steerable数据生成器。700个锚点相对方向模板，无需LLM API。

## 要下载的文件

```bash
# 1. PixMo-Points数据集 (含图像标注):
hf download allenai/pixmo-points --repo-type dataset --local-dir ./raw_data/pixmo_points

# 2. SAM3 ONNX模型 (用于掩码生成和路径验证):
#    手动放置到 ./models/sam3.onnx
```

## 环境依赖

运行管线需要安装以下Python包:



注意: `pip` 需要在服务器上预先安装 (`sudo apt-get install python3-pip`).

## 要跑的脚本

### 1. 构建和验证模板库
```bash
cd 3_steerable_pipeline/

# 生成700模板 (7组 × 100)
python build_templates.py --output template_library_raw.json

# 过滤验证模板 (去除禁用词、格式错误等)
python filter_generated_templates.py \
    --input template_library_raw.json \
    --output-json template_library.json --report filter_report.json
```

### 2. 运行Steerable数据生成管线
```bash
python run_pipeline.py \
    --config config.yaml \
    --input-labels ../raw_data/pixmo_points \
    --output ../processed_data/steerable \
    --sam3-model ../models/sam3.onnx \
    --template-library template_library.json \
    --max-samples 10000
```

### 3. 构建锚点问题指南
```bash
python build_anchor_guide.py --output anchor_guide.md
```

## 管线流程

```
原始图像 + 标注
   │
   ├─ 1. 物体清单 (label清洗, 同类别分组, 名词提取)
   │
   ├─ 2. 候选对生成 (几何检查: 最小距离, 方向范围)
   │
   ├─ 3. SAM3掩码生成 (每物体一个分割掩码)
   │    ├─ 路径验证: 从锚点到目标的3像素带宽路径
   │    ├─ 锚点排除: 目标掩码膨胀5像素, 确保锚点不在目标上
   │    └─ 方向容差: 基准方向±10度的锚点采样
   │
   ├─ 4. 模板填充 (700模板库占位符替换)
   │
   ├─ 5. LLM改写 (可选, 用于自然度提升)
   │
   ├─ 6. 多层验证
   │    ├─ 文本评分 (语法/自然度/特异性/关系忠实度)
   │    ├─ 规则检查 (禁用词、模板僵硬检测)
   │    └─ VLM审核 (高风险样本可视化审查)
   │
   └─ 7. 最终导出 (train_steerable.jsonl + 渲染锚点图像)
```

## 700模板库 (7组 × 100)

| 模板组 | 数量 | 示例 |
|--------|------|------|
| `straight_axis` | 100 | "Point to the {label} {axis} the blue point." |
| `single_diagonal` | 100 | "From the blue point, move {diag} until you reach the {label}." |
| `two_step_axis` | 100 | "Move {first} then {second} from the blue point to the {label}." |
| `nearest_among_objects` | 100 | "Point to the {label} nearest to the blue point." |
| `other_object_axis` | 100 | "From the {anchor}, point to the {label} to its {axis}." |
| `other_object_diagonal` | 100 | "From the {anchor}, move {diag} to the {label}." |
| `axis_constrained_nearest` | 100 | "Among the {label}s to the {axis}, point to the nearest." |

## 配置 (config.yaml)

```yaml
coord:
  assume_pct_0_100: true
  clip_soft_min: -1.0 / clip_soft_max: 101.0

pair:
  min_dx: 0.06          # 最小水平间隔
  min_dy: 0.06          # 最小垂直间隔
  min_dist: 0.03        # 最小锚点-目标距离
  overlap_drop_dist: 0.01
  uniqueness_margin: 0.03
  max_axis_offshoot: 0.25

filter:
  max_ambiguity_score: 0.35
  keep_qualities: ["high", "medium"]

render:
  point_radius_ratio: 0.015   # 蓝点半径
  point_color: [0, 102, 255]
```

## 输出格式

`train_steerable.jsonl`:
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

蓝色锚点直接渲染在图像上 (蓝色圆, 半径=min(H,W)*0.015, 白色轮廓)。

## 与AttnRes训练的关系

Steerable数据要求模型在跟随方向指令时保持锚点状态。
AttnRes提供跨块注意力残差机制，使模型能够在更深层中回顾锚点上下文。
两者结合将Steerability从50.5%提升至63.0%。