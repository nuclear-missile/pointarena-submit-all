#!/usr/bin/env python3
"""PointArena category validation pipeline with multi-threaded LLM inference."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

API_URL = "https://api.vectorengine.ai/v1/chat/completions"

FILE_TO_LABEL: Dict[str, str] = {
    "reasoning.txt": "Reasoning",
    "spatial_relation.txt": "Spatial Relation",
    "affordance.txt": "Affordance",
    "counting.txt": "Counting",
    "object_reference.txt": "Object Reference",
}

LABELS: List[str] = [
    "Reasoning",
    "Spatial Relation",
    "Affordance",
    "Counting",
    "Object Reference",
]

SYSTEM_PROMPT = """# Role and Objective
You are a highly precise Data Classifier for the Vision-Language Navigation dataset (Point Arena). Your task is to analyze an object-pointing instruction based on its underlying *semantic features and cognitive bottleneck*, classifying it into exactly ONE of FIVE predefined categories.

# CRITICAL FORMATTING RULES
You must output ONLY a valid, flat JSON object. 
- NO Markdown formatting (DO NOT use ```json or ```).
- NO conversational filler.
- Output exactly two keys: "thought" (under 20 words analyzing the semantic feature) and "category".

Example output:
{
  "thought": "Focuses on reading text/semantic info (hotel name) to pinpoint an exact UI/text element.",
  "category": "Object Reference"
}

# Semantic Category Definitions & Decision Boundaries
Do not rely on keywords. Evaluate the PRIMARY mental effort required to solve the task.

## 1. Object Reference (UI, Text, and Fine-Grained Pinpointing)
- **Semantic Feature:** The task acts like a precision crosshair. It targets purely digital interfaces, exact text strings, OR ultra-fine-grained extreme points of an object. It also includes relative navigation from a pre-existing cursor.
- **Key Characteristics:**
  - Reading text/labels (e.g., "the name of the hotel", "the input bar").
  - Extreme fine-grained specific points (e.g., "the tip of the shoes", "directly onto the blue lens").
  - Cursor-based relative movement (e.g., "existing point", "moving downward until...").

## 2. Counting (Enumeration and Collections)
- **Semantic Feature:** The task requires scanning the global scene to identify a *collection*, a mass of items, or a specific subset based on exact quantity. 
- **Key Characteristics:** 
  - Naked plural nouns acting as the main target without strict spatial anchors (e.g., "the statues", "Jeep vehicles").
  - Mass nouns describing a grouped state (e.g., "the sliced fruit").
  - Explicit counting ("two yellow hats", "all the birds").
- *Exception:* If the plural objects are strictly located by a geometric anchor (e.g., "objects *above the sofa*"), the primary task is spatial, so it belongs to Spatial Relation.

## 3. Spatial Relation (Geometric Layout and Distance)
- **Semantic Feature:** The primary cognitive bottleneck is 2D mapping, absolute/relative positioning, or distance comparison. It requires no complex physics or functional deduction.
- **Key Characteristics:**
  - Distance comparisons (e.g., "the nearest boat", "closest table").
  - Topological anchoring (e.g., "above the sofa", "between X and Y", "front wheels of").
  - *Exception:* If the target relies on a part-whole ownership/semantic binding (e.g., "the shoe of the cyclist"), it requires semantic reasoning, NOT pure spatial layout.

## 4. Affordance (Generic Human Tools and Instruments)
- **Semantic Feature:** Extremely strict definition. It applies ONLY to generic, manipulable tools, instruments, or standard devices defined purely by human hand-scale operations (eating, writing, hammering, playing music).
- **Key Characteristics:**
  - Generic objects named by their utility (e.g., "tool that creates music", "tool made of wood", "object used to eat").
  - *Boundary Rule:* If the object is a structural sub-component of a vehicle/room (e.g., "illuminate the car", "carry the baby", "where items are placed on a bike"), it requires scene-level deduction and belongs to Reasoning!

## 5. Reasoning (World Knowledge, States, and Complex Deduction)
- **Semantic Feature:** The catch-all for complex semantics. The target is isolated through physical states, part-whole ownership, causal events, or scene-level structural functionality.
- **Key Characteristics:**
  - Action/Physical states (e.g., "moving", "fastest", "direction in which...").
  - Scene-level functional deduction (e.g., "where people sit", "what illuminates the car", "object used to carry the baby").
  - Part-whole ownership linking specific actors (e.g., "the shoe of the cyclist", "part of the train").
  - Indirect logic/World knowledge (e.g., "caused it to fall", "color of grass").

---
# Input Task:
[INSERT_TASK_HERE]"""

thread_local = threading.local()


@dataclass
class PredictionResult:
    text: str
    ground_truth: str
    prediction: str
    analysis: str
    model_output: str
    status: str


class RequestLogger:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.path.write_text("", encoding="utf-8")

    def log(self, record: Dict) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self.lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PointArena classification evaluation")
    parser.add_argument("--api-key", default=os.getenv("VECTORENGINE_API_KEY"), help="API key")
    parser.add_argument(
        "--model",
        default="gemini-3-flash-preview-thinking",
        help="Model name",
    )
    parser.add_argument("--samples-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--connect-timeout", type=int, default=15)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--disable-thinking", action="store_true", help="Disable model thinking / thoughts in API requests")
    parser.add_argument("--output", default="out.md")
    parser.add_argument("--requests-log", default="requests_log.jsonl")
    parser.add_argument("--fail-output", default="fail.txt")
    return parser.parse_args()


def get_session() -> requests.Session:
    session = getattr(thread_local, "session", None)
    if session is None:
        session = requests.Session()
        thread_local.session = session
    return session


def normalize_category(category: str | None) -> str:
    if category is None:
        return "None"

    key = str(category).strip().lower()
    key = key.replace("_", " ").replace("-", " ")
    key = re.sub(r"\s+", " ", key)

    alias = {
        "reasoning": "Reasoning",
        "spatial relation": "Spatial Relation",
        "spatial": "Spatial Relation",
        "affordance": "Affordance",
        "counting": "Counting",
        "object reference": "Object Reference",
        "objectref": "Object Reference",
        "reference": "Object Reference",
        "none": "None",
    }

    if key in alias:
        return alias[key]

    for label in LABELS:
        if key == label.lower():
            return label

    return "None"


def strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def parse_model_output(content: str) -> Tuple[str, str]:
    cleaned = strip_code_fence(content)

    def try_parse_json(txt: str) -> Tuple[str, str] | None:
        try:
            obj = json.loads(txt)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict):
            return None
        analysis = str(obj.get("analysis") or obj.get("thought") or "").strip()
        category = normalize_category(obj.get("category"))
        return analysis, category

    parsed = try_parse_json(cleaned)
    if parsed:
        return parsed

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        parsed = try_parse_json(match.group(0))
        if parsed:
            return parsed

    cat_match = re.search(
        r"Reasoning|Spatial\s*Relation|Affordance|Counting|Object\s*Reference|None",
        cleaned,
        flags=re.IGNORECASE,
    )
    category = normalize_category(cat_match.group(0) if cat_match else None)
    analysis = cleaned.strip()
    return analysis, category


def predict_category(
    sample_id: int,
    text: str,
    api_key: str,
    model: str,
    connect_timeout: int,
    timeout: int,
    max_retries: int,
    disable_thinking: bool,
    request_logger: RequestLogger,
) -> Tuple[str, str, str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "temperature": 0,
        "stream": False,
        "max_tokens": 256,
        "response_format": {"type": "json_object"},
    }
    if disable_thinking:
        payload["extra_body"] = {
            "google": {
                "thinking_config": {
                    "include_thoughts": False,
                    "thinkingBudget": 0,
                }
            }
        }

    last_error = ""
    for attempt in range(max_retries):
        started = time.time()
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "sample_id": sample_id,
            "attempt": attempt + 1,
            "model": model,
            "request": payload,
        }
        try:
            session = get_session()
            resp = session.post(API_URL, headers=headers, json=payload, timeout=(connect_timeout, timeout))
            elapsed = round(time.time() - started, 3)
            record.update(
                {
                    "http_status": resp.status_code,
                    "elapsed_sec": elapsed,
                    "response_text": resp.text,
                }
            )
            request_logger.log(record)

            if resp.status_code in {429, 500, 502, 503, 504}:
                last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                sleep_s = min(8.0, (1.5 ** attempt) + random.random())
                time.sleep(sleep_s)
                continue

            if resp.status_code >= 400:
                last_error = f"HTTP {resp.status_code}: {resp.text[:500]}"
                return "None", last_error, "http_error", resp.text

            data = resp.json()
            message = data.get("choices", [{}])[0].get("message", {})
            content = str(message.get("content", ""))
            reasoning_content = str(message.get("reasoning_content", ""))

            analysis, category = parse_model_output(content)
            if not analysis and reasoning_content:
                analysis = reasoning_content.strip()

            return category, analysis, "ok", content

        except requests.RequestException as exc:
            elapsed = round(time.time() - started, 3)
            err = f"RequestException: {exc}"
            last_error = err
            record.update(
                {
                    "http_status": None,
                    "elapsed_sec": elapsed,
                    "error": err,
                }
            )
            request_logger.log(record)
            sleep_s = min(8.0, (1.5 ** attempt) + random.random())
            time.sleep(sleep_s)
        except Exception as exc:
            elapsed = round(time.time() - started, 3)
            err = f"Exception: {exc}"
            last_error = err
            record.update(
                {
                    "http_status": None,
                    "elapsed_sec": elapsed,
                    "error": err,
                }
            )
            request_logger.log(record)
            break

    return "None", last_error or "unknown_error", "failed", ""


def load_dataset(samples_per_class: int, seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    dataset: List[Dict[str, str]] = []

    for file_name, label in FILE_TO_LABEL.items():
        path = Path(file_name)
        if not path.exists():
            raise FileNotFoundError(f"Missing data file: {path}")

        queries = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(queries) < samples_per_class:
            raise ValueError(
                f"File {file_name} has only {len(queries)} lines, less than {samples_per_class}"
            )

        sampled = rng.sample(queries, samples_per_class)
        for text in sampled:
            dataset.append({"text": text, "ground_truth": label})

    rng.shuffle(dataset)
    return dataset


def markdown_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def write_fail_cases(results: List[PredictionResult], fail_output: Path) -> int:
    fails = [r for r in results if r.prediction != r.ground_truth or r.status != "ok"]

    with fail_output.open("w", encoding="utf-8") as f:
        for item in fails:
            output_and_analysis = f"model_output: {one_line(item.model_output)} || analysis: {one_line(item.analysis)}"
            line = (
                f"{output_and_analysis}\t"
                f"task: {one_line(item.text)}\t"
                f"ground_truth: {item.ground_truth}\t"
                f"prediction: {item.prediction}\t"
                f"status: {item.status}"
            )
            f.write(line)
            f.write("\n")

    return len(fails)


def generate_report(
    results: List[PredictionResult],
    model: str,
    workers: int,
    samples_per_class: int,
    seed: int,
    disable_thinking: bool,
    output_path: Path,
    requests_log_path: Path,
    fail_output_path: Path,
) -> None:
    y_true = [r.ground_truth for r in results]
    y_pred = [r.prediction for r in results]

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    macro_precision = report["macro avg"]["precision"]
    macro_recall = report["macro avg"]["recall"]
    macro_f1 = report["macro avg"]["f1-score"]

    cm_labels = LABELS.copy()
    if any(pred == "None" for pred in y_pred):
        cm_labels.append("None")
    cm = confusion_matrix(y_true, y_pred, labels=cm_labels)

    bad_cases = [r for r in results if r.prediction != r.ground_truth][:5]

    lines: List[str] = []
    lines.append("# Point Arena 分类评测报告")
    lines.append("")
    lines.append("## 1. 测试概览")
    lines.append(f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("- 测试文件: reasoning.txt, spatial_relation.txt, affordance.txt, counting.txt, object_reference.txt")
    lines.append(f"- 每类采样数量: {samples_per_class}")
    lines.append(f"- 总测试数量: {len(results)}")
    lines.append(f"- 随机种子: {seed}")
    lines.append(f"- 使用模型: {model}")
    lines.append(f"- 关闭思考: {'yes' if disable_thinking else 'no'}")
    lines.append(f"- API Endpoint: {API_URL}")
    lines.append(f"- 并发线程数: {workers}")
    lines.append(f"- 请求日志文件: {requests_log_path.name}")
    lines.append(f"- 失败案例文件: {fail_output_path.name}")
    lines.append("")

    lines.append("## 2. 整体评测指标")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Overall Accuracy | {accuracy:.4f} |")
    lines.append(f"| Macro Precision | {macro_precision:.4f} |")
    lines.append(f"| Macro Recall | {macro_recall:.4f} |")
    lines.append(f"| Macro F1 | {macro_f1:.4f} |")
    lines.append("")

    lines.append("## 3. 各类别详细表现")
    lines.append("| Category | Precision | Recall | F1-Score | Support |")
    lines.append("|---|---:|---:|---:|---:|")
    for label in LABELS:
        row = report[label]
        lines.append(
            f"| {label} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1-score']:.4f} | {int(row['support'])} |"
        )
    lines.append("")

    lines.append("## 4. 混淆矩阵")
    header = "| True \\ Pred | " + " | ".join(cm_labels) + " |"
    sep = "|---|" + "|".join(["---:" for _ in cm_labels]) + "|"
    lines.append(header)
    lines.append(sep)
    for i, true_label in enumerate(cm_labels):
        if true_label not in LABELS:
            continue
        row_vals = [str(int(v)) for v in cm[i]]
        lines.append(f"| {true_label} | " + " | ".join(row_vals) + " |")
    lines.append("")

    lines.append("## 5. Bad Case 分析")
    if not bad_cases:
        lines.append("本次采样中无错判样本。")
    else:
        lines.append("| Text | Ground Truth | Prediction | Analysis |")
        lines.append("|---|---|---|---|")
        for item in bad_cases:
            lines.append(
                "| "
                + markdown_escape(item.text)
                + " | "
                + item.ground_truth
                + " | "
                + item.prediction
                + " | "
                + markdown_escape(item.analysis[:3000])
                + " |"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    if not args.api_key:
        raise ValueError("Missing API key. Set --api-key or VECTORENGINE_API_KEY")

    print("[Step 1] 数据加载与采样开始...")
    dataset = load_dataset(args.samples_per_class, args.seed)
    print(f"[Step 1] 完成。采样得到 {len(dataset)} 条数据。")

    requests_log_path = Path(args.requests_log)
    fail_output_path = Path(args.fail_output)
    request_logger = RequestLogger(requests_log_path)
    print(f"[Step 2] 请求日志将保存到: {requests_log_path}")

    print("[Step 3] 开始执行并发预测...")
    results: List[PredictionResult] = [None] * len(dataset)  # type: ignore

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_idx = {
            executor.submit(
                predict_category,
                idx,
                item["text"],
                args.api_key,
                args.model,
                args.connect_timeout,
                args.timeout,
                args.max_retries,
                args.disable_thinking,
                request_logger,
            ): idx
            for idx, item in enumerate(dataset)
        }

        with tqdm(total=len(dataset), desc="[Step 3] Predicting", ncols=100) as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                gt = dataset[idx]["ground_truth"]
                txt = dataset[idx]["text"]
                try:
                    pred, analysis, status, model_output = future.result()
                except Exception as exc:
                    pred, analysis, status, model_output = "None", f"FutureException: {exc}", "failed", ""

                results[idx] = PredictionResult(
                    text=txt,
                    ground_truth=gt,
                    prediction=normalize_category(pred),
                    analysis=analysis,
                    model_output=model_output,
                    status=status,
                )
                pbar.update(1)

    failed_requests = sum(1 for r in results if r.status != "ok")
    print(f"[Step 3] 预测完成。失败/异常条数: {failed_requests}")

    print("[Step 4] 开始计算评测指标...")
    _ = classification_report(
        [r.ground_truth for r in results],
        [r.prediction for r in results],
        labels=LABELS,
        zero_division=0,
    )
    print("[Step 4] 指标计算完成。")

    print("[Step 5] 生成 out.md 与 fail.txt ...")
    fail_count = write_fail_cases(results, fail_output_path)
    generate_report(
        results=results,
        model=args.model,
        workers=args.workers,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
        disable_thinking=args.disable_thinking,
        output_path=Path(args.output),
        requests_log_path=requests_log_path,
        fail_output_path=fail_output_path,
    )
    print(f"[Step 5] 报告已生成: {args.output}")
    print(f"[Step 5] 失败案例已生成: {fail_output_path} (共 {fail_count} 条)")


if __name__ == "__main__":
    main()
