#!/usr/bin/env python3
"""Run the full same-session cleaning->multipoint->classification/rewrite pipeline on PointArena val and report step-1 recall."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import process_mix4 as pipeline  # noqa: E402


TASKTYPE_TO_LABEL: Dict[str, str] = {
    "reasoning": "Reasoning",
    "spatial_relation": "Spatial Relation",
    "affordance": "Affordance",
    "counting": "Counting",
    "object_reference": "Object Reference",
}
LABELS: List[str] = [
    "Reasoning",
    "Spatial Relation",
    "Affordance",
    "Counting",
    "Object Reference",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the full PointArena pipeline with emphasis on step-1 cleaning recall.")
    parser.add_argument("--api-key", default=os.getenv("VECTORENGINE_API_KEY"), help="VectorEngine API key.")
    parser.add_argument("--model", default=pipeline.DEFAULT_MODEL, help="Model name. Defaults to current evaluate_pointarena.py setting.")
    parser.add_argument("--input", default=str(PROJECT_ROOT / "data" / "unified" / "val_pointarena.jsonl"), help="PointArena val jsonl path.")
    parser.add_argument("--workers", type=int, default=24, help="Parallel workers across samples.")
    parser.add_argument("--timeout", type=int, default=45, help="Per-request read timeout in seconds.")
    parser.add_argument("--connect-timeout", type=int, default=10, help="Per-request connection timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=3, help="Network retry count per request.")
    parser.add_argument("--parse-retries", type=int, default=2, help="How many same-session JSON repair turns to allow per JSON stage.")
    parser.add_argument("--sample-retries", type=int, default=2, help="How many full-sample retries to allow for invalid or truncated outputs.")
    parser.add_argument("--rewrite-retries", type=int, default=3, help="How many same-session rewrite repair turns to allow.")
    parser.add_argument("--output-root", default="pointarena_full_clean_multipoint_counting_same_session", help="Output root directory under pointarena_extract.")
    parser.add_argument("--requests-log", default="requests_log.jsonl", help="Global request log filename under output root.")
    parser.add_argument("--summary-file", default="summary.json", help="Summary filename under output root.")
    parser.add_argument("--report-file", default="report.md", help="Markdown report filename under output root.")
    parser.add_argument("--overwrite", action="store_true", help="Re-run even if a per-sample output file already exists.")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional total cap for debugging; 0 means all samples.")
    return parser.parse_args()


def select_jobs(input_path: Path, max_samples: int = 0) -> List[pipeline.SampleJob]:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing PointArena input file: {input_path}")

    jobs: List[pipeline.SampleJob] = []
    global_rank = 0
    per_source_rank: Dict[str, int] = defaultdict(int)

    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if max_samples and global_rank >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            task_type = str(record.get("task_type") or record.get("label") or "").strip()
            if task_type not in TASKTYPE_TO_LABEL:
                raise ValueError(f"Unknown task_type at line {line_no}: {task_type!r}")
            per_source_rank[task_type] += 1
            global_rank += 1
            record["ground_truth_label"] = TASKTYPE_TO_LABEL[task_type]
            jobs.append(
                pipeline.SampleJob(
                    source=task_type,
                    source_rank=per_source_rank[task_type],
                    global_rank=global_rank,
                    source_file=str(input_path),
                    record=record,
                )
            )
    return jobs


def write_run_config(output_root: Path, args: argparse.Namespace, jobs: List[pipeline.SampleJob]) -> None:
    config = {
        "generated_at": utc_now(),
        "script": str(Path(__file__).resolve()),
        "pipeline_script": str((SCRIPT_DIR / "process_mix4.py").resolve()),
        "input": str(Path(args.input).resolve()),
        "model": args.model,
        "api_url": pipeline.API_URL,
        "workers": args.workers,
        "timeout": args.timeout,
        "connect_timeout": args.connect_timeout,
        "max_retries": args.max_retries,
        "parse_retries": args.parse_retries,
        "sample_retries": args.sample_retries,
        "rewrite_retries": args.rewrite_retries,
        "workflow": "cleaning->multipoint->classification_or_forced_counting->rewrite_same_session",
        "dataset": "PointArena val",
        "requested_jobs": len(jobs),
        "sources": sorted({job.source for job in jobs}),
        "all_stages_disable_thinking": True,
        "classification_prompt_from_current_eval": True,
        "cleaning_prompt_from_user_override": True,
        "multipoint_prompt_enabled": True,
        "rewrite_prompt_from_original_category_templates": True,
    }
    pipeline.atomic_write_json(output_root / "run_config.json", config)


def process_job(job: pipeline.SampleJob, args: argparse.Namespace, output_root: Path, request_logger: pipeline.RequestLogger) -> Dict[str, Any]:
    return pipeline.process_job(job, args, output_root, request_logger)


def compute_summary(output_root: Path, jobs: List[pipeline.SampleJob], model: str, input_path: str) -> Dict[str, Any]:
    results_by_status: Dict[str, int] = {}
    results_by_source_status: Dict[str, Dict[str, int]] = {}
    invalid_files: List[Dict[str, Any]] = []
    total_output_files = 0

    step1_total = 0
    step1_kept = 0
    step1_by_label: Dict[str, Dict[str, int]] = {label: {"total": 0, "kept": 0} for label in LABELS}
    false_reject_examples: List[Dict[str, Any]] = []

    final_category_total = 0
    final_category_correct = 0
    final_category_by_label: Dict[str, Dict[str, int]] = {label: {"total": 0, "correct": 0} for label in LABELS}

    for sample_file in sorted(output_root.glob("*/*.json")):
        total_output_files += 1
        sample, validation_reasons = pipeline.load_existing_sample(sample_file)
        if sample is None:
            status = "invalid_file"
            source = sample_file.parent.name
            results_by_status[status] = results_by_status.get(status, 0) + 1
            results_by_source_status.setdefault(source, {})
            results_by_source_status[source][status] = results_by_source_status[source].get(status, 0) + 1
            invalid_files.append({"path": str(sample_file), "validation_errors": validation_reasons})
            continue

        processing = sample.get("processing", {})
        status = processing.get("status", "unknown")
        source = processing.get("source", sample_file.parent.name)
        ground_truth = sample.get("record", {}).get("ground_truth_label") or TASKTYPE_TO_LABEL.get(source)

        if validation_reasons:
            status = "invalid_file"
            invalid_files.append(
                {
                    "path": str(sample_file),
                    "record_id": sample.get("record", {}).get("id"),
                    "validation_errors": validation_reasons,
                }
            )
        results_by_status[status] = results_by_status.get(status, 0) + 1
        results_by_source_status.setdefault(source, {})
        results_by_source_status[source][status] = results_by_source_status[source].get(status, 0) + 1

        if validation_reasons or ground_truth not in LABELS:
            continue

        step1_total += 1
        step1_by_label[ground_truth]["total"] += 1
        cleaning_keep = sample.get("cleaning", {}).get("keep") is True
        if cleaning_keep:
            step1_kept += 1
            step1_by_label[ground_truth]["kept"] += 1
        else:
            false_reject_examples.append(
                {
                    "path": str(sample_file),
                    "id": sample.get("record", {}).get("id"),
                    "ground_truth": ground_truth,
                    "query": sample.get("record", {}).get("query"),
                    "cleaning_reason": sample.get("cleaning", {}).get("reason"),
                    "status": processing.get("status"),
                }
            )

        if cleaning_keep and status == "completed":
            final_category = sample.get("classification", {}).get("category")
            final_category_total += 1
            final_category_by_label[ground_truth]["total"] += 1
            if final_category == ground_truth:
                final_category_correct += 1
                final_category_by_label[ground_truth]["correct"] += 1

    overall_recall = (step1_kept / step1_total) if step1_total else 0.0
    step1_recall_by_label = {
        label: {
            "total": stats["total"],
            "kept": stats["kept"],
            "recall": (stats["kept"] / stats["total"]) if stats["total"] else 0.0,
        }
        for label, stats in step1_by_label.items()
    }
    final_category_accuracy = (final_category_correct / final_category_total) if final_category_total else 0.0
    final_category_accuracy_by_label = {
        label: {
            "total": stats["total"],
            "correct": stats["correct"],
            "accuracy": (stats["correct"] / stats["total"]) if stats["total"] else 0.0,
        }
        for label, stats in final_category_by_label.items()
    }

    summary = {
        "generated_at": utc_now(),
        "dataset": "PointArena val",
        "input": str(Path(input_path).resolve()),
        "model": model,
        "api_url": pipeline.API_URL,
        "requested_jobs": len(jobs),
        "total_output_files": total_output_files,
        "results_by_status": results_by_status,
        "results_by_source_status": results_by_source_status,
        "step1_cleaning_recall": {
            "total": step1_total,
            "kept": step1_kept,
            "recall": overall_recall,
            "by_label": step1_recall_by_label,
            "false_reject_count": len(false_reject_examples),
        },
        "final_category_accuracy_on_completed": {
            "total": final_category_total,
            "correct": final_category_correct,
            "accuracy": final_category_accuracy,
            "by_label": final_category_accuracy_by_label,
        },
        "invalid_files": invalid_files,
        "false_reject_examples": false_reject_examples[:50],
    }
    return summary


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_report(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    step1 = summary["step1_cleaning_recall"]
    final_acc = summary["final_category_accuracy_on_completed"]

    lines.append("# PointArena Full Pipeline Report")
    lines.append("")
    lines.append("## 1. Run Overview")
    lines.append("")
    lines.append(f"- Generated at: `{summary['generated_at']}`")
    lines.append(f"- Dataset: `{summary['dataset']}`")
    lines.append(f"- Input: `{summary['input']}`")
    lines.append(f"- Model: `{summary['model']}`")
    lines.append(f"- Requested jobs: `{summary['requested_jobs']}`")
    lines.append(f"- Output files: `{summary['total_output_files']}`")
    lines.append("")
    lines.append("## 2. Step-1 Cleaning Recall")
    lines.append("")
    lines.append("This dataset contains only PointArena validation tasks, so step-1 recall is the right metric to watch: it measures how many valid PointArena tasks were kept by the cleaning gate.")
    lines.append("")
    lines.append(f"- Overall step-1 recall: `{step1['kept']}/{step1['total']}` = **{format_pct(step1['recall'])}**")
    lines.append(f"- False rejects: `{step1['false_reject_count']}`")
    lines.append("")
    lines.append("| Label | Total | Kept By Step 1 | Recall |")
    lines.append("| --- | ---: | ---: | ---: |")
    for label in LABELS:
        row = step1["by_label"][label]
        lines.append(f"| {label} | {row['total']} | {row['kept']} | {format_pct(row['recall'])} |")
    lines.append("")
    lines.append("## 3. Pipeline Outcome Counts")
    lines.append("")
    for status, count in sorted(summary["results_by_status"].items()):
        lines.append(f"- `{status}`: `{count}`")
    lines.append("")
    lines.append("## 4. Final Category Accuracy On Completed Samples")
    lines.append("")
    lines.append(f"- Accuracy on completed samples: `{final_acc['correct']}/{final_acc['total']}` = **{format_pct(final_acc['accuracy'])}**")
    lines.append("")
    lines.append("| Label | Completed | Final Category Correct | Accuracy |")
    lines.append("| --- | ---: | ---: | ---: |")
    for label in LABELS:
        row = final_acc["by_label"][label]
        lines.append(f"| {label} | {row['total']} | {row['correct']} | {format_pct(row['accuracy'])} |")
    lines.append("")

    false_reject_examples = summary.get("false_reject_examples", [])
    lines.append("## 5. Step-1 False Reject Examples")
    lines.append("")
    if not false_reject_examples:
        lines.append("No false rejects were found.")
    else:
        for idx, item in enumerate(false_reject_examples[:20], start=1):
            lines.append(f"{idx}. Ground truth: `{item['ground_truth']}`")
            lines.append(f"   Query: {item['query']}")
            lines.append(f"   Cleaning reason: {item['cleaning_reason']}")
            lines.append(f"   File: `{item['path']}`")
            lines.append("")

    invalid_files = summary.get("invalid_files", [])
    lines.append("## 6. Invalid Files")
    lines.append("")
    if not invalid_files:
        lines.append("No invalid files remain after the run.")
    else:
        for item in invalid_files[:20]:
            lines.append(f"- `{item.get('path')}` -> `{item.get('validation_errors')}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise ValueError("Missing API key. Set VECTORENGINE_API_KEY or pass --api-key.")

    input_path = Path(args.input)
    output_root = SCRIPT_DIR / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    request_logger = pipeline.RequestLogger(output_root / args.requests_log)

    print("[Step 1] Selecting PointArena validation jobs...")
    jobs = select_jobs(input_path, max_samples=args.max_samples)
    write_run_config(output_root, args, jobs)
    print(f"[Step 1] Prepared {len(jobs)} jobs across {len(sorted({job.source for job in jobs}))} labels.")

    print("[Step 2] Running shared-session cleaning + multipoint + classification/counting + rewrite pipeline...")
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_job = {
            executor.submit(process_job, job, args, output_root, request_logger): job
            for job in jobs
        }
        with tqdm(total=len(future_to_job), desc="pointarena full pipeline", ncols=110) as pbar:
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = {"status": "worker_failed", "source": job.source, "error": str(exc)}
                results.append(result)
                pbar.update(1)

    print("[Step 3] Building summary and markdown report...")
    summary = compute_summary(output_root, jobs, args.model, args.input)
    pipeline.atomic_write_json(output_root / args.summary_file, summary)
    (output_root / args.report_file).write_text(build_report(summary) + "\n", encoding="utf-8")

    status_counts = Counter(result["status"] for result in results)
    print(f"[Done] Output root: {output_root}")
    print(f"[Done] Status counts: {json.dumps(dict(status_counts), ensure_ascii=False)}")
    print(
        "[Done] Step-1 recall: "
        f"{summary['step1_cleaning_recall']['kept']}/{summary['step1_cleaning_recall']['total']} "
        f"= {format_pct(summary['step1_cleaning_recall']['recall'])}"
    )


if __name__ == "__main__":
    main()
