#!/usr/bin/env python3
"""Streaming runner for mix4 PointArena processing with custom selection and early-stop support."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import process_mix4 as pipeline  # noqa: E402


def parse_csv(raw_value: str) -> List[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def parse_source_quotas(raw_value: str) -> Dict[str, int]:
    quotas: Dict[str, int] = {}
    for item in parse_csv(raw_value):
        if ':' not in item:
            raise ValueError(f"Invalid source quota item: {item}")
        source, value = item.split(':', 1)
        quotas[source.strip()] = int(value.strip())
    return quotas


def unique_resolved_paths(paths: Iterable[str | Path]) -> List[Path]:
    seen: set[str] = set()
    resolved_paths: List[Path] = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        resolved_paths.append(path)
    return resolved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mix4 processing with custom selection and optional early stop.")
    parser.add_argument("--api-key", default=os.getenv("VECTORENGINE_API_KEY"), help="VectorEngine API key.")
    parser.add_argument("--model", default=pipeline.DEFAULT_MODEL, help="Model name. Defaults to current evaluate_pointarena.py setting.")
    parser.add_argument("--workers", type=int, default=32, help="Parallel workers across samples.")
    parser.add_argument("--timeout", type=int, default=60, help="Per-request read timeout in seconds.")
    parser.add_argument("--connect-timeout", type=int, default=15, help="Per-request connection timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=6, help="Network retry count per request.")
    parser.add_argument("--parse-retries", type=int, default=2, help="How many same-session JSON repair turns to allow per JSON stage.")
    parser.add_argument("--sample-retries", type=int, default=4, help="How many full-sample retries to allow for invalid or truncated outputs.")
    parser.add_argument("--rewrite-retries", type=int, default=3, help="How many same-session rewrite repair turns to allow.")
    parser.add_argument("--output-root", required=True, help="Output root directory under pointarena_extract.")
    parser.add_argument("--requests-log", default="requests_log.jsonl", help="Global request log filename under output root.")
    parser.add_argument("--summary-file", default="summary.json", help="Summary filename under output root.")
    parser.add_argument("--selection-file", default="selection_stats.json", help="Selection stats filename under output root.")
    parser.add_argument("--progress-file", default="target_progress.json", help="Target progress filename under output root.")
    parser.add_argument("--overwrite", action="store_true", help="Re-run even if a per-sample output file already exists.")
    parser.add_argument("--sources", default="", help="Comma-separated source names to include. Empty means all.")
    parser.add_argument("--exclude-sources", default="", help="Comma-separated source names to exclude.")
    parser.add_argument("--source-quotas", default="", help="Comma-separated source quotas like pixmo_points:3334,refspatial:3333.")
    parser.add_argument("--max-new-jobs", type=int, default=10000, help="Maximum number of newly selected jobs to run.")
    parser.add_argument("--resume-from-output-root", default="", help="Existing output root used to derive per-source start ranks.")
    parser.add_argument("--count-roots", default="", help="Comma-separated output roots used to seed existing category counts.")
    parser.add_argument("--points-abs-len-eq", type=int, default=None, help="Only select records whose points_abs length equals this value.")
    parser.add_argument("--disable-counting-for-single-point", action="store_true", help="When points_abs has exactly one point, remove Counting from the classification label set and treat counting-like tasks as None.")
    parser.add_argument("--target-category", default="", help="If set, stop submitting new work once this category count reaches target-total.")
    parser.add_argument("--target-total", type=int, default=0, help="Desired total count for target-category across count-roots and this run.")
    return parser.parse_args()


def derive_start_ranks(output_root: Optional[Path]) -> Dict[str, int]:
    if output_root is None or not output_root.exists():
        return {}

    starts: Dict[str, int] = {}
    for sample_file in sorted(output_root.glob("*/*.json")):
        try:
            sample = json.loads(sample_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        processing = sample.get("processing", {})
        source = processing.get("source") or sample_file.parent.name
        source_rank = processing.get("source_rank")
        if not isinstance(source_rank, int):
            prefix = sample_file.stem.split("__", 1)[0]
            if prefix.isdigit():
                source_rank = int(prefix)
            else:
                continue
        starts[source] = max(starts.get(source, 0), source_rank)
    return starts


def count_completed_categories(output_roots: Iterable[Path]) -> Counter[str]:
    category_counts: Counter[str] = Counter()
    for output_root in output_roots:
        if not output_root.exists():
            continue
        for sample_file in sorted(output_root.glob("*/*.json")):
            sample, reasons = pipeline.load_existing_sample(sample_file)
            if sample is None or reasons:
                continue
            if sample.get("processing", {}).get("status") != "completed":
                continue
            category = sample.get("classification", {}).get("category")
            if isinstance(category, str) and category:
                category_counts[category] += 1
    return category_counts


def select_jobs(
    include_sources: set[str],
    exclude_sources: set[str],
    source_quotas: Dict[str, int],
    start_ranks: Dict[str, int],
    max_new_jobs: int,
    points_abs_len_eq: Optional[int],
) -> tuple[List[pipeline.SampleJob], Dict[str, Any]]:
    jobs: List[pipeline.SampleJob] = []
    selected_by_source: Counter[str] = Counter()
    scanned_by_source: Counter[str] = Counter()
    skipped_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    global_rank = 0

    for source_file in sorted(pipeline.BY_SOURCE_DIR.glob("*.jsonl")):
        source = source_file.stem
        if include_sources and source not in include_sources:
            continue
        if source in exclude_sources:
            continue
        quota = source_quotas.get(source)
        if quota is not None and quota <= 0:
            continue

        start_rank = start_ranks.get(source, 0)
        with source_file.open("r", encoding="utf-8") as f:
            for source_rank, line in enumerate(f, start=1):
                if source_rank <= start_rank:
                    continue
                scanned_by_source[source] += 1
                record = json.loads(line)
                points_abs = record.get("points_abs")
                if points_abs_len_eq is not None:
                    if not isinstance(points_abs, list):
                        skipped_by_source[source]["points_abs_not_list"] += 1
                        continue
                    if len(points_abs) != points_abs_len_eq:
                        skipped_by_source[source][f"points_abs_len_ne_{points_abs_len_eq}"] += 1
                        continue

                global_rank += 1
                jobs.append(
                    pipeline.SampleJob(
                        source=source,
                        source_rank=source_rank,
                        global_rank=global_rank,
                        source_file=str(source_file),
                        record=record,
                    )
                )
                selected_by_source[source] += 1
                if len(jobs) >= max_new_jobs:
                    selection = {
                        "selected_jobs": len(jobs),
                        "selected_by_source": dict(sorted(selected_by_source.items())),
                        "scanned_by_source": dict(sorted(scanned_by_source.items())),
                        "skipped_by_source": {k: dict(sorted(v.items())) for k, v in sorted(skipped_by_source.items())},
                        "start_ranks": dict(sorted(start_ranks.items())),
                        "points_abs_len_eq": points_abs_len_eq,
                        "source_quotas": dict(sorted(source_quotas.items())),
                        "selection_stopped_because": "max_new_jobs_reached",
                    }
                    return jobs, selection
                if quota is not None and selected_by_source[source] >= quota:
                    break

    selection = {
        "selected_jobs": len(jobs),
        "selected_by_source": dict(sorted(selected_by_source.items())),
        "scanned_by_source": dict(sorted(scanned_by_source.items())),
        "skipped_by_source": {k: dict(sorted(v.items())) for k, v in sorted(skipped_by_source.items())},
        "start_ranks": dict(sorted(start_ranks.items())),
        "points_abs_len_eq": points_abs_len_eq,
        "source_quotas": dict(sorted(source_quotas.items())),
        "selection_stopped_because": "source_exhausted",
    }
    return jobs, selection


def read_completed_category(output_path: Path) -> Optional[str]:
    sample, reasons = pipeline.load_existing_sample(output_path)
    if sample is None or reasons:
        return None
    if sample.get("processing", {}).get("status") != "completed":
        return None
    category = sample.get("classification", {}).get("category")
    if isinstance(category, str) and category:
        return category
    return None


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_summary(output_root: Path) -> Dict[str, Any]:
    summary_script = SCRIPT_DIR / "summarize_mix4_outputs.py"
    subprocess.run(
        [sys.executable, str(summary_script), "--output-root", str(output_root)],
        check=True,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    summary_path = output_root / "summary_detailed.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise ValueError("Missing API key. Set VECTORENGINE_API_KEY or pass --api-key.")

    include_sources = set(parse_csv(args.sources))
    exclude_sources = set(parse_csv(args.exclude_sources))
    source_quotas = parse_source_quotas(args.source_quotas)
    output_root = (SCRIPT_DIR / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    resume_root = (SCRIPT_DIR / args.resume_from_output_root).resolve() if args.resume_from_output_root else None
    start_ranks = derive_start_ranks(resume_root)

    count_roots = unique_resolved_paths((SCRIPT_DIR / root).resolve() if not Path(root).is_absolute() else Path(root) for root in parse_csv(args.count_roots))
    count_roots = unique_resolved_paths([*count_roots, output_root])
    existing_category_counts = count_completed_categories(count_roots)

    jobs, selection = select_jobs(
        include_sources=include_sources,
        exclude_sources=exclude_sources,
        source_quotas=source_quotas,
        start_ranks=start_ranks,
        max_new_jobs=args.max_new_jobs,
        points_abs_len_eq=args.points_abs_len_eq,
    )

    selection_payload = {
        "generated_at": pipeline.utc_now(),
        "script": str(Path(__file__).resolve()),
        "output_root": str(output_root),
        "resume_root": str(resume_root) if resume_root else None,
        "count_roots": [str(path) for path in count_roots],
        "existing_category_counts": dict(sorted(existing_category_counts.items())),
        "target_category": args.target_category or None,
        "target_total": args.target_total,
        **selection,
    }
    write_json(output_root / args.selection_file, selection_payload)

    request_logger = pipeline.RequestLogger(output_root / args.requests_log)
    if not hasattr(args, "limit_per_source"):
        args.limit_per_source = None
    pipeline.write_run_config(output_root, args, jobs)

    current_category_count = existing_category_counts.get(args.target_category, 0) if args.target_category else 0
    target_reached = bool(args.target_category and args.target_total and current_category_count >= args.target_total)
    results: List[Dict[str, Any]] = []
    new_completed_category_counts: Counter[str] = Counter()
    stop_reason = "target_already_reached" if target_reached else "source_exhausted_or_job_limit"

    print(f"[Select] Prepared {len(jobs)} jobs.")
    if args.target_category:
        print(f"[Target] Existing {args.target_category}: {current_category_count} / {args.target_total}")

    executor = ThreadPoolExecutor(max_workers=args.workers)
    futures: Dict[Future[Dict[str, Any]], pipeline.SampleJob] = {}
    job_iter = iter(jobs)
    submitted_jobs = 0

    def submit_more() -> None:
        nonlocal submitted_jobs
        while not target_reached and len(futures) < args.workers:
            try:
                job = next(job_iter)
            except StopIteration:
                return
            future = executor.submit(pipeline.process_job, job, args, output_root, request_logger)
            futures[future] = job
            submitted_jobs += 1

    try:
        submit_more()
        while futures:
            done, _ = wait(set(futures.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                job = futures.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = {
                        "status": "worker_failed",
                        "source": job.source,
                        "output_path": str(pipeline.output_path_for_job(output_root, job)),
                        "error": str(exc),
                    }
                results.append(result)

                output_path = Path(result.get("output_path", "")) if result.get("output_path") else None
                if output_path and output_path.exists():
                    category = read_completed_category(output_path)
                    if category:
                        new_completed_category_counts[category] += 1
                        if args.target_category and category == args.target_category:
                            current_category_count += 1
                            if args.target_total and current_category_count >= args.target_total:
                                target_reached = True
                                stop_reason = "target_reached"

            if not target_reached:
                submit_more()
            elif futures:
                for future in list(futures.keys()):
                    future.cancel()
    finally:
        executor.shutdown(wait=True, cancel_futures=False)

    if target_reached and args.target_total and current_category_count >= args.target_total:
        stop_reason = "target_reached"

    pipeline.write_summary(output_root, args.summary_file, args, jobs)
    summary = run_summary(output_root)

    progress_payload = {
        "generated_at": pipeline.utc_now(),
        "output_root": str(output_root),
        "resume_root": str(resume_root) if resume_root else None,
        "count_roots": [str(path) for path in count_roots],
        "target_category": args.target_category or None,
        "target_total": args.target_total,
        "existing_category_counts": dict(sorted(existing_category_counts.items())),
        "new_completed_category_counts": dict(sorted(new_completed_category_counts.items())),
        "final_category_counts": dict(sorted((existing_category_counts + new_completed_category_counts).items())),
        "submitted_jobs": submitted_jobs,
        "finished_jobs": len(results),
        "target_reached": bool(args.target_category and args.target_total and current_category_count >= args.target_total),
        "stop_reason": stop_reason,
        "summary_detailed_file": str(output_root / "summary_detailed.json"),
        "summary_detailed": summary,
    }
    write_json(output_root / args.progress_file, progress_payload)

    print(json.dumps(progress_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
