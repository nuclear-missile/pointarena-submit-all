#!/usr/bin/env python3
"""Summarize mix4 cleaning/classification/rewrite outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict

import process_mix4 as pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize mix4 pipeline output files.")
    parser.add_argument("--output-root", required=True, help="Output root directory to summarize.")
    parser.add_argument("--summary-file", default="summary_detailed.json", help="Detailed JSON summary filename.")
    parser.add_argument("--report-file", default="report_detailed.md", help="Markdown report filename.")
    return parser.parse_args()


def load_sample(path: Path) -> tuple[Dict[str, Any] | None, list[str]]:
    try:
        sample = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return None, [f"file_read_or_json_error:{exc}"]
    return sample, pipeline.validate_sample_payload(sample)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    if not output_root.exists():
        raise FileNotFoundError(f"Missing output root: {output_root}")

    by_status: Counter[str] = Counter()
    by_source_status: dict[str, Counter[str]] = defaultdict(Counter)
    category_counts: Counter[str] = Counter()
    by_source_category: dict[str, Counter[str]] = defaultdict(Counter)
    filter_stage_counts: Counter[str] = Counter()
    rule_filtered_counts: Counter[str] = Counter()
    invalid_reason_counts: Counter[str] = Counter()

    total_files = 0
    valid_files = 0
    legal_completed = 0
    legal_completed_by_source: Counter[str] = Counter()

    for sample_file in sorted(output_root.glob('*/*.json')):
        total_files += 1
        sample, validation_errors = load_sample(sample_file)
        if sample is None:
            by_status['invalid_file'] += 1
            invalid_reason_counts.update(validation_errors)
            continue
        if validation_errors:
            by_status['invalid_file'] += 1
            invalid_reason_counts.update(validation_errors)
            source = sample.get('processing', {}).get('source', sample_file.parent.name)
            by_source_status[source]['invalid_file'] += 1
            continue

        valid_files += 1
        processing = sample.get('processing', {})
        source = processing.get('source', sample_file.parent.name)
        status = processing.get('status', 'unknown')
        by_status[status] += 1
        by_source_status[source][status] += 1

        if status == 'completed':
            legal_completed += 1
            legal_completed_by_source[source] += 1
            category = sample.get('classification', {}).get('category')
            if category:
                category_counts[category] += 1
                by_source_category[source][category] += 1

        if status == 'filtered_out':
            filter_stage = processing.get('filter_stage', 'unknown')
            filter_stage_counts[filter_stage] += 1
            rule_filtered = processing.get('rule_filtered')
            if rule_filtered:
                rule_filtered_counts[rule_filtered] += 1

    summary = {
        'output_root': str(output_root),
        'total_files': total_files,
        'valid_files': valid_files,
        'legal_completed_samples': legal_completed,
        'legal_completed_by_source': dict(sorted(legal_completed_by_source.items())),
        'results_by_status': dict(sorted(by_status.items())),
        'results_by_source_status': {k: dict(sorted(v.items())) for k, v in sorted(by_source_status.items())},
        'category_counts': dict(sorted(category_counts.items())),
        'category_counts_by_source': {k: dict(sorted(v.items())) for k, v in sorted(by_source_category.items())},
        'filter_stage_counts': dict(sorted(filter_stage_counts.items())),
        'rule_filtered_counts': dict(sorted(rule_filtered_counts.items())),
        'invalid_reason_counts': dict(sorted(invalid_reason_counts.items())),
    }

    summary_path = output_root / args.summary_file
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    lines = [
        '# Mix4 Pipeline Summary',
        '',
        f'- Output root: `{output_root}`',
        f'- Total files: {total_files}',
        f'- Valid files: {valid_files}',
        f'- Legal completed samples: {legal_completed}',
        '',
        '## Status Counts',
    ]
    for key, value in sorted(by_status.items()):
        lines.append(f'- {key}: {value}')
    lines.extend(['', '## Legal Completed By Source'])
    for key, value in sorted(legal_completed_by_source.items()):
        lines.append(f'- {key}: {value}')
    lines.extend(['', '## Category Counts'])
    for key, value in sorted(category_counts.items()):
        lines.append(f'- {key}: {value}')
    lines.extend(['', '## Filter Stage Counts'])
    for key, value in sorted(filter_stage_counts.items()):
        lines.append(f'- {key}: {value}')
    lines.extend(['', '## Rule Filter Counts'])
    for key, value in sorted(rule_filtered_counts.items()):
        lines.append(f'- {key}: {value}')
    if invalid_reason_counts:
        lines.extend(['', '## Invalid Reason Counts'])
        for key, value in sorted(invalid_reason_counts.items()):
            lines.append(f'- {key}: {value}')
    (output_root / args.report_file).write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
