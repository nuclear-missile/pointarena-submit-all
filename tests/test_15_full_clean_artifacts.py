from __future__ import annotations

import json
from pathlib import Path


def _count_jsonl(path: Path) -> int:
    return sum(1 for ln in path.read_text(encoding='utf-8').splitlines() if ln.strip())


def test_full_clean_outputs_exist() -> None:
    root = Path('artifacts')
    assert (root / 'clean_full' / 'full_clean_stats.json').exists()
    assert (root / 'clean_full' / 'final_stats_report.json').exists()
    assert (root / 'clean_full' / 'merged_coords_local.jsonl').exists()
    assert (root / 'clean_full' / 'merged_coords_with_missing_images.jsonl').exists()


def test_where2place_question_files_exist() -> None:
    root = Path('artifacts/raw/where2place_full')
    p1 = root / 'point_questions.jsonl'
    p2 = root / 'bbox_questions.jsonl'
    assert p1.exists()
    assert p2.exists()
    assert _count_jsonl(p1) == 100
    assert _count_jsonl(p2) == 100


def test_robopoint_clean_has_expected_scale() -> None:
    stats = json.loads(Path('artifacts/clean_full/final_stats_report.json').read_text(encoding='utf-8'))
    r = stats['clean_counts']['robopoint_with_coords_all_task']
    # Tolerant band around expected public counts 347K/320K.
    assert 330000 <= r.get('object_ref', 0) <= 360000
    assert 300000 <= r.get('free_space_ref', 0) <= 340000
