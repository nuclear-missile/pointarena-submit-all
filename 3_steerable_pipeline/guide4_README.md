# Guide4 strict SAM3-filtered task pipeline

Guide4 rebuilds the steerable-task data flow with stricter programmatic filters.
It does not use an LLM/Gemini-style judge during candidate construction.  The
only optional API stage is question rewriting, and it is disabled by default.

## Enabled Task Types

- `move_until_axis_fixed`
- `move_until_axis_from_other_object`
- `move_then_axis_limited_fixed`
- `nearest_among_objects`
- `nearest_among_objects_axis_constrained`

Disabled in guide4:

- `relative_cardinal_angle10`
- `part_or_subpart_controlled`

## Core Rules

- Candidate judging is programmatic, not API-based.
- Filtering is stricter and drops text-like, caption-like, attribute-only, abstract,
  relational, and artifact labels before candidate generation.
- SAM3 masks are built per image by sending every retained label category and
  all of that category's points to `sam3_tracker`.
- Target ambiguity and anchor validity are checked against SAM3 masks.
- Target-class masks are dilated by 5 pixels before anchor exclusion.
- Directional anchors are sampled with polar angles within `+/-10` degrees of
  the requested direction.
- `move_until_axis_from_other_object` uses another retained object center as the
  anchor point and ignores that anchor object's SAM group when checking path
  blocking.
- Single-direction tasks draw one 3-pixel-wide segment from anchor to target.
- Two-step tasks draw two connected 3-pixel-wide segments.
- For path ambiguity, the path pixel set is subtracted by the target-class mask
  union, and the remainder must be connected.
- Directional candidates are also rejected when the path crosses non-target SAM
  masks, so the line of movement stays visually clean.
- `nearest_among_objects` only uses labels with 2-4 object instances in the same
  image, and nearestness is decided by programmatic Euclidean distance with an
  explicit distance margin.
- `nearest_among_objects_axis_constrained` uses axis-only anchors and requires
  the target direction to stay within `+/-10` degrees while also rejecting any
  same-class object that is closer inside the same direction sector
  (`+/-45` degrees).
- `task_candidates.parquet` is kept in deterministic image/object/task order for
  sequential traversal.

## Run

Smoke run:

```bash
python make_steerable1/guide4/scripts/run_pipeline.py --force --limit-images 3 --target-per-task 5
```

Larger run:

```bash
python make_steerable1/guide4/scripts/run_pipeline.py --force --limit-images 50 --target-per-task 50
```

Curated subset smoke run:

```bash
python make_steerable1/guide4/scripts/run_pipeline.py --force \
  --point-records make_steerable1/guide4/_cache/curated_smoke_points.parquet \
  --limit-images 5 --target-per-task 3 \
  --output-root make_steerable1/guide4/outputs_curated_smoke_v2
```

Optional API-only rewrite stage:

```bash
export GUIDE4_REWRITE_API_KEY=...
python make_steerable1/guide4/scripts/run_pipeline.py --stage rewrite --enable-api-rewrite \
  --rewrite-api-key-env GUIDE4_REWRITE_API_KEY \
  --rewrite-api-base-url https://api.vectorengine.ai/v1 \
  --rewrite-model your-rewrite-model
```

## Outputs

Default root:

```text
make_steerable1/guide4/outputs/
```

Important files:

- `01_filtered/object_records.parquet`: strict object-level records.
- `01_filtered/dropped_objects.parquet`: programmatic object drops and reasons.
- `02_sam_masks/<image_id>/preview.png`: SAM3 segmentation preview.
- `02_sam_masks/<image_id>/masks.json`: per-label SAM3 mask metadata.
- `03_task_pool/task_candidates.parquet`: accepted candidates.
- `03_task_pool/dropped_task_candidates.parquet`: candidate-stage rejection reasons.
- `04_rewrites/rewrite_results.parquet`: template and optional API rewrites.
- `05_visuals/candidates/*.png`: segmentation + anchor/path + path-judge visualizations.
- `05_visuals/rewrites/*.png`: final-question visualizations.
- `07_final/accepted_samples.jsonl`: final examples with multiple styles.
