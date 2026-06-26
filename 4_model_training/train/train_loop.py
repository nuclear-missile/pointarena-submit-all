from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, WeightedRandomSampler
except Exception:  # pragma: no cover
    torch = None
    F = None
    DataLoader = None
    WeightedRandomSampler = None

from src.eval.render_report import render_report_markdown
from src.train.checkpoint_manager import update_leaderboard
from src.utils.seed import set_seed


def to_device(batch: dict[str, Any], device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def save_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def build_loss_token_sets(tokenizer: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if torch is None:
        raise RuntimeError("torch required")

    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
    token_ids = list(range(vocab_size))
    toks = tokenizer.convert_ids_to_tokens(token_ids)

    coord_ids: list[int] = []
    for tid, tok in enumerate(toks):
        t = str(tok)
        if any(ch.isdigit() for ch in t):
            coord_ids.append(tid)

    label_ids = set(tokenizer.encode("target", add_special_tokens=False))
    label_ids.update(tokenizer.encode(" target", add_special_tokens=False))

    coord_tensor = torch.as_tensor(coord_ids, dtype=torch.long)
    label_tensor = torch.as_tensor(sorted(label_ids), dtype=torch.long)
    return coord_tensor, label_tensor


def compute_weighted_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    mode: str,
    coord_token_ids: torch.Tensor | None = None,
    label_token_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if F is None:
        raise RuntimeError("torch.nn.functional unavailable")

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_input_ids = input_ids[:, 1:].contiguous()

    vocab = int(shift_logits.size(-1))
    loss_flat = F.cross_entropy(
        shift_logits.view(-1, vocab),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    )
    token_loss = loss_flat.view_as(shift_labels)

    valid = (shift_labels != -100)
    weights = valid.float()

    m = str(mode or "uniform").strip().lower()
    if m == "coord_x2" and coord_token_ids is not None and int(coord_token_ids.numel()) > 0:
        coord_mask = torch.isin(shift_input_ids, coord_token_ids.to(shift_input_ids.device))
        weights = weights * (1.0 + coord_mask.float())
    elif m == "ignore_label_text" and label_token_ids is not None and int(label_token_ids.numel()) > 0:
        label_mask = torch.isin(shift_input_ids, label_token_ids.to(shift_input_ids.device))
        weights = weights * (~label_mask).float()

    denom = weights.sum().clamp_min(1.0)
    return (token_loss * weights).sum() / denom


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-GPU Molmo2 LoRA training")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--context_len",
        type=int,
        required=True,
        help="Sequence budget. Recommended: 1792 (with prefiltered dataset) or 1984 (no sample drop).",
    )

    # Keep micro-batch at 1 by default, scale with grad accumulation.
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_eps", type=float, default=1e-6)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_scheduler", choices=["cosine_with_warmup", "constant"], default="cosine_with_warmup")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)

    # max_epoch may be fractional for short controlled runs (e.g., 8k-step ablations)
    parser.add_argument("--max_epoch", type=float, default=1.0)
    parser.add_argument("--save_every_steps", type=int, default=2000)
    parser.add_argument("--eval_every_steps", type=int, default=2000)
    parser.add_argument("--keep_top_m", type=int, default=10)

    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_scope",
        choices=["vision+llm", "llm_only", "vision_only", "all"],
        default="vision+llm",
        help="Ablation scope over LoRA target modules.",
    )
    parser.add_argument(
        "--lora_include_projector_in_llm_only",
        dest="lora_include_projector_in_llm_only",
        action="store_true",
    )
    parser.add_argument(
        "--lora_exclude_projector_in_llm_only",
        dest="lora_include_projector_in_llm_only",
        action="store_false",
    )
    parser.add_argument("--lora_include_projector_in_vision_only", action="store_true")
    parser.add_argument(
        "--init_adapter_path",
        default="",
        help="Optional LoRA adapter path to initialize from (e.g., continue from checkpoint-2000).",
    )
    parser.add_argument(
        "--freeze_lora_branches",
        default="",
        help="Comma-separated LoRA branches to freeze after init: vision,projector,llm",
    )

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--gradient_checkpointing", dest="gradient_checkpointing", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no_amp", dest="amp", action="store_false")

    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--prefetch_factor", type=int, default=4)

    parser.add_argument("--max_train_samples", type=int, default=0, help="<=0 means use full train_jsonl")
    parser.add_argument("--max_target_points", type=int, default=64, help="Limit supervised points per sample")
    parser.add_argument(
        "--target_text_variant",
        choices=["sample_label", "html_target", "html_empty", "html_dot", "html_x", "html_self_closing", "coord_csv"],
        default="sample_label",
        help="Assistant target text protocol variant for pointing supervision.",
    )
    parser.add_argument(
        "--loss_weight_mode",
        choices=["uniform", "coord_x2", "ignore_label_text"],
        default="uniform",
        help="Token-level loss weighting mode.",
    )
    parser.add_argument("--point_mode", choices=["single", "multi", "all"], default="single")
    parser.add_argument("--include_counting_train", dest="include_counting_train", action="store_true")
    parser.add_argument("--exclude_counting_train", dest="include_counting_train", action="store_false")
    parser.add_argument("--drop_1000_scaled_data", dest="drop_1000_scaled_data", action="store_true")
    parser.add_argument("--keep_1000_scaled_data", dest="drop_1000_scaled_data", action="store_false")
    parser.add_argument("--allow_pointarena_train_data", action="store_true")

    parser.add_argument("--sampling_strategy", choices=["shuffle", "source_balanced"], default="shuffle")
    parser.add_argument("--balanced_sampling_power", type=float, default=1.0)
    parser.add_argument(
        "--sampling_stats_window_steps",
        type=int,
        default=10000,
        help="Collect sampled source stats in first N micro-steps.",
    )

    parser.add_argument("--fast_eval_subset", type=int, default=0)
    parser.add_argument("--eval_prompt_mode", choices=["plain"], default="plain")
    parser.add_argument("--eval_exclude_counting", dest="eval_exclude_counting", action="store_true")
    parser.add_argument("--eval_include_counting", dest="eval_exclude_counting", action="store_false")
    parser.add_argument("--eval_require_single_point", dest="eval_require_single_point", action="store_true")
    parser.add_argument("--eval_allow_multi_point", dest="eval_require_single_point", action="store_false")
    parser.add_argument("--eval_at_step0", dest="eval_at_step0", action="store_true")
    parser.add_argument("--no_eval_at_step0", dest="eval_at_step0", action="store_false")

    parser.add_argument("--resource_dir", default="/mnt/data/lv_qi/xing/pointarena")
    parser.add_argument("--max_crops", type=int, default=12)
    parser.add_argument("--high_res_max_crops", type=int, default=24)
    parser.add_argument("--p_high_res", type=float, default=0.7)
    parser.add_argument("--mock_train", action="store_true", help="Use fake training for unit test smoke")

    parser.set_defaults(
        gradient_checkpointing=True,
        amp=True,
        include_counting_train=False,
        drop_1000_scaled_data=True,
        eval_exclude_counting=True,
        eval_require_single_point=True,
        eval_at_step0=True,
        lora_include_projector_in_llm_only=True,
    )
    args = parser.parse_args()

    if float(args.max_epoch) <= 0:
        raise ValueError("--max_epoch must be > 0")
    if int(args.per_device_batch_size) != 1:
        raise ValueError("--per_device_batch_size must be 1; use --gradient_accumulation_steps to scale effective batch size")
    if int(args.gradient_accumulation_steps) <= 0:
        raise ValueError("--gradient_accumulation_steps must be > 0")
    if int(args.eval_every_steps) <= 0 or int(args.save_every_steps) <= 0:
        raise ValueError("--eval_every_steps and --save_every_steps must be > 0")

    args.max_train_samples = int(args.max_train_samples or 0)
    args.target_text_variant = str(args.target_text_variant or "sample_label")
    args.loss_weight_mode = str(args.loss_weight_mode or "uniform")
    args.num_workers = max(0, int(args.num_workers))
    args.prefetch_factor = max(2, int(args.prefetch_factor))
    args.sampling_stats_window_steps = max(1, int(args.sampling_stats_window_steps))
    args.balanced_sampling_power = float(args.balanced_sampling_power)
    return args


def build_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace, total_update_steps: int):
    if args.lr_scheduler == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _s: 1.0)

    warmup_steps = max(0, int(args.warmup_steps))
    min_ratio = float(args.min_lr_ratio)
    min_ratio = max(0.0, min(1.0, min_ratio))

    def _lr_lambda(step_idx: int) -> float:
        if warmup_steps > 0 and step_idx < warmup_steps:
            return float(step_idx + 1) / float(max(warmup_steps, 1))

        if total_update_steps <= warmup_steps:
            return 1.0

        progress = float(step_idx - warmup_steps) / float(max(total_update_steps - warmup_steps, 1))
        progress = max(0.0, min(1.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


def build_lora_param_groups(
    trainable_named_params: list[tuple[str, torch.nn.Parameter]],
    weight_decay: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []

    for name, param in trainable_named_params:
        lower = name.lower()
        if param.ndim <= 1 or lower.endswith(".bias") or "norm" in lower:
            no_decay.append(param)
        else:
            decay.append(param)

    groups: list[dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(weight_decay)})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})

    stats = {
        "num_decay_params": len(decay),
        "num_no_decay_params": len(no_decay),
    }
    return groups, stats


def parse_freeze_branch_set(raw: str) -> set[str]:
    out = set()
    for x in str(raw or "").split(","):
        x = x.strip().lower()
        if not x:
            continue
        out.add(x)

    allowed = {"vision", "projector", "llm"}
    bad = sorted(x for x in out if x not in allowed)
    if bad:
        raise ValueError(f"--freeze_lora_branches has invalid values: {bad}; allowed={sorted(allowed)}")
    return out


def classify_param_branch(param_name: str) -> str:
    n = (param_name or "").strip().lower()
    if "vision_backbone.image_vit" in n:
        return "vision"
    if "vision_backbone.image_projector" in n:
        return "projector"
    if "model.transformer." in n or ".transformer.blocks." in n:
        return "llm"
    return "other"


def apply_lora_branch_freeze(model: Any, freeze_branches: set[str]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "requested_branches": sorted(freeze_branches),
        "lora_total_params": 0,
        "lora_total_numel": 0,
        "frozen_params": 0,
        "frozen_numel": 0,
        "frozen_by_branch": {},
    }
    frozen_by_branch: Counter[str] = Counter()
    frozen_numel_by_branch: Counter[str] = Counter()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lower = name.lower()
        # Keep scope tight: only affect LoRA trainable tensors.
        if "lora_" not in lower:
            continue

        branch = classify_param_branch(name)
        report["lora_total_params"] += 1
        report["lora_total_numel"] += int(param.numel())

        if branch in freeze_branches:
            param.requires_grad = False
            report["frozen_params"] += 1
            report["frozen_numel"] += int(param.numel())
            frozen_by_branch[branch] += 1
            frozen_numel_by_branch[branch] += int(param.numel())

    report["frozen_by_branch"] = {
        k: {"params": int(v), "numel": int(frozen_numel_by_branch[k])}
        for k, v in sorted(frozen_by_branch.items())
    }
    return report


def configure_image_preprocessor(processor: Any, args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {
        "requested": {
            "max_crops": int(args.max_crops),
            "high_res_max_crops": int(args.high_res_max_crops),
            "p_high_res": float(args.p_high_res),
        },
        "applied": {},
        "warnings": [],
    }

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        report["warnings"].append("processor.image_processor is missing; cannot apply crop overrides")
        return report

    report["applied"]["max_crops_before"] = getattr(image_processor, "max_crops", None)
    if hasattr(image_processor, "max_crops") and int(args.max_crops) > 0:
        image_processor.max_crops = int(args.max_crops)
    report["applied"]["max_crops_after"] = getattr(image_processor, "max_crops", None)

    # HF Molmo2 processor currently does not expose high-res crop branch knobs.
    if hasattr(image_processor, "high_res_max_crops"):
        image_processor.high_res_max_crops = int(args.high_res_max_crops)
        report["applied"]["high_res_max_crops"] = getattr(image_processor, "high_res_max_crops", None)
    else:
        report["warnings"].append("high_res_max_crops not available in current HF Molmo2 image processor")

    if hasattr(image_processor, "p_high_res"):
        image_processor.p_high_res = float(args.p_high_res)
        report["applied"]["p_high_res"] = getattr(image_processor, "p_high_res", None)
    else:
        report["warnings"].append("p_high_res not available in current HF Molmo2 image processor")

    report["applied"]["size"] = getattr(image_processor, "size", None)
    report["applied"]["patch_size"] = getattr(image_processor, "patch_size", None)
    report["applied"]["pooling_size"] = getattr(image_processor, "pooling_size", None)
    return report


def build_train_loader(dataset, collator, args: argparse.Namespace):
    sampling_report: dict[str, Any] = {
        "strategy": str(args.sampling_strategy),
        "balanced_sampling_power": float(args.balanced_sampling_power),
    }

    source_counts = Counter(str((x.get("metadata", {}) or {}).get("source", "unknown")) for x in dataset.rows)
    sampling_report["dataset_source_counts"] = {k: int(v) for k, v in source_counts.items()}

    loader_kwargs: dict[str, Any] = {
        "batch_size": args.per_device_batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collator,
    }

    if args.sampling_strategy == "source_balanced":
        if WeightedRandomSampler is None:
            raise RuntimeError("WeightedRandomSampler is unavailable; torch install may be broken")

        power = float(args.balanced_sampling_power)
        row_sources = [str((x.get("metadata", {}) or {}).get("source", "unknown")) for x in dataset.rows]
        weights = [math.pow(1.0 / max(1, int(source_counts[s])), power) for s in row_sources]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
        )
        loader_kwargs["sampler"] = sampler
        loader_kwargs["shuffle"] = False

        target_probs = Counter()
        for s, c in source_counts.items():
            # expected source probability under weighted sampling
            w = math.pow(1.0 / max(1, int(c)), power)
            target_probs[s] = float(c) * w
        total = float(sum(target_probs.values()) or 1.0)
        sampling_report["balanced_expected_source_ratio"] = {
            k: float(v / total) for k, v in sorted(target_probs.items())
        }
    else:
        loader_kwargs["shuffle"] = True

    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    loader = DataLoader(dataset, **loader_kwargs)
    return loader, sampling_report


def run_mock_training(args: argparse.Namespace) -> int:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    board = []
    terminal_step = max(1, int(math.ceil(float(args.max_epoch or 1.0))))
    steps: list[int] = []
    s = int(args.save_every_steps)
    while s <= terminal_step:
        steps.append(s)
        s += int(args.save_every_steps)
    if terminal_step not in steps:
        steps.append(terminal_step)

    for step in steps:
        ckpt = out / f"checkpoint-{step}"
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "adapter_model.safetensors").write_text("mock", encoding="utf-8")
        save_json({"step": step, "mock": True}, ckpt / "train_state.json")
        metrics = {
            "step": step,
            "overall": float(20 + step % 17),
            "subtasks": {
                "affordance": 20.0,
                "spatial": 20.0,
                "reasoning": 20.0,
                "steerability": 20.0,
                "counting": 20.0,
            },
            "invalid_predictions": 0,
            "num_samples": 10,
            "checkpoint_dir": str(ckpt.resolve()),
        }
        save_json(metrics, ckpt / f"metrics_step_{step}.json")
        render_report_markdown(metrics, ckpt / f"pointarena_report_step_{step}.md", experiment=Path(args.output_dir).name)
        board = update_leaderboard(out, metrics, keep_top_m=args.keep_top_m)

    print(f"[OK] mock training completed, checkpoints={len(board)}")
    return 0


def _counter_ratio(counter: Counter[str]) -> dict[str, dict[str, float]]:
    total = int(sum(counter.values()))
    out: dict[str, dict[str, float]] = {}
    for k, v in sorted(counter.items()):
        out[k] = {
            "count": int(v),
            "ratio": float(v / max(total, 1)),
        }
    return out


def main() -> int:
    args = parse_args()
    if args.mock_train:
        return run_mock_training(args)

    if torch is None:
        raise RuntimeError("torch is required for real training; install torch or use --mock_train")

    from peft import PeftModel, get_peft_model
    from tqdm.auto import tqdm
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from src.eval.run_pointarena_eval import evaluate_model_on_val
    from src.train.collator import Molmo2PointCollator
    from src.train.lora_config import LoraArgs, build_lora_config, select_target_modules_by_scope
    from src.train.point_dataset import UnifiedPointDataset

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for real Molmo2 training")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_samples = None if int(args.max_train_samples) <= 0 else int(args.max_train_samples)
    dataset = UnifiedPointDataset(
        args.train_jsonl,
        max_samples=max_samples,
        max_target_points=int(args.max_target_points),
        point_mode=args.point_mode,
        include_counting=bool(args.include_counting_train),
        drop_1000_scaled_data=bool(args.drop_1000_scaled_data),
    )
    if len(dataset) == 0:
        raise RuntimeError(
            "No train samples available after filtering. "
            f"Check --point_mode={args.point_mode}, --include_counting_train={args.include_counting_train}, "
            f"--drop_1000_scaled_data={args.drop_1000_scaled_data}."
        )

    train_source_counts = Counter(str((x.get("metadata", {}) or {}).get("source", "unknown")) for x in dataset.rows)
    pointarena_like_sources = {
        k: int(v)
        for k, v in train_source_counts.items()
        if "pointarena" in str(k).lower()
    }
    if pointarena_like_sources and not bool(args.allow_pointarena_train_data):
        raise RuntimeError(
            "Train dataset contains PointArena-like sources, but PointArena should be eval-only. "
            f"Found: {pointarena_like_sources}. "
            "If you intentionally want this, pass --allow_pointarena_train_data."
        )

    if args.precision == "bf16":
        weight_dtype = torch.bfloat16
        amp_dtype = torch.bfloat16
    elif args.precision == "fp16":
        weight_dtype = torch.float16
        amp_dtype = torch.float16
    else:
        weight_dtype = torch.float32
        amp_dtype = None
    use_amp = bool(args.amp and amp_dtype is not None)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    processor_overrides = configure_image_preprocessor(processor, args)
    for w in (processor_overrides.get("warnings", []) or []):
        print(f"[WARN] {w}")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=weight_dtype,
    )
    model.to(device)

    coord_token_ids: torch.Tensor | None = None
    label_token_ids: torch.Tensor | None = None
    if str(args.loss_weight_mode).strip().lower() != "uniform":
        coord_token_ids, label_token_ids = build_loss_token_sets(processor.tokenizer)

    selected_target_names, all_target_candidates = select_target_modules_by_scope(
        model,
        target_scope=args.lora_target_scope,
        include_projector_in_llm_only=bool(args.lora_include_projector_in_llm_only),
        include_projector_in_vision_only=bool(args.lora_include_projector_in_vision_only),
    )
    if not selected_target_names:
        raise RuntimeError(
            f"No target modules selected for scope={args.lora_target_scope}. "
            "Adjust scope/projector flags."
        )

    selected_set = set(selected_target_names)
    selected_target_candidates = [x for x in all_target_candidates if x["module_name"] in selected_set]

    lora_args = LoraArgs(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=selected_target_names,
        bias="none",
    )
    lora_cfg = build_lora_config(model, lora_args)

    init_adapter_path = str(args.init_adapter_path or "").strip()
    if init_adapter_path:
        if not Path(init_adapter_path).exists():
            raise RuntimeError(f"--init_adapter_path does not exist: {init_adapter_path}")
        model = PeftModel.from_pretrained(model, init_adapter_path, is_trainable=True)
    else:
        model = get_peft_model(model, lora_cfg)

    # Enforce model parameter dtype to match precision selection (including LoRA params).
    if args.precision == "bf16":
        model = model.to(dtype=torch.bfloat16)
    elif args.precision == "fp16":
        model = model.to(dtype=torch.float16)

    freeze_branches = parse_freeze_branch_set(args.freeze_lora_branches)
    if freeze_branches:
        freeze_report = apply_lora_branch_freeze(model, freeze_branches)
    else:
        freeze_report = {
            "requested_branches": [],
            "lora_total_params": 0,
            "lora_total_numel": 0,
            "frozen_params": 0,
            "frozen_numel": 0,
            "frozen_by_branch": {},
        }

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.print_trainable_parameters()

    collator = Molmo2PointCollator(
        processor=processor,
        context_len=args.context_len,
        coordinate_scale="1000",
        target_text_variant=args.target_text_variant,
    )

    loader, sampling_report = build_train_loader(dataset, collator, args)

    trainable_named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    trainable_params = [p for _, p in trainable_named_params]

    param_dtype_numel = Counter()
    trainable_dtype_numel = Counter()
    for _n, _p in model.named_parameters():
        param_dtype_numel[str(_p.dtype)] += int(_p.numel())
        if _p.requires_grad:
            trainable_dtype_numel[str(_p.dtype)] += int(_p.numel())

    optimizer_groups, group_stats = build_lora_param_groups(trainable_named_params, float(args.weight_decay))

    optimizer = torch.optim.AdamW(
        optimizer_groups,
        lr=float(args.learning_rate),
        betas=(float(args.adam_beta1), float(args.adam_beta2)),
        eps=float(args.adam_eps),
        weight_decay=0.0,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=bool(use_amp and amp_dtype == torch.float16))

    num_batches = len(loader)
    if num_batches <= 0:
        raise RuntimeError("No train batches available")

    target_epochs = float(args.max_epoch)
    total_micro_steps = max(1, int(math.ceil(target_epochs * num_batches)))
    total_update_steps = max(1, math.ceil(total_micro_steps / max(1, int(args.gradient_accumulation_steps))))
    scheduler = build_scheduler(optimizer, args, total_update_steps)

    all_branch_counts = Counter(x["branch"] for x in all_target_candidates)
    selected_branch_counts = Counter(x["branch"] for x in selected_target_candidates)
    selected_leaf_counts = Counter(x["leaf_name"] for x in selected_target_candidates)

    setup_report = {
        "train_jsonl": str(Path(args.train_jsonl).resolve()),
        "val_jsonl": str(Path(args.val_jsonl).resolve()),
        "dataset_rows": len(dataset),
        "train_source_counts": {k: int(v) for k, v in train_source_counts.items()},
        "max_epoch": float(target_epochs),
        "target_micro_steps": int(total_micro_steps),
        "point_mode": args.point_mode,
        "target_text_variant": str(args.target_text_variant),
        "loss_weight_mode": str(args.loss_weight_mode),
        "loss_weight_token_set_sizes": {
            "coord_token_ids": int(coord_token_ids.numel()) if coord_token_ids is not None else 0,
            "label_token_ids": int(label_token_ids.numel()) if label_token_ids is not None else 0,
        },
        "include_counting_train": bool(args.include_counting_train),
        "drop_1000_scaled_data": bool(args.drop_1000_scaled_data),
        "precision": args.precision,
        "amp": bool(args.amp),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "processor_overrides": processor_overrides,
        "sampling": sampling_report,
        "optimizer_param_groups": group_stats,
        "lora": {
            "r": int(args.lora_r),
            "alpha": int(args.lora_alpha),
            "dropout": float(args.lora_dropout),
            "target_scope": str(args.lora_target_scope),
            "include_projector_in_llm_only": bool(args.lora_include_projector_in_llm_only),
            "include_projector_in_vision_only": bool(args.lora_include_projector_in_vision_only),
            "target_modules": list(lora_cfg.target_modules),
            "init_adapter_path": (str(Path(init_adapter_path).resolve()) if init_adapter_path else None),
            "freeze_lora_branches": sorted(freeze_branches),
            "freeze_report": freeze_report,
            "all_target_candidates": int(len(all_target_candidates)),
            "selected_target_candidates": int(len(selected_target_candidates)),
            "all_branch_counts": {k: int(v) for k, v in all_branch_counts.items()},
            "selected_branch_counts": {k: int(v) for k, v in selected_branch_counts.items()},
            "selected_leaf_counts": {k: int(v) for k, v in selected_leaf_counts.items()},
            "selected_module_table_head100": selected_target_candidates[:100],
            "all_module_table_head100": all_target_candidates[:100],
        },
        "param_dtype_numel": {k: int(v) for k, v in param_dtype_numel.items()},
        "trainable_dtype_numel": {k: int(v) for k, v in trainable_dtype_numel.items()},
    }
    save_json(setup_report, out_dir / "train_setup.json")

    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    board: list[dict[str, Any]] = []

    step = 0
    update_step = 0
    epoch = 0
    sampled_source_counts_window: Counter[str] = Counter()
    recent_losses: deque[float] = deque(maxlen=100)

    # Optional step-0 checkpoint and eval for ablations.
    if bool(args.eval_at_step0):
        ckpt_dir = out_dir / "checkpoint-0"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_dir, safe_serialization=True)
        state0 = {
            "step": 0,
            "update_step": 0,
            "epoch": 0,
            "running_loss": 0.0,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "grad_norm": None,
            "loss_avg_last_100_steps": None,
            "sampled_source_counts_first_window": {},
            "time": time.time(),
        }
        save_json(state0, ckpt_dir / "train_state.json")

        preds_path0 = ckpt_dir / "predictions_step_0.jsonl"
        metrics0 = evaluate_model_on_val(
            model=model,
            processor=processor,
            val_jsonl=args.val_jsonl,
            step=0,
            output_predictions_path=preds_path0,
            fast_eval_subset=args.fast_eval_subset,
            prompt_mode=args.eval_prompt_mode,
            resource_dir=args.resource_dir,
            exclude_counting=bool(args.eval_exclude_counting),
            require_single_point=bool(args.eval_require_single_point),
        )
        metrics0["checkpoint_dir"] = str(ckpt_dir.resolve())
        metrics0["learning_rate"] = float(optimizer.param_groups[0]["lr"])
        metrics0["update_step"] = 0
        metrics0["train_loss_avg_last_100_steps"] = None
        save_json(metrics0, ckpt_dir / "metrics_step_0.json")
        render_report_markdown(metrics0, ckpt_dir / "pointarena_report_step_0.md", experiment=out_dir.name)
        board = update_leaderboard(out_dir, metrics0, keep_top_m=args.keep_top_m)

    pbar = tqdm(total=total_micro_steps, desc="train", unit="step", dynamic_ncols=True)

    while step < total_micro_steps:
        epoch += 1

        for batch_idx, batch in enumerate(loader, start=1):
            step += 1
            model.train()

            batch_sources = batch.pop("_meta_sources", []) if isinstance(batch, dict) else []
            _ = batch.pop("_meta_ids", None) if isinstance(batch, dict) else None
            if batch_sources and step <= int(args.sampling_stats_window_steps):
                for src in batch_sources:
                    sampled_source_counts_window[str(src)] += 1

            batch = to_device(batch, device)

            if "labels" in batch:
                vocab_size = int(model.config.vocab_size)
                bad = (batch["labels"] >= vocab_size) | (batch["labels"] < -100)
                batch["labels"][bad] = -100

            forward_batch = {k: v for k, v in batch.items() if k != "labels"}
            labels = batch["labels"]

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(**forward_batch)
                loss_full = compute_weighted_ce_loss(
                    outputs.logits,
                    labels,
                    forward_batch["input_ids"],
                    mode=str(args.loss_weight_mode),
                    coord_token_ids=coord_token_ids,
                    label_token_ids=label_token_ids,
                )
                raw_loss = float(loss_full.detach().float().item())
                loss = loss_full / args.gradient_accumulation_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running_loss += raw_loss
            recent_losses.append(raw_loss)

            reached_target = step >= total_micro_steps
            do_opt_step = (step % args.gradient_accumulation_steps == 0) or reached_target

            grad_norm = None
            if do_opt_step:
                max_gn = float(args.max_grad_norm)
                if max_gn > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_gn)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                update_step += 1

            lr_now = float(optimizer.param_groups[0]["lr"])
            avg_loss = running_loss / max(step, 1)
            loss_avg_100 = float(sum(recent_losses) / len(recent_losses)) if recent_losses else None
            pbar.set_postfix(
                loss=f"{raw_loss:.4f}",
                avg=f"{avg_loss:.4f}",
                l100=("NA" if loss_avg_100 is None else f"{loss_avg_100:.4f}"),
                lr=f"{lr_now:.2e}",
                upd=update_step,
                epoch=epoch,
            )
            pbar.update(1)

            do_eval = (step % args.eval_every_steps == 0) or reached_target
            do_save = (step % args.save_every_steps == 0) or do_eval or reached_target

            ckpt_dir: Path | None = None
            if do_save:
                ckpt_dir = out_dir / f"checkpoint-{step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(ckpt_dir, safe_serialization=True)
                state = {
                    "step": step,
                    "update_step": update_step,
                    "epoch": epoch,
                    "epoch_progress": float(step / max(1, num_batches)),
                    "running_loss": running_loss,
                    "learning_rate": lr_now,
                    "grad_norm": None
                    if grad_norm is None
                    else float(grad_norm.detach().float().item() if torch.is_tensor(grad_norm) else grad_norm),
                    "loss_avg_last_100_steps": loss_avg_100,
                    "sampled_source_counts_first_window": {
                        "window_steps": int(min(step, args.sampling_stats_window_steps)),
                        "total_seen": int(sum(sampled_source_counts_window.values())),
                        "by_source": _counter_ratio(sampled_source_counts_window),
                    },
                    "time": time.time(),
                }
                save_json(state, ckpt_dir / "train_state.json")

            if do_eval:
                assert ckpt_dir is not None
                preds_path = ckpt_dir / f"predictions_step_{step}.jsonl"
                metrics = evaluate_model_on_val(
                    model=model,
                    processor=processor,
                    val_jsonl=args.val_jsonl,
                    step=step,
                    output_predictions_path=preds_path,
                    fast_eval_subset=args.fast_eval_subset,
                    prompt_mode=args.eval_prompt_mode,
                    resource_dir=args.resource_dir,
                    exclude_counting=bool(args.eval_exclude_counting),
                    require_single_point=bool(args.eval_require_single_point),
                )
                metrics["checkpoint_dir"] = str(ckpt_dir.resolve())
                metrics["learning_rate"] = lr_now
                metrics["update_step"] = update_step
                metrics["train_loss_avg_last_100_steps"] = loss_avg_100
                save_json(metrics, ckpt_dir / f"metrics_step_{step}.json")
                render_report_markdown(metrics, ckpt_dir / f"pointarena_report_step_{step}.md", experiment=out_dir.name)
                board = update_leaderboard(out_dir, metrics, keep_top_m=args.keep_top_m)

            if reached_target:
                break

    pbar.close()
    print(
        f"[OK] training done. micro_steps={step}, update_steps={update_step}, "
        f"epochs_seen={epoch}, target_micro_steps={total_micro_steps}, leaderboard={len(board)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
