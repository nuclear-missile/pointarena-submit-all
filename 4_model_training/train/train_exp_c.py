#!/usr/bin/env python3
"""
Training script for Experiments B and C.

Modifies the standard training pipeline to:
  - Wrap Molmo2 with Molmo2IterPosWrapper (ViT hooks + point injection)
  - Use IterPosCollatorB or IterPosCollatorC
  - Mixed joint training (base pointing + correction)

Usage:
  python3 -m filtered_data_ft.run_lora_train_iter_pos_BC --exp B [args...]
  python3 -m filtered_data_ft.run_lora_train_iter_pos_BC --exp C [args...]
"""
from __future__ import annotations

import argparse, json, math, os, sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel, get_peft_model

from src.train.pointarena_rewritten_dataset_steerable_d import (
    PointArenaSteerableDTrainingDataset,
    DEFAULT_STEERABLE_D_SUMMARY_PATH,
)
from src.train.checkpoint_manager import update_leaderboard
from src.utils.seed import set_seed

from filtered_data_ft.plan_iter_pos.molmo2_iter_pos_model import Molmo2IterPosWrapper
from filtered_data_ft.plan_iter_pos.collator_iter_pos_BC import IterPosCollatorB, IterPosCollatorC
from filtered_data_ft.plan_iter_pos.iter_pos_dataset import IterPosDatasetWrapper


def build_lora_config(model, lora_args):
    from peft import LoraConfig
    return LoraConfig(
        r=lora_args["r"], lora_alpha=lora_args["alpha"],
        lora_dropout=lora_args["dropout"],
        target_modules=lora_args["target_modules"],
        bias="none", task_type="CAUSAL_LM",
    )


def select_target_modules(model, target_scope):
    """Select LoRA target modules, reusing baseline logic."""
    from src.train.lora_config import select_target_modules_by_scope
    selected, _all = select_target_modules_by_scope(model, target_scope=target_scope)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["B", "C"], required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--summary_path", default=str(DEFAULT_STEERABLE_D_SUMMARY_PATH))
    parser.add_argument("--output_base", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--clean3_roots_json", default="[]")
    parser.add_argument("--sam_clean_jsonl", default="")
    parser.add_argument("--val_jsonl", default="")
    parser.add_argument("--init_adapter_path", default="")
    parser.add_argument("--resume_from_checkpoint", default="",
                        help="Resume training from checkpoint dir (loads weights + optimizer + scheduler)")

    parser.add_argument("--noise_sigma", type=float, default=100.0)
    parser.add_argument("--drop_prob", type=float, default=0.15)
    parser.add_argument("--insert_prob", type=float, default=0.15)
    parser.add_argument("--mix_ratio", type=float, default=0.5)

    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--context_len", type=int, default=2048)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_scheduler", default="cosine_with_warmup")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--save_every_steps", type=int, default=2000)
    parser.add_argument("--eval_every_steps", type=int, default=2000)
    parser.add_argument("--keep_top_m", type=int, default=99999)

    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_scope", default="vision+llm")

    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--point_mode", default="all")
    parser.add_argument("--category_balance_json", default="{}")
    parser.add_argument("--sampling_strategy", default="shuffle")
    parser.add_argument("--allow_missing_or_invalid_images", action="store_true")
    parser.add_argument("--exclude_pointarena_eval", action="store_true")
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--max_target_points", type=int, default=64)

    parser.add_argument("--eval_include_counting", action="store_true")
    parser.add_argument("--eval_allow_multi_point", action="store_true")
    parser.add_argument("--no_eval_at_step0", action="store_true")
    parser.add_argument("--fast_eval_subset", type=int, default=0)
    parser.add_argument("--eval_prompt_mode", default="plain")
    parser.add_argument("--max_crops", type=int, default=12)
    parser.add_argument("--high_res_max_crops", type=int, default=24)
    parser.add_argument("--p_high_res", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_text_variant", default="sample_label")

    args = parser.parse_args()
    device = torch.device("cuda")

    print(f"=== Experiment {args.exp} ===")
    print(f"Point MLP + Visual: {args.exp} in [B,C]")
    print(f"Coord Map: {args.exp == 'C'}")
    print(f"mix_ratio={args.mix_ratio} noise={args.noise_sigma}")
    print(f"Output: {args.output_dir}")

    # ── Dataset ──────────────────────────────────────────────
    output_base = Path(args.output_base)
    clean3_roots = json.loads(args.clean3_roots_json) if args.clean3_roots_json else []
    sam_jsonl = args.sam_clean_jsonl or None
    val_jsonl = args.val_jsonl or None

    dataset = PointArenaSteerableDTrainingDataset(
        summary_path=args.summary_path,
        pointarena_output_base=output_base,
        pointarena_roots=None,
        clean3_roots=[Path(r) for r in clean3_roots],
        sam_clean_jsonl=Path(sam_jsonl) if sam_jsonl else None,
        force_rebuild_summary=False, rebuild_if_missing=True,
        require_local_image=not args.allow_missing_or_invalid_images,
        exclude_pointarena_eval=args.exclude_pointarena_eval,
        point_mode=args.point_mode,
        max_target_points=args.max_target_points,
        category_balance=json.loads(args.category_balance_json) if args.category_balance_json else None,
        shuffle_selected=True,
        selection_seed=args.seed,
    )
    dataset = IterPosDatasetWrapper(
        dataset, noise_sigma=args.noise_sigma,
        drop_prob=args.drop_prob, insert_prob=args.insert_prob,
        mix_ratio=args.mix_ratio, seed=args.seed,
    )
    print(f"Dataset: {len(dataset)} samples (joint, mix={args.mix_ratio})")

    # ── Model ─────────────────────────────────────────────────
    print(f"Loading model from {args.model_path}...")
    weight_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16

    base_model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=weight_dtype,
    )
    base_model.to(device)

    # Step 1: Apply LoRA to base model
    lora_targets = select_target_modules(base_model, args.lora_target_scope)
    print(f"LoRA targets: {len(lora_targets)} modules")

    lora_cfg = build_lora_config(base_model, {
        "r": args.lora_r, "alpha": args.lora_alpha,
        "dropout": args.lora_dropout, "target_modules": lora_targets,
    })
    model = get_peft_model(base_model, lora_cfg)

    # Load LoRA adapter: resume from checkpoint takes priority over init_adapter_path
    if args.resume_from_checkpoint:
        model = PeftModel.from_pretrained(model, args.resume_from_checkpoint, is_trainable=True)
        model.to(device)
        print(f"Loaded LoRA adapter from checkpoint: {args.resume_from_checkpoint}")
    elif args.init_adapter_path:
        model = PeftModel.from_pretrained(model, args.init_adapter_path, is_trainable=True)
        model.to(device)
        print(f"Loaded LoRA adapter from init: {args.init_adapter_path}")

    model = model.to(dtype=weight_dtype)

    # Step 2: Wrap LoRA model with iterative position correction
    wrapper = Molmo2IterPosWrapper(model, exp_mode=args.exp)
    wrapper.to(device)
    model = wrapper  # Use wrapper for all subsequent training ops

    # Load wrapper params if resuming
    if args.resume_from_checkpoint:
        wp_path = Path(args.resume_from_checkpoint) / "wrapper_params.pt"
        if wp_path.exists():
            wp = torch.load(wp_path, map_location=device)
            model.point_mlp.load_state_dict(wp["point_mlp"])
            model.visual_proj.load_state_dict(wp["visual_proj"])
            model.point_norm.load_state_dict(wp["point_norm"])
            if model.coord_map_encoder is not None and "coord_map_encoder" in wp:
                model.coord_map_encoder.load_state_dict(wp["coord_map_encoder"])
            print(f"Loaded wrapper params from {wp_path}")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.print_trainable_parameters()

    # ── Collator ──────────────────────────────────────────────
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    CollatorCls = IterPosCollatorC if args.exp == "C" else IterPosCollatorB
    collator = CollatorCls(
        processor=processor, context_len=args.context_len,
        coordinate_scale="1000", target_text_variant=args.target_text_variant,
        noise_sigma=args.noise_sigma, drop_prob=args.drop_prob,
        insert_prob=args.insert_prob,
    )

    # ── DataLoader builder (recreated each epoch for category rolling) ──
    def _build_train_loader(dset):
        sampler = None
        if args.sampling_strategy == "source_balanced":
            from collections import Counter as _C
            src_counts = _C()
            for i in range(len(dset)):
                try:
                    s = dset._base.rows[i] if hasattr(dset, '_base') else dset[i]
                except Exception:
                    s = dset[i]
                src_counts[str(s.get("metadata", {}).get("source", "unknown"))] += 1
            weights = [1.0 / max(1, src_counts.get(
                str(dset[i].get("metadata", {}).get("source", "unknown")), 1))
                       for i in range(len(dset))]
            sampler = torch.utils.data.WeightedRandomSampler(weights, len(dset))

        return DataLoader(
            dset, batch_size=args.per_device_batch_size,
            shuffle=(sampler is None), sampler=sampler,
            num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
            collate_fn=collator, pin_memory=True,
        )

    loader = _build_train_loader(dataset)

    # ── Optimizer ─────────────────────────────────────────────
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=args.learning_rate,
                                   betas=(args.adam_beta1, args.adam_beta2),
                                   weight_decay=args.weight_decay, eps=1e-6)

    # ── Scheduler ─────────────────────────────────────────────
    total_updates = max(1, math.ceil(args.max_steps / max(1, args.gradient_accumulation_steps)))
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=max(1, args.warmup_steps))
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_updates - args.warmup_steps),
                                eta_min=args.learning_rate * args.min_lr_ratio)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                              milestones=[max(1, args.warmup_steps)])

    # ── Training ──────────────────────────────────────────────
    print(f"Training: {args.max_steps} steps, GA={args.gradient_accumulation_steps}")
    model.train()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    accum_loss = 0.0
    recent_losses = []

    # Resume from checkpoint (load optimizer + scheduler state)
    resume_path = None
    if args.resume_from_checkpoint:
        resume_path = Path(args.resume_from_checkpoint)
    if resume_path and resume_path.exists():
        opt_path = resume_path / "optimizer.pt"
        sched_path = resume_path / "scheduler.pt"
        state_path = resume_path / "train_state.pt"
        if opt_path.exists():
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
            print(f"Loaded optimizer state from {opt_path}")
        if sched_path.exists():
            scheduler.load_state_dict(torch.load(sched_path, map_location=device))
            print(f"Loaded scheduler state from {sched_path}")
        if state_path.exists():
            ts = torch.load(state_path, map_location=device)
            step = ts.get("step", 0)
            accum_loss = ts.get("loss", 0.0)
            print(f"Resumed from step {step}")

    # Save train_setup
    setup = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (output_dir / "train_setup.json").write_text(json.dumps(setup, indent=2, ensure_ascii=False))

    eval_dir = output_dir / "eval"
    eval_dir.mkdir(exist_ok=True)

    nan_count = 0
    skipped_steps = 0
    epoch = 0

    # If resuming from a different training run (output_dir mismatch), rotate data.
    # Same-run resumes (e.g., after OOM crash) should NOT rotate — they're mid-epoch.
    if args.resume_from_checkpoint and step > 0:
        resume_dir = str(Path(args.resume_from_checkpoint).resolve())
        output_dir_str = str(output_dir.resolve())
        if not resume_dir.startswith(output_dir_str):
            dataset.rotate_balanced_samples()
            loader = _build_train_loader(dataset)
            print(f"  rotated dataset after cross-run resume (step={step}, {len(dataset)} samples)")
        else:
            print(f"  same-run resume, no rotation (step={step})")

    while step < args.max_steps:
        epoch += 1
        if epoch > 1:
            # Rotate category-balanced sliding windows to show different data each epoch
            dataset.rotate_balanced_samples()
            loader = _build_train_loader(dataset)
            print(f"  epoch {epoch}: {len(dataset)} samples, {len(loader)} batches")

        for batch in loader:
            if step >= args.max_steps:
                break

            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps

            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                skipped_steps += 1
                if nan_count <= 5 or nan_count % 50 == 0:
                    print(f"  step {step}: NaN loss detected, skipping (NaN count={nan_count})")
                if skipped_steps >= args.gradient_accumulation_steps:
                    optimizer.zero_grad()
                    accum_loss = 0.0
                    skipped_steps = 0
                step += 1
                continue

            loss.backward()
            accum_loss += loss.item()

            step += 1
            skipped_steps = 0
            recent_losses.append(loss.item() * args.gradient_accumulation_steps)
            if len(recent_losses) > 100:
                recent_losses.pop(0)

            if step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % 100 == 0:
                avg_loss = sum(recent_losses) / max(1, len(recent_losses))
                print(f"  step {step}/{args.max_steps} loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}")

            # Save checkpoint (weights + optimizer + scheduler)
            save_step = step
            if save_step % args.save_every_steps == 0:
                ckpt_dir = output_dir / f"checkpoint-{save_step}"
                ckpt_dir.mkdir(exist_ok=True)
                model.save_pretrained(str(ckpt_dir))
                torch.save(optimizer.state_dict(), str(ckpt_dir / "optimizer.pt"))
                torch.save(scheduler.state_dict(), str(ckpt_dir / "scheduler.pt"))
                torch.save({"step": step, "loss": accum_loss}, str(ckpt_dir / "train_state.pt"))
                wp = {"point_mlp": model.point_mlp.state_dict(),
                      "visual_proj": model.visual_proj.state_dict(),
                      "point_norm": model.point_norm.state_dict()}
                if model.coord_map_encoder is not None:
                    wp["coord_map_encoder"] = model.coord_map_encoder.state_dict()
                torch.save(wp, str(ckpt_dir / "wrapper_params.pt"))
                print(f"  saved checkpoint-{save_step} (weights+optim+scheduler+wrapper)")

            # Eval
            if step % args.eval_every_steps == 0 and step > 0 and args.val_jsonl:
                print(f"  eval at step {step}...")
                model.eval()
                try:
                    from src.eval.run_pointarena_eval import evaluate_model_on_val
                    metrics = evaluate_model_on_val(
                        model=model, processor=processor, val_jsonl=args.val_jsonl,
                        step=step, output_predictions_path=str(eval_dir / f"predictions_{step}.jsonl"),
                        fast_eval_subset=args.fast_eval_subset,
                        prompt_mode=args.eval_prompt_mode,
                        exclude_counting=False, require_single_point=False,
                    )
                    overall = metrics.get("overall", 0)
                    print(f"  eval overall: {overall:.2f}%")

                    leaderboard_path = output_dir / "leaderboard.json"
                    ckpt_path_str = str(output_dir / f"checkpoint-{step}")
                    update_leaderboard(leaderboard_path, step, metrics,
                                       ckpt_path_str,
                                       lr=scheduler.get_last_lr()[0],
                                       avg_loss=sum(recent_losses)/max(1,len(recent_losses)))
                except Exception as e:
                    print(f"  eval failed: {e}")
                model.train()

    print(f"Training complete at step {step}")
    output_dir_final = output_dir / f"checkpoint-{step}"
    output_dir_final.mkdir(exist_ok=True)
    model.save_pretrained(str(output_dir_final))
    return 0


if __name__ == "__main__":
    main()
