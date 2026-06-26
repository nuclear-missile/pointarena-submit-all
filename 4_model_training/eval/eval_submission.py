#!/usr/bin/env python3
"""
PointArena Submission Evaluation.

Official data directory, original sample order, CSV-based steerability detection.

Usage:
  python3 eval_submission.py --data-dir DATA_DIR [--checkpoint CKPT] [--gpu N] [--seed S]
"""
import argparse, csv, gc, json, os, random, sys

import numpy as np
import safetensors.torch as st
import torch
from PIL import Image

# Add repo root to path for local parse_output
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.parse_output import parse_points_from_text


def _load_mask(mask_path):
    p = os.path.normpath(mask_path)
    if not os.path.exists(p) or not os.path.isfile(p):
        return None
    try:
        return np.array(Image.open(p).convert("L")) > 0
    except Exception:
        return None


def _point_in_mask(mask, x, y):
    if mask is None:
        return False
    h, w = mask.shape[:2]
    if x < 0 or y < 0 or x >= w or y >= h:
        return False
    return bool(mask[y, x])


def _expected_count(row, target):
    meta = row.get("meta", {}) or {}
    try:
        return max(int(meta.get("count", len(target.get("points", [])))), 0)
    except (ValueError, TypeError):
        return max(len(target.get("points", [])), 0)


CATEGORY_EXPERT = {
    "affordance": "C", "counting": "C", "reasoning": "B",
    "spatial": "C", "steerability": "steer",
}
EXPECTED = {
    "affordance": 94.44, "counting": 70.41, "reasoning": 81.35,
    "spatial": 82.56, "steerability": 63.50,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--checkpoint", default="./checkpoints/checkpoint.pt")
    parser.add_argument("--base-model", default="allenai/Molmo2-8B")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="./results")
    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_dir = os.path.abspath(args.data_dir)
    val_jsonl = os.path.join(data_dir, "val_pointarena.jsonl")
    pixmo_csv = os.path.join(data_dir, "pixmo_metadata.csv")
    ckpt_path = os.path.abspath(args.checkpoint)
    device = torch.device(f"cuda:{args.gpu}")
    dtype = torch.bfloat16

    # Load steerable image index from CSV (official data directory)
    steerable_images = set()
    for p in [pixmo_csv]:
        if os.path.exists(p):
            with open(p) as f:
                for row in csv.DictReader(f):
                    steerable_images.add(row["image_filename"])
            print(f"Steerable index: {len(steerable_images)} entries from {p}", flush=True)
            break

    # Load checkpoint
    print(f"Checkpoint: {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Load processor
    from transformers import AutoProcessor
    print(f"Processor: {args.base_model}", flush=True)
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)

    # Load val data (OFFICIAL JSONL, unmodified, original order)
    with open(val_jsonl) as f:
        all_rows = [json.loads(l) for l in f if l.strip()]
    print(f"Val samples: {len(all_rows)} (original order)", flush=True)

    # Build expert cache (single active model)
    from transformers import AutoModelForImageTextToText, AutoConfig
    from peft import PeftModel

    _expert_model = None
    _expert_name = None

    def load_expert(adp):
        nonlocal _expert_model, _expert_name
        if _expert_name == adp and _expert_model is not None:
            return _expert_model
        if _expert_model is not None:
            del _expert_model; gc.collect(); torch.cuda.empty_cache()

        d = f"/tmp/adp_{adp}_s{args.seed}"
        os.makedirs(d, exist_ok=True)
        st.save_file(ckpt["lora_adapters"][adp], f"{d}/adapter_model.safetensors")
        with open(f"{d}/adapter_config.json", "w") as fc:
            json.dump(ckpt["lora_configs"][adp], fc)

        if adp == "steer":
            cfg = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
            tc = getattr(cfg, "text_config", cfg)
            tc.attnres_enabled = True; tc.attnres_layers_per_block = 4
            tc.attnres_apply_pre_attn = True; tc.attnres_apply_pre_mlp = True
            tc.attnres_gate_init = 0.0
            base = AutoModelForImageTextToText.from_pretrained(
                args.base_model, config=cfg, torch_dtype=dtype, trust_remote_code=True).to(device)
        else:
            base = AutoModelForImageTextToText.from_pretrained(
                args.base_model, torch_dtype=dtype, trust_remote_code=True).to(device)

        m = PeftModel.from_pretrained(base, d, is_trainable=False).to(device)
        m.eval()
        _expert_model = m; _expert_name = adp
        return m

    # Evaluate in ORIGINAL order
    total = correct = invalid = 0
    stat = {c: {"n": 0, "ok": 0} for c in EXPECTED}
    os.makedirs(args.output, exist_ok=True)

    for idx, row in enumerate(all_rows):
        image_path = str(row.get("image_path", ""))
        query = str(row.get("query", ""))
        target = row.get("target", {}) or {}
        image_size = target.get("image_size", [0, 0])
        w, h = int(image_size[0]), int(image_size[1])
        meta = row.get("meta", {}) or {}
        expected_count = _expected_count(row, target)
        gt_mask_path = str(meta.get("gt_mask_path", ""))
        category = str(meta.get("category", "")).strip().lower()

        if category == "affordable":
            category = "affordance"
        elif category == "steerable":
            category = "steerability"

        # Steerability detection FROM CSV ONLY (no embedded anchor in JSONL)
        img_fname = os.path.basename(image_path)
        if steerable_images and img_fname in steerable_images:
            category = "steerability"

        adp = CATEGORY_EXPERT.get(category, "B")

        if w <= 0 or h <= 0:
            try:
                with Image.open(image_path) as im: w, h = im.size
            except Exception: w, h = 1, 1

        prompt = (query or "").strip() or "Point to the target."

        with Image.open(image_path).convert("RGB") as im:
            messages = [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": im.copy()},
            ]}]

        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        model = load_expert(adp)
        total += 1
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)

        gen_tokens = generated_ids[0, inputs["input_ids"].size(1):]
        text = processor.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        parsed_points = parse_points_from_text(text, w, h)

        parsed_ok = bool(parsed_points)
        if not parsed_ok:
            invalid += 1
            parsed_points = [(max(w // 2, 0), max(h // 2, 0))]

        points_for_scoring = parsed_points
        if category != "counting":
            points_for_scoring = [parsed_points[0]]

        count_ok = True
        if category == "counting":
            count_ok = len(points_for_scoring) == max(expected_count, 0)

        mask = _load_mask(gt_mask_path)
        all_in_mask = all(
            _point_in_mask(mask, p[0], p[1]) for p in points_for_scoring)
        ok = bool(count_ok and all_in_mask)
        if ok:
            correct += 1

        if category in stat:
            stat[category]["n"] += 1
            stat[category]["ok"] += int(ok)

        if (idx + 1) % 100 == 0:
            print(f"  [{idx+1}/{len(all_rows)}] {category} "
                  f"acc={100.0*stat[category]['ok']/stat[category]['n']:.2f}% "
                  f"overall={100.0*correct/total:.2f}%", flush=True)

    # ── Results ──
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS (seed={args.seed})")
    print(f"{'='*60}")
    print(f"Total: {total}  Correct: {correct}  Invalid: {invalid}")
    if total:
        print(f"Overall: {100.0*correct/total:.5f}% = {correct}/{total}")
    for cat in ["affordance", "counting", "reasoning", "spatial", "steerability"]:
        s = stat.get(cat, {"n": 0, "ok": 0})
        if s["n"]:
            acc = 100.0 * s["ok"] / s["n"]
            exp = EXPECTED.get(cat, 0.0)
            flag = "MATCH" if abs(acc - exp) < 2.0 else "DIFF"
            print(f"  {cat}: {s['ok']}/{s['n']} = {acc:.2f}% (exp {exp:.2f}%) {flag}")

    metrics_path = os.path.join(args.output, f"metrics_seed{args.seed}.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "seed": args.seed,
            "overall": round(100.0 * correct / total, 5) if total else 0,
            "correct": correct, "total": total, "invalid": invalid,
            "subtasks": {
                c: {"n": s["n"], "ok": s["ok"],
                    "accuracy": round(100.0 * s["ok"] / s["n"], 2) if s["n"] else 0}
                for c, s in stat.items()},
        }, f, indent=2)
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
