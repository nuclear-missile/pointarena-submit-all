"""
PointArena Submission — Shared backbone + multi-adapter LoRA routing.

2 backbones: standard Molmo2-8B (C, B) + AttnRes Molmo2-8B (counting, steer).
Each backbone has 2 LoRA adapters loaded simultaneously via PEFT add_adapter().
Category routing: set_adapter() + optional wrapper → inference.

Seed=42. Temperature=0.0. Deterministic.
"""
from __future__ import annotations

import os, random, gc
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

_SEED = 42

def _set_seeds():
    random.seed(_SEED); np.random.seed(_SEED)
    torch.manual_seed(_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

_set_seeds()

CATEGORY_TO_ADAPTER = {
    "affordance": "C", "counting": "C", "reasoning": "B",
    "spatial": "C", "steerability": "steer",
}

# Which backbone each adapter uses
ADAPTER_BACKBONE = {"C": "standard", "B": "standard", "counting": "attnres", "steer": "attnres"}


class PointArenaSubmission(nn.Module):
    def __init__(
        self,
        checkpoint_path: str,
        base_model_path: str,
        attnres_model_path: Optional[str] = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self._device_str = device
        self._dtype = torch_dtype
        self._processor = None
        self._base_model_path = base_model_path  # store for processor loading

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self._lora_states = ckpt["lora_adapters"]
        self._lora_configs = ckpt["lora_configs"]
        self._arch_states = ckpt["arch_modules"]
        self._meta = ckpt["meta"]

        # Build two shared backbones with multi-adapters
        self._backbones: Dict[str, nn.Module] = {}
        self._wrappers: Dict[str, nn.Module] = {}
        self._active_adapter: Optional[str] = None

        self._build_backbone("standard", base_model_path, ["C", "B"])
        self._build_backbone("attnres", attnres_model_path or base_model_path, ["counting", "steer"])

    @property
    def processor(self):
        if self._processor is None:
            from transformers import AutoProcessor
            self._processor = AutoProcessor.from_pretrained(
                self.base_model_path, trust_remote_code=True)
        return self._processor

    @property
    def device(self):
        return torch.device(self._device_str)

    def _build_backbone(self, name: str, model_path: str, adapters: list):
        """Build one backbone with multiple LoRA adapters."""
        from transformers import AutoModelForImageTextToText, AutoConfig
        from peft import LoraConfig, get_peft_model

        is_attnres = (name == "attnres")
        if is_attnres:
            cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            tc = getattr(cfg, "text_config", cfg)
            tc.attnres_enabled = True
            tc.attnres_layers_per_block = 4
            tc.attnres_apply_pre_attn = True
            tc.attnres_apply_pre_mlp = True
            tc.attnres_gate_init = 0.0
            base = AutoModelForImageTextToText.from_pretrained(
                model_path, config=cfg, torch_dtype=self._dtype,
                trust_remote_code=True).to(self.device)
        else:
            base = AutoModelForImageTextToText.from_pretrained(
                model_path, torch_dtype=self._dtype,
                trust_remote_code=True).to(self.device)

        # First adapter via get_peft_model (creates "default" adapter)
        first = adapters[0]
        lora_cfg = LoraConfig(**self._lora_configs[first])
        model = get_peft_model(base, lora_cfg)
        self._load_lora(model, self._lora_states[first])

        # Additional adapters via add_adapter
        for adp in adapters[1:]:
            lora_cfg = LoraConfig(**self._lora_configs[adp])
            model.add_adapter(adp, lora_cfg)
            model.set_adapter(adp)
            self._load_lora(model, self._lora_states[adp])

        model.set_adapter("default")  # reset
        model.eval()
        self._backbones[name] = model

    def _load_lora(self, model, lora_state):
        ms = model.state_dict()
        matched = 0
        for k, v in lora_state.items():
            if k in ms:
                ms[k].copy_(v.to(ms[k].dtype))
                matched += 1
        return matched

    def set_adapter(self, adapter_name: str):
        """Switch active LoRA adapter (no model reload needed)."""
        if self._active_adapter == adapter_name:
            return
        backbone_name = ADAPTER_BACKBONE[adapter_name]
        model = self._backbones[backbone_name]
        adp = "default" if adapter_name == list(ADAPTER_BACKBONE.keys())[0] or adapter_name not in ["C"] else adapter_name
        # Map adapter names to PEFT names: "C" is "default" (first added), others by name
        peft_name = "default" if adapter_name == "C" else adapter_name
        model.set_adapter(peft_name)
        self._active_adapter = adapter_name
        self._active_model = model

    @torch.no_grad()
    def generate(
        self,
        question: str,
        image: Any,
        metadata: Optional[dict] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        category = self._detect_category(question, metadata)
        adapter = CATEGORY_TO_ADAPTER.get(category, "B")
        self.set_adapter(adapter)

        from PIL import Image
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]}]
        inp = self.processor.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt")
        inp = {k: v.to(self.device) for k, v in inp.items()}

        with torch.amp.autocast("cuda", dtype=self._dtype):
            gen = self._active_model.generate(
                **inp, max_new_tokens=max_new_tokens, do_sample=(temperature > 0.0))
        tok = gen[0, inp["input_ids"].size(1):]
        return self.processor.tokenizer.decode(tok, skip_special_tokens=True)

    def _detect_category(self, question: str, metadata: Optional[dict] = None) -> str:
        if metadata:
            anchor = metadata.get("anchor_norm01")
            if anchor and isinstance(anchor, (list, tuple)) and len(anchor) == 2:
                return "steerability"
        q = (question or "").lower()
        for kw in ["how many", "count the", "number of", "total number"]:
            if kw in q: return "counting"
        for kw in ["what can", "suitable for", "can be used"]:
            if kw in q: return "affordance"
        for kw in ["left of", "right of", "above", "below", "behind",
                    "in front of", "next to", "closest", "nearest",
                    "farthest", "where is", "between"]:
            if kw in q: return "spatial"
        return "reasoning"

    @property
    def base_model_path(self):
        # Use the path passed at init, not from meta
        return self._base_model_path
