"""
Architecture: Molmo2 with AttnRes (Attention Residual) blocks.

AttnRes wraps every N transformer layers with an additional residual
attention block. Configured via model config overrides:

  text_config.attnres_enabled = True
  text_config.attnres_layers_per_block = 4
  text_config.attnres_apply_pre_attn = True
  text_config.attnres_apply_pre_mlp = True
  text_config.attnres_gate_init = 0.0

This architecture is used exclusively for the Steerability expert.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def build_attnres_model(base_model_path: str, torch_dtype=torch.bfloat16):
    """
    Load Molmo2-8B with AttnRes architecture enabled.

    The AttnRes layers are part of the model structure (not LoRA).
    They must be enabled at config time before model construction.
    """
    from transformers import AutoConfig, AutoModelForImageTextToText

    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    text_config = getattr(config, "text_config", config)

    # Enable AttnRes architecture
    text_config.attnres_enabled = True
    text_config.attnres_layers_per_block = 4
    text_config.attnres_apply_pre_attn = True
    text_config.attnres_apply_pre_mlp = True
    text_config.attnres_gate_init = 0.0

    model = AutoModelForImageTextToText.from_pretrained(
        base_model_path,
        config=config,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    return model
