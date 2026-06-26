from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForImageTextToText


MODEL_DIR = "/mnt/data/lv_qi/xing/pointarena/models/allenai/Molmo2-8B-attnres"


def _tiny_molmo2_cfg(*, attnres_enabled: bool):
    cfg = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
    cfg.vit_config = None
    cfg.adapter_config = None

    text = cfg.text_config
    text.hidden_size = 32
    text.num_attention_heads = 4
    text.num_key_value_heads = 2
    text.head_dim = 8
    text.intermediate_size = 64
    text.num_hidden_layers = 2
    text.vocab_size = 128
    text.additional_vocab_size = 0
    text.max_position_embeddings = 64
    text.use_cache = False
    text.attn_implementation = "eager"
    text.use_qk_norm = False
    text.rope_scaling = {"rope_type": "linear", "factor": 1.0}

    text.attnres_enabled = bool(attnres_enabled)
    text.attnres_layers_per_block = 1
    text.attnres_apply_pre_attn = True
    text.attnres_apply_pre_mlp = True
    text.attnres_gate_init = 0.0
    return cfg


def test_attnres_modules_to_save_list() -> None:
    from filtered_data_ft.run_lora_train_filtered_steerable_d_attnres import build_attnres_modules_to_save

    modules = build_attnres_modules_to_save(
        attnres_enabled=True,
        apply_pre_attn=True,
        apply_pre_mlp=True,
    )
    assert modules == [
        "attn_res_proj",
        "attn_res_norm",
        "attn_res_gate",
        "mlp_res_proj",
        "mlp_res_norm",
        "mlp_res_gate",
    ]


def test_attnres_zero_gate_preserves_base_logits() -> None:
    torch.manual_seed(0)

    base_cfg = _tiny_molmo2_cfg(attnres_enabled=False)
    attnres_cfg = copy.deepcopy(base_cfg)
    attnres_cfg.text_config.attnres_enabled = True

    base_model = AutoModelForImageTextToText.from_config(base_cfg, trust_remote_code=True).eval()
    attnres_model = AutoModelForImageTextToText.from_config(attnres_cfg, trust_remote_code=True).eval()

    incompatible = attnres_model.load_state_dict(base_model.state_dict(), strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = [key for key in incompatible.missing_keys if "attn_res_" not in key and "mlp_res_" not in key]
    assert unexpected == []
    assert missing == []

    input_ids = torch.randint(low=0, high=64, size=(2, 6), dtype=torch.long)
    with torch.no_grad():
        base_logits = base_model(input_ids=input_ids, use_cache=False).logits
        attnres_logits = attnres_model(input_ids=input_ids, use_cache=False).logits

    torch.testing.assert_close(base_logits, attnres_logits, atol=1e-6, rtol=1e-6)


def test_attnres_train_smoke_mock(tmp_path: Path) -> None:
    val_jsonl = tmp_path / "val.jsonl"
    val_jsonl.write_text("{}\n", encoding="utf-8")
    out_dir = tmp_path / "run"

    cmd = [
        sys.executable,
        "-m",
        "filtered_data_ft.run_lora_train_filtered_steerable_d_attnres",
        "--model_path",
        MODEL_DIR,
        "--val_jsonl",
        str(val_jsonl),
        "--output_dir",
        str(out_dir),
        "--max_steps",
        "2",
        "--save_every_steps",
        "1",
        "--eval_every_steps",
        "1",
        "--attnres_layers_per_block",
        "2",
        "--attnres_gate_init",
        "0.0",
        "--mock_train",
    ]
    subprocess.run(cmd, check=True)

    assert (out_dir / "checkpoint-1" / "train_state.json").exists()
