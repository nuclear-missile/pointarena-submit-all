from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from peft import LoraConfig, TaskType


DEFAULT_TARGET_MODULE_PATTERNS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    # Molmo2 / OLMo2 style names
    "wq",
    "wk",
    "wv",
    "wo",
    "w1",
    "w2",
    "w3",
    "att_proj",
    "attn_out",
    "ff_proj",
    "ff_out",
]


def classify_module_branch(module_name: str) -> str:
    """Map module path to a high-level branch used in ablations."""
    n = (module_name or "").strip().lower()
    if not n:
        return "other"

    if "vision_backbone.image_vit" in n:
        return "vision"
    if "vision_backbone.image_projector" in n:
        return "projector"
    if n.startswith("model.transformer.") or ".transformer.blocks." in n:
        return "llm"
    return "other"


def collect_lora_target_candidates(
    model: Any,
    target_patterns: list[str] | None = None,
) -> list[dict[str, Any]]:
    patterns = set(target_patterns or DEFAULT_TARGET_MODULE_PATTERNS)
    out: list[dict[str, Any]] = []

    for module_name, module in model.named_modules():
        leaf = module_name.split(".")[-1]
        if leaf not in patterns:
            continue

        weight = getattr(module, "weight", None)
        if weight is not None and hasattr(weight, "numel"):
            trainable_numel = int(weight.numel())
        else:
            # Fallback for custom modules without a direct `weight` field.
            try:
                trainable_numel = int(sum(int(p.numel()) for p in module.parameters(recurse=False)))
            except Exception:
                trainable_numel = 0

        out.append(
            {
                "module_name": module_name,
                "leaf_name": leaf,
                "branch": classify_module_branch(module_name),
                "trainable_numel": trainable_numel,
            }
        )

    out.sort(key=lambda x: x["module_name"])
    return out


def select_target_modules_by_scope(
    model: Any,
    *,
    target_scope: str = "vision+llm",
    target_patterns: list[str] | None = None,
    include_projector_in_llm_only: bool = True,
    include_projector_in_vision_only: bool = False,
) -> tuple[list[str], list[dict[str, Any]]]:
    scope = (target_scope or "vision+llm").strip().lower()
    if scope == "all":
        scope = "vision+llm"

    cands = collect_lora_target_candidates(model, target_patterns=target_patterns)

    if scope == "vision+llm":
        allowed_branches = {"vision", "projector", "llm"}
    elif scope == "llm_only":
        allowed_branches = {"llm"}
        if include_projector_in_llm_only:
            allowed_branches.add("projector")
    elif scope == "vision_only":
        allowed_branches = {"vision"}
        if include_projector_in_vision_only:
            allowed_branches.add("projector")
    else:
        raise ValueError(f"Unsupported target scope: {target_scope}")

    selected = [x for x in cands if x["branch"] in allowed_branches]
    selected_names = [x["module_name"] for x in selected]
    return selected_names, cands


@dataclass
class LoraArgs:
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: list[str] | None = None
    bias: str = "none"


def detect_target_modules(model) -> list[str]:
    names: set[str] = set()
    for n, _m in model.named_modules():
        leaf = n.split(".")[-1]
        if leaf in DEFAULT_TARGET_MODULE_PATTERNS:
            names.add(leaf)
    if not names:
        names = set(DEFAULT_TARGET_MODULE_PATTERNS)
    return sorted(names)


def build_lora_config(model, args: LoraArgs) -> LoraConfig:
    target_modules = args.target_modules if args.target_modules else detect_target_modules(model)
    return LoraConfig(
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=target_modules,
        bias=args.bias,
        task_type=TaskType.CAUSAL_LM,
    )
