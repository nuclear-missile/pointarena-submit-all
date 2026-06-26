from __future__ import annotations

import pytest
import torch

from src.train.parse_output_singlepos import parse_points_from_text_singlepos
from src.train.singlepos import (
    SINGLEPOS_TOKEN_PREFIX,
    SinglePosInputEmbedding,
    SinglePosLMHead,
    build_singlepos_coords_attr,
    decode_singlepos_token,
    encode_singlepos_value,
)


def test_encode_decode_singlepos_token_roundtrip() -> None:
    assert SINGLEPOS_TOKEN_PREFIX == "POS_"
    assert encode_singlepos_value(0) == "POS_0000"
    assert encode_singlepos_value(475) == "POS_0475"
    assert encode_singlepos_value(1000) == "POS_1000"
    assert decode_singlepos_token("POS_0475") == 475
    assert decode_singlepos_token("POS_1000") == 1000


def test_build_singlepos_coords_attr_uses_count_then_xy_tokens() -> None:
    coords = build_singlepos_coords_attr(
        points=[(0.475, 0.612), (0.632, 0.069)],
        single_point_only=False,
    )
    assert coords == "2 POS_0475 POS_0612 POS_0632 POS_0069"


def test_build_singlepos_coords_attr_scales_absolute_pixels() -> None:
    coords = build_singlepos_coords_attr(
        points=[(32, 16)],
        single_point_only=False,
        scale=[64, 64],
    )
    assert coords == "1 POS_0500 POS_0250"


def test_parse_points_from_text_singlepos_maps_back_to_pixels() -> None:
    text = '<points coords="2 POS_0475 POS_0612 POS_0632 POS_0069">target</points>'
    pts = parse_points_from_text_singlepos(text, image_w=1001, image_h=1001)
    assert pts == [(475, 612), (632, 69)]


def test_parse_points_from_text_singlepos_accepts_mixed_body_text() -> None:
    text = 'Counting the <points coords="1 POS_0500 POS_0250">target</points> shows a total of 1.'
    pts = parse_points_from_text_singlepos(text, image_w=1001, image_h=1001)
    assert pts == [(500, 250)]


def test_singlepos_input_embedding_avoids_full_vocab_cat(monkeypatch: pytest.MonkeyPatch) -> None:
    module = SinglePosInputEmbedding(
        base_weight=torch.randn(4, 3),
        extra_rows=2,
        initializer_range=0.02,
    )
    input_ids = torch.tensor([[0, 4, 1, 5]])

    def fail_cat(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("singlepos input embedding should not materialize full vocab with torch.cat")

    monkeypatch.setattr(torch, "cat", fail_cat)
    out = module(input_ids)

    assert out.shape == (1, 4, 3)


def test_singlepos_lm_head_avoids_full_weight_cat(monkeypatch: pytest.MonkeyPatch) -> None:
    module = SinglePosLMHead(
        base_weight=torch.randn(4, 3),
        extra_rows=2,
        initializer_range=0.02,
    )
    hidden_states = torch.randn(1, 2, 3)
    original_cat = torch.cat

    def fail_cat(*args: object, **kwargs: object) -> torch.Tensor:
        tensors = list(args[0]) if args else []
        if tensors and all(isinstance(t, torch.Tensor) and t.ndim == 2 for t in tensors):
            raise AssertionError("singlepos lm head should not materialize full output weight with torch.cat")
        return original_cat(*args, **kwargs)

    monkeypatch.setattr(torch, "cat", fail_cat)
    logits = module(hidden_states)

    assert logits.shape == (1, 2, 6)
