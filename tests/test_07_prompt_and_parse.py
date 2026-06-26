from __future__ import annotations

from src.train.prompting import build_prompt, build_target
from src.train.parse_output import parse_point_from_text, parse_points_from_text


def test_prompt_contains_query_and_single_format_default():
    p = build_prompt("Point to the mug")
    assert "Query Region:" in p
    assert "integer coordinates in [0,1000]" in p
    assert "<point>x,y</point>" in p
    assert "exactly one point" in p


def test_prompt_multi_mode_and_target_modes():
    p_multi = build_prompt("Point to all apples", single_point_only=False)
    assert "<points>x1,y1; x2,y2; ...</points>" in p_multi

    assert build_target([(0.1234, 0.5678)]) == "<point>123,568</point>"
    # single-point default keeps only first point
    assert build_target([(0.1, 0.2), (0.3, 0.4)]) == "<point>100,200</point>"
    # explicit multi-point mode remains available (for eval compatibility)
    assert build_target([(0.1, 0.2), (0.3, 0.4)], single_point_only=False) == "<points>100,200; 300,400</points>"


def test_parse_point_tag_integer_1000_space():
    # On 100x100 image, (120,340) in 0-1000 space -> (12,34) px
    assert parse_point_from_text("<point>120,340</point>", 100, 100) == (12, 34)


def test_parse_point_tag_float_rounding_norm_legacy():
    # Legacy normalized output support: (0.127, 0.342) -> around (13,34)
    assert parse_point_from_text("<point>0.127,0.342</point>", 100, 100) == (13, 34)


def test_parse_clipping_abs_fallback():
    assert parse_point_from_text("<point>9999,-10</point>", 20, 30) == (19, 0)


def test_parse_with_extra_text_and_multi_points():
    txt = "I think answer is <points>100,200; 300,400</points> maybe"
    pts = parse_points_from_text(txt, 100, 100)
    assert pts[0] == (10, 20)
    assert pts[1] == (30, 40)
