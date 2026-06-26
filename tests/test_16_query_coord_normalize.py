from __future__ import annotations

from src.data._common import normalize_coords_in_query


def test_query_coord_tuple_keeps_norm_and_to_1000_int():
    q = "Which point is better, (0.208,0.233) or (0.883, 0.511)?"
    out = normalize_coords_in_query(q, w=1000, h=1000, decimals=3)
    assert "(208, 233)" in out
    assert "(883, 511)" in out


def test_query_coord_tuple_percent_to_1000_int():
    q = "Use point (20, 35) as reference."
    out = normalize_coords_in_query(q, w=1280, h=720, decimals=3)
    assert "(200, 350)" in out


def test_query_coord_tuple_abs_to_1000_int_and_clamp():
    q = "Pick point (2000, -15)."
    out = normalize_coords_in_query(q, w=1000, h=500, decimals=3)
    assert "(1000, 0)" in out
