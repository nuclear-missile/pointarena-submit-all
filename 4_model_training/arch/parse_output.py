from __future__ import annotations

import re
from typing import Iterable

POINT_TAG_RE = re.compile(r"<point>\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*</point>", re.IGNORECASE)
POINTS_TEXT_RE = re.compile(r"<points>\s*(.*?)\s*</points>", re.IGNORECASE | re.DOTALL)
POINT_PAIR_RE = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)")

# Molmo coords tag parser (legacy protocol)
MOLMO_COORDS_RE = re.compile(r"<(?:points|tracks)\b[^>]*\bcoords=\"([0-9\t:;, .]+)\"[^>]*>", re.IGNORECASE)
MOLMO_POINT_RE = re.compile(r"([0-9]{1,3})\s+([0-9]{1,4}(?:\.[0-9]+)?)\s+([0-9]{1,4}(?:\.[0-9]+)?)")
MOLMO_FRAME_RE = re.compile(r"(?:^|[\t:;,])([0-9]+(?:\.[0-9]+)?)\s+([0-9.\s]+)")

# V3 strict no-extra-text mode:
# whole output must be one empty-body points tag, optionally self-closing.
MOLMO_V3_WHOLE_RE = re.compile(
    r'^\s*<points\b[^>]*\bcoords="([0-9\t:;, .]+)"[^>]*>\s*</points>\s*$',
    re.IGNORECASE | re.DOTALL,
)
MOLMO_V3_SELF_RE = re.compile(
    r'^\s*<points\b[^>]*\bcoords="([0-9\t:;, .]+)"[^>]*/>\s*$',
    re.IGNORECASE | re.DOTALL,
)


def _clip_xy(x: float, y: float, w: int, h: int) -> tuple[int, int]:
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    xi = max(0, min(max(w - 1, 0), xi))
    yi = max(0, min(max(h - 1, 0), yi))
    return xi, yi


def _norm_to_pixel(xn: float, yn: float, w: int, h: int) -> tuple[int, int]:
    den_w = float(max(w - 1, 1))
    den_h = float(max(h - 1, 1))
    x = xn * den_w
    y = yn * den_h
    return _clip_xy(x, y, w, h)


def _try_parse_to_pixel(
    x: float,
    y: float,
    image_w: int,
    image_h: int,
    *,
    allow_abs_fallback: bool,
) -> tuple[int, int] | None:
    # Backward compatibility for legacy normalized outputs like 0.123,0.456.
    x_is_int = abs(x - round(x)) < 1e-6
    y_is_int = abs(y - round(y)) < 1e-6
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and (not x_is_int or not y_is_int):
        return _norm_to_pixel(x, y, image_w, image_h)

    # Canonical format: integer [0,1000] -> normalize to [0,1] then map to pixels.
    if 0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0:
        return _norm_to_pixel(x / 1000.0, y / 1000.0, image_w, image_h)

    # Backward compatibility for edge normalized values exactly 0 or 1.
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        return _norm_to_pixel(x, y, image_w, image_h)

    # Very old percent-style compatibility.
    if 0.0 <= x <= 100.0 and 0.0 <= y <= 100.0:
        return _norm_to_pixel(x / 100.0, y / 100.0, image_w, image_h)

    if allow_abs_fallback:
        return _clip_xy(x, y, image_w, image_h)
    return None


def _iter_molmo_xy_from_coord_block(block: str):
    has_frame_prefix = False
    for frame_match in MOLMO_FRAME_RE.finditer(block):
        has_frame_prefix = True
        coord_body = frame_match.group(2)
        for p in MOLMO_POINT_RE.finditer(coord_body):
            yield float(p.group(2)), float(p.group(3))

    if has_frame_prefix:
        return

    # Backward compatibility: accept coords blocks without explicit frame prefix.
    for p in MOLMO_POINT_RE.finditer(block):
        yield float(p.group(2)), float(p.group(3))


def parse_points_from_text(text: str, image_w: int, image_h: int, max_points: int | None = None) -> list[tuple[int, int]]:
    text = text or ""
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def _push(parsed: tuple[int, int] | None) -> None:
        if parsed is None:
            return
        if parsed in seen:
            return
        seen.add(parsed)
        out.append(parsed)

    # 1) Molmo2 legacy style XML coords blocks.
    for m0 in MOLMO_COORDS_RE.finditer(text):
        block = m0.group(1)
        for x, y in _iter_molmo_xy_from_coord_block(block):
            _push(_try_parse_to_pixel(x, y, image_w, image_h, allow_abs_fallback=False))
            if max_points is not None and len(out) >= max_points:
                return out

    # 2) New multi-point target format: <points>x1,y1; x2,y2; ...</points>
    for m_points in POINTS_TEXT_RE.finditer(text):
        body = m_points.group(1)
        for pair in POINT_PAIR_RE.finditer(body):
            x = float(pair.group(1))
            y = float(pair.group(2))
            _push(_try_parse_to_pixel(x, y, image_w, image_h, allow_abs_fallback=True))
            if max_points is not None and len(out) >= max_points:
                return out

    # 3) Single-point target format.
    for m in POINT_TAG_RE.finditer(text):
        x = float(m.group(1))
        y = float(m.group(2))
        _push(_try_parse_to_pixel(x, y, image_w, image_h, allow_abs_fallback=True))
        if max_points is not None and len(out) >= max_points:
            return out

    # 4) Strict fallback to avoid hallucinated numeric text.
    if not out:
        for m2 in POINT_PAIR_RE.finditer(text):
            x = float(m2.group(1))
            y = float(m2.group(2))
            _push(_try_parse_to_pixel(x, y, image_w, image_h, allow_abs_fallback=False))
            if max_points is not None and len(out) >= max_points:
                return out
            if len(out) >= 16:
                break

    return out


def parse_points_from_text_v3(text: str, image_w: int, image_h: int, max_points: int | None = None) -> list[tuple[int, int]]:
    """V3 parser: legacy Molmo coords + normalization + no-extra-text strictness.

    Constraints for valid parsing:
    - Output must be exactly one <points coords="..."></points> tag (or self-closing form).
    - No extra text before/after tag.
    - Tag body must be empty.
    - Coords are parsed with old Molmo triplet protocol and normalized from [0,1000].
    """
    text = text or ""

    m = MOLMO_V3_WHOLE_RE.match(text)
    if m is None:
        m = MOLMO_V3_SELF_RE.match(text)
    if m is None:
        return []

    block = m.group(1)
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    for x, y in _iter_molmo_xy_from_coord_block(block):
        parsed = _try_parse_to_pixel(x, y, image_w, image_h, allow_abs_fallback=False)
        if parsed is None:
            continue
        if parsed in seen:
            continue
        seen.add(parsed)
        out.append(parsed)
        if max_points is not None and len(out) >= max_points:
            break

    return out


def parse_point_from_text(text: str, image_w: int, image_h: int) -> tuple[int, int] | None:
    pts = parse_points_from_text(text, image_w, image_h, max_points=1)
    return pts[0] if pts else None


def parse_point_from_text_v3(text: str, image_w: int, image_h: int) -> tuple[int, int] | None:
    pts = parse_points_from_text_v3(text, image_w, image_h, max_points=1)
    return pts[0] if pts else None


def parse_first_valid(texts: Iterable[str], image_w: int, image_h: int) -> tuple[int, int] | None:
    for t in texts:
        p = parse_point_from_text(t, image_w, image_h)
        if p is not None:
            return p
    return None
