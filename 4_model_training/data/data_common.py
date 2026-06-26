from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            yield json.loads(ln)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_image_size(path: str | Path) -> tuple[int, int]:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return (0, 0)
    try:
        with Image.open(p) as im:
            return int(im.size[0]), int(im.size[1])
    except Exception:
        return (0, 0)


def clip_point_xy(x: float, y: float, w: int, h: int) -> tuple[float, float]:
    if w <= 0 or h <= 0:
        return (0.0, 0.0)
    x = max(0.0, min(float(w - 1), float(x)))
    y = max(0.0, min(float(h - 1), float(y)))
    return (x, y)


def convert_points_to_abs(points: list[dict[str, Any]], w: int, h: int) -> list[list[float]]:
    out: list[list[float]] = []
    for p in points or []:
        try:
            x = float(p["x"])
            y = float(p["y"])
        except Exception:
            continue

        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            x = x * w
            y = y * h
        elif x <= 100.0 and y <= 100.0 and (w > 100 or h > 100):
            x = (x / 100.0) * w
            y = (y / 100.0) * h

        x, y = clip_point_xy(x, y, w, h)
        out.append([float(x), float(y)])
    return out


def convert_points_to_norm(
    points: list[dict[str, Any]],
    w: int,
    h: int,
    decimals: int = 3,
) -> list[list[float]]:
    out: list[list[float]] = []
    if w <= 0 or h <= 0:
        return out
    den_w = float(max(w - 1, 1))
    den_h = float(max(h - 1, 1))

    for p in points or []:
        try:
            x = float(p["x"])
            y = float(p["y"])
        except Exception:
            continue

        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            xn, yn = x, y
        elif 0.0 <= x <= 100.0 and 0.0 <= y <= 100.0 and (w > 100 or h > 100):
            xn, yn = x / 100.0, y / 100.0
        else:
            xn, yn = x / den_w, y / den_h

        xn = max(0.0, min(1.0, float(xn)))
        yn = max(0.0, min(1.0, float(yn)))
        out.append([round(xn, int(decimals)), round(yn, int(decimals))])
    return out


def pixel_to_norm_xy(x: float, y: float, w: int, h: int, decimals: int = 3) -> tuple[float, float]:
    den_w = float(max(w - 1, 1))
    den_h = float(max(h - 1, 1))
    xn = max(0.0, min(1.0, float(x) / den_w))
    yn = max(0.0, min(1.0, float(y) / den_h))
    return (round(xn, int(decimals)), round(yn, int(decimals)))


def sample_points_from_mask(
    mask_path: str | Path,
    num_points: int,
    seed: str | int = 0,
) -> list[tuple[float, float]]:
    if int(num_points) <= 0:
        return []
    p = Path(mask_path)
    if not p.exists() or not p.is_file():
        return []
    try:
        arr = np.array(Image.open(p).convert("L")) > 0
    except Exception:
        return []
    ys, xs = np.where(arr)
    n = len(xs)
    if n <= 0:
        return []

    rng = random.Random(str(seed))
    k = int(num_points)
    if n >= k:
        idxs = rng.sample(range(n), k)
    else:
        idxs = [rng.randrange(n) for _ in range(k)]

    out: list[tuple[float, float]] = []
    for i in idxs:
        out.append((float(xs[i]), float(ys[i])))
    return out


COORD_TUPLE_RE = re.compile(
    r"(?P<br_open>[\(\[])(?P<space1>\s*)(?P<x>[+-]?\d+(?:\.\d+)?)(?P<space2>\s*),(?P<space3>\s*)(?P<y>[+-]?\d+(?:\.\d+)?)(?P<space4>\s*)(?P<br_close>[\)\]])"
)


def _normalize_maybe_coord(x: float, y: float, w: int, h: int) -> tuple[float, float]:
    den_w = float(max(int(w) - 1, 1))
    den_h = float(max(int(h) - 1, 1))

    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        xn, yn = x, y
    elif 0.0 <= x <= 100.0 and 0.0 <= y <= 100.0 and (w > 100 or h > 100):
        xn, yn = x / 100.0, y / 100.0
    elif 0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0:
        xn, yn = x / 1000.0, y / 1000.0
    else:
        xn, yn = x / den_w, y / den_h

    xn = max(0.0, min(1.0, float(xn)))
    yn = max(0.0, min(1.0, float(yn)))
    return (xn, yn)


def normalize_coords_in_query(
    query: str,
    w: int,
    h: int,
    decimals: int = 3,
) -> str:
    text = str(query or "")
    if not text:
        return text

    def _repl(m: re.Match[str]) -> str:
        try:
            x = float(m.group("x"))
            y = float(m.group("y"))
        except Exception:
            return m.group(0)
        xn, yn = _normalize_maybe_coord(x, y, int(w), int(h))
        x1000 = int(round(max(0.0, min(1000.0, xn * 1000.0))))
        y1000 = int(round(max(0.0, min(1000.0, yn * 1000.0))))
        return f"{m.group('br_open')}{x1000}, {y1000}{m.group('br_close')}"

    return COORD_TUPLE_RE.sub(_repl, text)


def infer_task_from_query(query: str, default: str = "unknown") -> str:
    q = (query or "").lower()
    if any(k in q for k in ["vacant", "free space", "empty space", "empty area"]):
        return "free_space_reference"
    if any(k in q for k in ["left", "right", "between", "behind", "in front", "closest", "farthest"]):
        return "spatial_relation"
    if any(k in q for k in ["how many", "count", "all "]):
        return "counting"
    if any(k in q for k in ["tool", "used for", "can be used", "afford"]):
        return "affordance"
    if any(k in q for k in ["reason", "why", "best", "likely", "safest"]):
        return "reasoning"
    if "point" in q:
        return "object_reference"
    return default
