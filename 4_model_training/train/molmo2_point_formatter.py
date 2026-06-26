from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class Molmo2PointFormatter:
    """Lightweight Molmo2 point formatter for image pointing tasks.

    This keeps the same key conventions as Molmo2's unified formatter:
    - coordinate scale fixed at 1000
    - <points ... coords="1 id x y ...">label</points>
    - mode `pointing` returns only the points tag.
    """

    coordinate_scale: str = "1000"

    def _scale_point(self, point: tuple[float, float], scale: tuple[float, float] | list[float] | float) -> tuple[int, int]:
        if isinstance(scale, (tuple, list)):
            sx, sy = float(scale[0]), float(scale[1])
        else:
            sx = float(scale)
            sy = float(scale)

        sx = max(sx, 1.0)
        sy = max(sy, 1.0)

        x = max(0.0, min(1.0, float(point[0]) / sx))
        y = max(0.0, min(1.0, float(point[1]) / sy))

        if self.coordinate_scale != "1000":
            raise NotImplementedError(self.coordinate_scale)
        return int(round(1000.0 * x)), int(round(1000.0 * y))

    def _format_single_image_coordinates(
        self,
        points: Iterable[tuple[float, float]],
        scale: tuple[float, float] | list[float] | float,
    ) -> str:
        scaled = [self._scale_point(p, scale) for p in points]
        # Match official formatter behavior: deterministic xy ordering.
        scaled = sorted(scaled, key=lambda p: (p[0], p[1]))
        chunks: list[str] = []
        for i, (x, y) in enumerate(scaled, start=1):
            chunks.append(f"{i} {x:03d} {y:03d}")
        return " ".join(chunks)

    @staticmethod
    def _build_point_str(label: str, coord_str: str) -> str:
        safe_label = (label or "target").strip()
        if not safe_label:
            safe_label = "target"
        return f'<points coords="{coord_str}">{safe_label}</points>'

    @staticmethod
    def _build_output(point_str: str, count: int, mode: str) -> str:
        m = (mode or "pointing").strip().lower()
        if count == 0:
            return "There are none."
        if m in {"point", "pointing", "cosyn_point"}:
            return point_str
        if m in {"point_count", "point_then_count"}:
            return f"Counting the {point_str} shows a total of {count}."
        if m in {"count_then_point"}:
            return f"There are {count} {point_str}."
        if m in {"count"}:
            return str(count)
        return point_str

    def format_image_points(
        self,
        points: list[tuple[float, float]],
        scale: tuple[float, float] | list[float] | float,
        label: str,
        mode: str = "pointing",
    ) -> str:
        if not points:
            return "There are none."
        coord = self._format_single_image_coordinates(points, scale)
        coord = f"1 {coord}"
        point_str = self._build_point_str(label, coord)
        return self._build_output(point_str, len(points), mode)
