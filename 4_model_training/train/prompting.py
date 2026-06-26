from __future__ import annotations

PROMPT_TEMPLATE_SINGLE = (
    "<image>\n"
    "Task: Locate one target point for the query region in the image.\n"
    "Query Region: {query}\n"
    "Output Constraints:\n"
    "1) Use integer coordinates in [0,1000]. (0,0)=top-left, (1000,1000)=bottom-right.\n"
    "2) Output exactly one point using this format: <point>x,y</point>\n"
    "3) Return only coordinates, no extra words."
)

PROMPT_TEMPLATE_MULTI = (
    "<image>\n"
    "Task: Locate target point(s) for the query region in the image.\n"
    "Query Region: {query}\n"
    "Output Constraints:\n"
    "1) Use integer coordinates in [0,1000]. (0,0)=top-left, (1000,1000)=bottom-right.\n"
    "2) If there is exactly one target point, output: <point>x,y</point>\n"
    "3) If there are multiple target points, output: <points>x1,y1; x2,y2; ...</points>\n"
    "4) Return only coordinates, no extra words."
)

# V3 protocol (Molmo2 native coords tag):
# Keep a single <points coords="..."></points> tag and do not output any extra text.
PROMPT_V3_TEMPLATE_SINGLE = (
    "<image>\n"
    "Task: Locate the target point for the query region.\n"
    "Query Region: {query}\n"
    "Coordinate System:\n"
    "1) Use a fixed 1000x1000 coordinate space: x,y in [0,1000].\n"
    "2) Coordinates must be integers.\n"
    "Output Format (single point):\n"
    "1) Output exactly one tag: <points coords=\"1 1 x y\"></points>\n"
    "2) In coords, x and y are the target coordinates.\n"
    "3) The text between <points> and </points> must be empty.\n"
    "4) Do not output any text before or after the tag.\n"
    "5) Output only this one tag."
)

PROMPT_V3_TEMPLATE_MULTI = (
    "<image>\n"
    "Task: Locate all target points for the query region.\n"
    "Query Region: {query}\n"
    "Coordinate System:\n"
    "1) Use a fixed 1000x1000 coordinate space: x,y in [0,1000].\n"
    "2) Coordinates must be integers.\n"
    "Output Format (multiple points):\n"
    "1) Output exactly one tag: <points coords=\"1 1 x1 y1 2 x2 y2 3 x3 y3 ...\"></points>\n"
    "2) Keep Molmo2 coords style; each point is represented in the coords sequence.\n"
    "3) The text between <points> and </points> must be empty.\n"
    "4) Do not output any text before or after the tag.\n"
    "5) Output only this one tag."
)


def build_prompt(query: str, single_point_only: bool = True) -> str:
    template = PROMPT_TEMPLATE_SINGLE if single_point_only else PROMPT_TEMPLATE_MULTI
    return template.format(query=(query or "").strip())


def build_prompt_v3(query: str, single_point_only: bool = True) -> str:
    template = PROMPT_V3_TEMPLATE_SINGLE if single_point_only else PROMPT_V3_TEMPLATE_MULTI
    return template.format(query=(query or "").strip())


def _clip_1000(v: float) -> int:
    return int(round(max(0.0, min(1000.0, float(v)))))


def _to_1000_point(x: float, y: float) -> tuple[int, int]:
    x = float(x)
    y = float(y)

    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        return (_clip_1000(x * 1000.0), _clip_1000(y * 1000.0))

    if 0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0:
        return (_clip_1000(x), _clip_1000(y))

    return (_clip_1000(x), _clip_1000(y))


def build_target(points_or_x, y: float | None = None, single_point_only: bool = True) -> str:
    points: list[tuple[int, int]] = []

    if y is not None:
        points = [_to_1000_point(float(points_or_x), float(y))]
    else:
        for p in (points_or_x or []):
            try:
                x0 = float(p[0])
                y0 = float(p[1])
            except Exception:
                continue
            points.append(_to_1000_point(x0, y0))

    if not points:
        points = [(500, 500)]

    if single_point_only:
        x0, y0 = points[0]
        return f"<point>{x0},{y0}</point>"

    if len(points) == 1:
        x0, y0 = points[0]
        return f"<point>{x0},{y0}</point>"

    joined = "; ".join(f"{x0},{y0}" for x0, y0 in points)
    return f"<points>{joined}</points>"
