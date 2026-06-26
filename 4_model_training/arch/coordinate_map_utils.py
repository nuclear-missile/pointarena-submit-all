"""
Coordinate Map Utilities for Iterative Correction Experiments

Builds multi-layer Gaussian coordinate maps as images.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def build_coordinate_map_tensor(
    points: list[tuple[float, float]],
    grid_size: int = 100,
    sigmas: list[float] | None = None,
    coordinate_scale: float = 1000.0,
) -> torch.Tensor:
    """
    Build multi-layer Gaussian coordinate map as tensor.

    Args:
        points: [(x1, y1), ...] in coordinate_scale space
        grid_size: output grid size (grid_size × grid_size)
        sigmas: sigma for each Gaussian layer (decreasing = finer precision)
        coordinate_scale: max coordinate value

    Returns:
        Tensor (num_layers, grid_size, grid_size)
    """
    if sigmas is None:
        sigmas = [8.0, 4.0, 2.0, 1.0]

    H = W = grid_size
    y_g, x_g = torch.meshgrid(
        torch.linspace(0, coordinate_scale, H),
        torch.linspace(0, coordinate_scale, W),
        indexing='ij',
    )

    maps = []
    for sigma in sigmas:
        layer = torch.zeros(H, W)
        for x, y in points:
            g = torch.exp(-((x_g - x)**2 + (y_g - y)**2) / (2 * sigma**2))
            layer = torch.maximum(layer, g)
        maps.append(layer)

    return torch.stack(maps, dim=0)  # (N_layers, H, W)


def build_coordinate_map_image(
    points: list[tuple[float, float]],
    output_size: tuple[int, int] = (378, 378),
    sigmas: list[float] | None = None,
    coordinate_scale: float = 1000.0,
) -> Image.Image:
    """
    Render coordinate map as an RGB image for input to Molmo2.

    Each sigma layer is rendered in a different color channel:
    - Channel R: sigma=8.0 (coarse)
    - Channel G: sigma=4.0 (medium-coarse)
    - Channel B: sigma=2.0 (medium-fine)
    If 4 layers, the 4th (sigma=1.0) is blended as alpha/intensity.

    Args:
        points: point coordinates in coordinate_scale space
        output_size: (width, height) of output image
        sigmas: Gaussian sigma per layer
        coordinate_scale: coordinate range

    Returns:
        PIL Image (RGB, output_size)
    """
    if sigmas is None:
        sigmas = [8.0, 4.0, 2.0, 1.0]

    sigmas = list(sigmas)[:4]  # Max 4 layers
    while len(sigmas) < 3:
        sigmas = sigmas + [sigmas[-1] * 0.5]

    H, W = output_size[1], output_size[0]  # PIL is (W, H)

    y_g, x_g = torch.meshgrid(
        torch.linspace(0, coordinate_scale, H),
        torch.linspace(0, coordinate_scale, W),
        indexing='ij',
    )

    # Build each layer
    layers = []
    for sigma in sigmas:
        layer = torch.zeros(H, W)
        for x, y in points:
            g = torch.exp(-((x_g - x)**2 + (y_g - y)**2) / (2 * sigma**2))
            layer = torch.maximum(layer, g)
        layers.append(layer)

    # Map to RGB channels
    rgb = torch.zeros(H, W, 3)
    if len(layers) >= 1:
        rgb[:, :, 0] = layers[0]  # R: coarsest
    if len(layers) >= 2:
        rgb[:, :, 1] = layers[1]  # G: medium
    if len(layers) >= 3:
        rgb[:, :, 2] = layers[2]  # B: fine
    if len(layers) >= 4:
        # Blend 4th layer as intensity boost
        boost = layers[3].unsqueeze(-1) * 0.5
        rgb = torch.clamp(rgb + boost * rgb, 0, 1)

    # Convert to PIL
    rgb_np = (rgb.numpy() * 255).astype(np.uint8)
    return Image.fromarray(rgb_np, mode="RGB")


def build_point_mlp_embedding(
    coords: list[tuple[float, float]],
    mlp: torch.nn.Module | None = None,
    coordinate_scale: float = 1000.0,
) -> torch.Tensor:
    """
    Encode point coordinates via MLP.

    Args:
        coords: [(x1, y1), ...]
        mlp: optional pre-built MLP module
        coordinate_scale: coordinate range

    Returns:
        Tensor (num_points, embedding_dim)
    """
    if mlp is None:
        # Default: simple 3-layer MLP
        mlp = torch.nn.Sequential(
            torch.nn.Linear(2, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, 256),
        )

    if not coords:
        return torch.zeros(0, 256)

    # Normalize to [0, 1]
    normalized = torch.tensor(
        [[x / coordinate_scale, y / coordinate_scale] for x, y in coords],
        dtype=torch.float32,
    )
    with torch.no_grad():
        return mlp(normalized)
