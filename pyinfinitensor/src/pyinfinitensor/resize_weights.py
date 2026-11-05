"""Shape-only weights for separable linear ONNX Resize."""

import math

import numpy as np


def linear_resize_weights(source, target, coordinates, exclude, inverse_scale=None, antialias=False):
    if source <= 0 or target <= 0:
        raise ValueError("Linear Resize requires positive dimensions")
    scale = source / target if inverse_scale is None else inverse_scale
    if coordinates == "align_corners":
        scale = (source - 1) / (target - 1) if target > 1 else 0.0
        centers = np.arange(target, dtype=np.float64) * scale + .5
    elif coordinates == "half_pixel":
        centers = (np.arange(target, dtype=np.float64) + .5) * scale
    else:
        raise ValueError(f"Unsupported linear Resize coordinates: {coordinates}")
    support = max(1.0, scale) if antialias else 1.0
    padding = math.ceil(support) + 1
    if (source + 2 * padding) * target > 4_194_304:
        raise NotImplementedError("Antialiased Resize weights exceed the bounded lowering")
    indices = np.arange(-padding, source + padding)
    weights = np.maximum(0.0, 1.0 - np.abs(indices[None, :] + .5 - centers[:, None]) / support)
    result = weights[:, padding:padding + source].copy()
    if not exclude:
        result[:, 0] += weights[:, :padding].sum(axis=1)
        result[:, -1] += weights[:, padding + source:].sum(axis=1)
    denominator = result.sum(axis=1, keepdims=True)
    if np.any(denominator == 0):
        raise ValueError("Linear Resize has an empty sampling window")
    return (result / denominator).T
