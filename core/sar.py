"""SatQuery AI - SAR preprocessing proxy (classical, no trained fusion).

Cross-modal (optical + SAR) scaffolding for resource.md §3B "Optical-SAR
fusion (mandatory)". SAR input is treated as single-band backscatter
amplitude: a dB-scale (log) contrast stretch, then a median speckle filter,
then water/built-up rule heads derived from backscatter level + local
texture (double-bounce proxy). This is a classical, deterministic proxy —
not a trained SAR classifier or a learned fusion model. v1 scope per
resource.md: "No trained fusion needed for v1."
"""
from __future__ import annotations

import cv2
import numpy as np


def to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def backscatter_stretch(gray: np.ndarray) -> np.ndarray:
    """Log (dB-like) contrast stretch of raw amplitude to visualise backscatter."""
    g = gray.astype(np.float32) + 1.0
    db = 20.0 * np.log10(g)
    lo, hi = float(np.percentile(db, 2)), float(np.percentile(db, 98))
    stretched = np.clip((db - lo) / (hi - lo + 1e-6), 0, 1)
    return (stretched * 255).astype(np.uint8)


def despeckle(gray: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Median-filter speckle reduction — classical proxy for Lee/Gamma-MAP filters."""
    k = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.medianBlur(gray, k)


def local_texture(gray: np.ndarray, ksize: int = 9) -> np.ndarray:
    """Local standard deviation as a texture proxy (built-up = high texture + backscatter)."""
    g = gray.astype(np.float32)
    mean = cv2.blur(g, (ksize, ksize))
    sq_mean = cv2.blur(g * g, (ksize, ksize))
    var = np.clip(sq_mean - mean * mean, 0, None)
    return np.sqrt(var)


def preprocess(rgb_or_gray: np.ndarray) -> dict:
    """SAR preprocessing chain: stretch -> despeckle -> water/built-up rule head.

    Returns processed grayscale (as RGB for rendering), boolean water/built
    masks and their scene fractions, plus the exact params used — all
    classical/deterministic (resource.md §3B v1 scope).
    """
    gray0 = to_grayscale(np.asarray(rgb_or_gray))
    if gray0.size == 0:
        raise ValueError("empty SAR raster")
    stretched = backscatter_stretch(gray0)
    filtered = despeckle(stretched)
    texture = local_texture(filtered)

    # Water: smooth (low texture) specular return -> low backscatter.
    water_thresh = np.percentile(filtered, 25)
    tex_lo = np.percentile(texture, 40)
    water_mask = (filtered < water_thresh) & (texture < tex_lo)

    # Built-up: strong corner-reflector return + high local texture (double-bounce).
    built_thresh = np.percentile(filtered, 75)
    tex_hi = np.percentile(texture, 70)
    built_mask = (filtered > built_thresh) & (texture > tex_hi)

    rgb_out = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)
    total = float(filtered.size)
    return {
        "processed_gray": filtered,
        "processed_rgb": rgb_out,
        "water_mask": water_mask,
        "built_mask": built_mask,
        "water_fraction": float(water_mask.sum() / total),
        "built_fraction": float(built_mask.sum() / total),
        "shape": filtered.shape,
        "params": {"speckle_filter": "median_5x5",
                   "stretch": "log_db_2_98pct",
                   "water_rule": "backscatter<p25 & texture<p40",
                   "built_rule": "backscatter>p75 & texture>p70"},
    }


def resize_mask(mask: np.ndarray, shape_hw: tuple) -> np.ndarray:
    """Nearest-neighbour resize of a boolean mask to (height, width)."""
    h, w = shape_hw
    m = mask.astype(np.uint8)
    r = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return r.astype(bool)


def resize_rgb(rgb: np.ndarray, shape_hw: tuple) -> np.ndarray:
    h, w = shape_hw
    return cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
