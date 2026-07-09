from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np


def random_pixel_mask(
    height: int,
    width: int,
    missing_ratio: float,
    seed: int | None = None,
) -> np.ndarray:
    """Generate random pixel mask. 1=keep, 0=missing"""
    if not 0.0 <= missing_ratio <= 1.0:
        raise ValueError(f"missing_ratio must be in [0, 1], got {missing_ratio}")

    rng = np.random.default_rng(seed)
    mask = np.ones((height, width), dtype=np.float32)
    n_missing = int(round(height * width * missing_ratio))
    if n_missing > 0:
        flat = mask.ravel()
        indices = rng.choice(height * width, n_missing, replace=False)
        flat[indices] = 0.0
        mask = flat.reshape(height, width)
    return mask


def block_mask_from_image(
    mask_image_path: str | Path,
    target_size: tuple[int, int] | None = None,
    *,
    black_is_missing: bool = True,
) -> np.ndarray:
    """
    Convert block mask PNG to keep mask (1=keep, 0=missing).
    Per project notes: black region=1, white=0 in source mask image.
    """
    gray = cv2.imread(str(mask_image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot read mask image: {mask_image_path}")

    if target_size is not None:
        h, w = target_size
        gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_NEAREST)

    # Per project notes: black -> 1 (keep), white -> 0 (missing).
    keep = (gray < 128).astype(np.float32)
    if not black_is_missing:
        keep = 1.0 - keep
    return keep


def mask_to_visual(mask: np.ndarray) -> np.ndarray:
    """Visualize keep mask as RGB (white=keep, black=missing)."""
    vis = np.clip(mask, 0.0, 1.0)
    if vis.ndim == 2:
        vis = vis[..., None]
    return np.repeat(vis, 3, axis=2)
