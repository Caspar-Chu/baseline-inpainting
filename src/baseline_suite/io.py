from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    """Load image as float32 RGB array in [0, 1]."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def save_image(path: str | Path, image: np.ndarray) -> None:
    """Save float32 RGB image in [0, 1] to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(image, 0.0, 1.0)
    bgr = cv2.cvtColor((clipped * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    h, w = size
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)


def apply_mask(image: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
    """Apply keep mask: 1=keep, 0=missing. masked = image * mask."""
    if keep_mask.ndim == 2:
        keep_mask = keep_mask[..., None]
    return image * keep_mask.astype(np.float32)
