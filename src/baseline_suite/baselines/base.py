from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class InpaintResult:
    image: np.ndarray
    time_sec: float
    metadata: dict | None = None


class BaseInpainter(ABC):
    name: str

    @abstractmethod
    def inpaint(
        self,
        masked_image: np.ndarray,
        keep_mask: np.ndarray,
        *,
        reference: np.ndarray | None = None,
    ) -> InpaintResult:
        """
        Restore a masked image.

        Args:
            masked_image: RGB float32 in [0, 1], already multiplied by keep_mask.
            keep_mask: 2D float32, 1=keep/observed, 0=missing.
            reference: optional ground truth (not used at inference for fair comparison).
        """

    def hole_mask(self, keep_mask: np.ndarray) -> np.ndarray:
        """Return hole mask where 1=missing (common in diffusion repos)."""
        return 1.0 - keep_mask
