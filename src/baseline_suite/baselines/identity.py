from __future__ import annotations

import numpy as np

from baseline_suite.baselines.base import BaseInpainter, InpaintResult
from baseline_suite.metrics import Timer


class IdentityInpainter(BaseInpainter):
    """Passthrough baseline for pipeline smoke tests."""

    name = "identity"

    def inpaint(
        self,
        masked_image: np.ndarray,
        keep_mask: np.ndarray,
        *,
        reference: np.ndarray | None = None,
    ) -> InpaintResult:
        with Timer() as timer:
            restored = masked_image.copy()
        return InpaintResult(image=restored, time_sec=timer.elapsed)
