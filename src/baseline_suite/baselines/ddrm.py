from __future__ import annotations

from pathlib import Path

from baseline_suite.baselines.base import BaseInpainter, InpaintResult
from baseline_suite.metrics import Timer
from baseline_suite.paths import EXTERNAL_DIR, resolve


class DdrmInpainter(BaseInpainter):
    name = "ddrm"

    def __init__(self, repo_path: str | Path | None = None) -> None:
        self.repo_path = resolve(repo_path or EXTERNAL_DIR / "ddrm")

    def _check_repo(self) -> None:
        if not self.repo_path.exists():
            raise FileNotFoundError(
                f"DDRM repo not found at {self.repo_path}. "
                "Clone https://github.com/BGUCompSci/DDRM into external/ddrm."
            )

    def inpaint(
        self,
        masked_image,
        keep_mask,
        *,
        reference=None,
    ) -> InpaintResult:
        self._check_repo()
        with Timer() as timer:
            raise NotImplementedError(
                "DDRM wrapper stub. Configure DDRM inpainting task and checkpoint."
            )
        return InpaintResult(image=masked_image, time_sec=timer.elapsed)
