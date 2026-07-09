from __future__ import annotations

from pathlib import Path

from baseline_suite.baselines.base import BaseInpainter, InpaintResult
from baseline_suite.metrics import Timer
from baseline_suite.paths import EXTERNAL_DIR, resolve


class RePaintInpainter(BaseInpainter):
    name = "repaint"

    def __init__(self, repo_path: str | Path | None = None) -> None:
        self.repo_path = resolve(repo_path or EXTERNAL_DIR / "repaint")

    def _check_repo(self) -> None:
        if not self.repo_path.exists():
            raise FileNotFoundError(
                f"RePaint repo not found at {self.repo_path}. "
                "Clone https://github.com/andreas128/RePaint into external/repaint."
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
                "RePaint wrapper stub. Hole mask convention: 1=missing."
            )
        return InpaintResult(image=masked_image, time_sec=timer.elapsed)
