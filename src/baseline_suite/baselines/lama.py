from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from baseline_suite.baselines.base import BaseInpainter, InpaintResult
from baseline_suite.metrics import Timer
from baseline_suite.paths import EXTERNAL_DIR, resolve


def _resolve_device(device: str | None):
    import torch

    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _LamaBackend:
    """Loads big-lama TorchScript with CPU map_location (required on macOS)."""

    def __init__(self, device) -> None:
        import torch
        from simple_lama_inpainting.models.model import LAMA_MODEL_URL
        from simple_lama_inpainting.utils import download_model, prepare_img_and_mask

        self.device = device
        self._prepare = prepare_img_and_mask
        model_path = download_model(LAMA_MODEL_URL)
        self.model = torch.jit.load(model_path, map_location="cpu")
        self.model.eval()
        self.model.to(device)

    def __call__(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        import torch

        image_t, mask_t = self._prepare(image, mask, self.device)
        with torch.inference_mode():
            inpainted = self.model(image_t, mask_t)
            cur_res = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
            cur_res = np.clip(cur_res * 255, 0, 255).astype(np.uint8)
            return Image.fromarray(cur_res)


class LamaInpainter(BaseInpainter):
    """LaMa inpainting via simple-lama-inpainting (big-lama weights).

    The official repo is expected at external/lama for reproducibility.
    Inference uses the TorchScript big-lama model, matching LaMa predict.py
    mask semantics: non-zero mask pixels = holes to inpaint.
    """

    name = "lama"

    def __init__(
        self,
        repo_path: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        self.repo_path = resolve(repo_path or EXTERNAL_DIR / "lama")
        self.device = device
        self._model: _LamaBackend | None = None

    def _check_repo(self) -> None:
        if not self.repo_path.exists():
            raise FileNotFoundError(
                f"LaMa repo not found at {self.repo_path}. "
                "Clone https://github.com/advimman/lama into external/lama."
            )

    def _get_model(self) -> _LamaBackend:
        if self._model is None:
            self._model = _LamaBackend(_resolve_device(self.device))
        return self._model

    @staticmethod
    def _to_pil_rgb(image: np.ndarray) -> Image.Image:
        arr = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    @staticmethod
    def _hole_mask_pil(keep_mask: np.ndarray) -> Image.Image:
        hole = 1.0 - keep_mask
        hole_uint8 = (np.clip(hole, 0.0, 1.0) * 255.0).astype(np.uint8)
        return Image.fromarray(hole_uint8, mode="L")

    def inpaint(
        self,
        masked_image: np.ndarray,
        keep_mask: np.ndarray,
        *,
        reference: np.ndarray | None = None,
    ) -> InpaintResult:
        self._check_repo()
        source = reference if reference is not None else masked_image

        image_pil = self._to_pil_rgb(source)
        mask_pil = self._hole_mask_pil(keep_mask)

        with Timer() as timer:
            result_pil = self._get_model()(image_pil, mask_pil)

        restored = np.asarray(result_pil).astype(np.float32) / 255.0
        target_h, target_w = source.shape[:2]
        restored = restored[:target_h, :target_w]
        return InpaintResult(
            image=restored,
            time_sec=timer.elapsed,
            metadata={"backend": "simple-lama-inpainting"},
        )
