from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

from baseline_suite.baselines.base import BaseInpainter, InpaintResult
from baseline_suite.metrics import Timer
from baseline_suite.paths import EXTERNAL_DIR, resolve


def _resolve_device(device: str | None):
    import torch

    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MAT uses StyleGAN ops without MPS kernels; CPU ref ops are more reliable on macOS.
    return torch.device("cpu")


def _pad_to_multiple(
    image: np.ndarray,
    keep_mask: np.ndarray,
    multiple: int = 512,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    h, w = image.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return image, keep_mask, h, w
    image_padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    mask_padded = np.pad(
        keep_mask,
        ((0, pad_h), (0, pad_w)),
        mode="constant",
        constant_values=0.0,
    )
    return image_padded, mask_padded, h, w


def _copy_params_and_buffers(src_module, dst_module, require_all: bool = False) -> None:
    import torch

    def named_params_and_buffers(module):
        return list(module.named_parameters()) + list(module.named_buffers())

    src_tensors = {
        name: tensor for name, tensor in named_params_and_buffers(src_module)
    }
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)


class _MatBackend:
    """Programmatic MAT inference with CPU/CUDA support (no hardcoded cuda)."""

    def __init__(
        self,
        repo_path: Path,
        checkpoint: Path,
        device,
        *,
        seed: int = 240,
        truncation_psi: float = 1.0,
    ) -> None:
        import torch

        self.device = device
        self.truncation_psi = truncation_psi
        self.seed = seed

        repo = str(repo_path.resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)

        import dnnlib
        import legacy
        from networks.mat import Generator

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)

        orig_torch_load = torch.load

        def _cpu_torch_load(*args, **kwargs):
            kwargs.setdefault("map_location", "cpu")
            return orig_torch_load(*args, **kwargs)

        torch.load = _cpu_torch_load
        try:
            with open(checkpoint, "rb") as f:
                g_saved = legacy.load_network_pkl(f)["G_ema"]
        finally:
            torch.load = orig_torch_load
        g_saved = g_saved.to(device).eval().requires_grad_(False)

        self.generator = Generator(
            z_dim=512,
            c_dim=0,
            w_dim=512,
            img_resolution=512,
            img_channels=3,
        ).to(device).eval().requires_grad_(False)
        _copy_params_and_buffers(g_saved, self.generator, require_all=True)
        self._label = torch.zeros([1, self.generator.c_dim], device=device)
        self._dnnlib = dnnlib

    def __call__(self, image_rgb: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
        import torch

        padded_image, padded_mask, orig_h, orig_w = _pad_to_multiple(image_rgb, keep_mask)
        resolution = padded_image.shape[0]

        image_u8 = (np.clip(padded_image, 0.0, 1.0) * 255.0).astype(np.uint8)
        image_chw = image_u8.transpose(2, 0, 1)[:3]
        image_t = (
            torch.from_numpy(image_chw).float().to(self.device) / 127.5 - 1.0
        ).unsqueeze(0)

        mask_t = (
            torch.from_numpy(padded_mask.astype(np.float32))
            .float()
            .to(self.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        noise_mode = "random" if resolution > 512 else "const"
        z = torch.from_numpy(np.random.randn(1, self.generator.z_dim)).to(self.device)

        with torch.inference_mode():
            output = self.generator(
                image_t,
                mask_t,
                z,
                self._label,
                truncation_psi=self.truncation_psi,
                noise_mode=noise_mode,
            )
            output = (
                (output.permute(0, 2, 3, 1) * 127.5 + 127.5)
                .round()
                .clamp(0, 255)
                .to(torch.uint8)
            )
            restored = output[0].cpu().numpy()

        restored = restored[:orig_h, :orig_w].astype(np.float32) / 255.0
        return restored


class MatInpainter(BaseInpainter):
    """MAT inpainting via external/mat (Places-512 checkpoint).

    Mask convention matches the project: 1=keep, 0=hole (same as MAT official masks).
    Images are padded to the next multiple of 512 before inference, then cropped back.
    """

    name = "mat"
    default_checkpoint = "pretrained/Places_512_FullData.pkl"

    def __init__(
        self,
        repo_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
        *,
        truncation_psi: float = 1.0,
        seed: int = 240,
    ) -> None:
        self.repo_path = resolve(repo_path or EXTERNAL_DIR / "mat")
        self.checkpoint_path = (
            resolve(checkpoint_path)
            if checkpoint_path is not None
            else self.repo_path / self.default_checkpoint
        )
        self.device = device
        self.truncation_psi = truncation_psi
        self.seed = seed
        self._model: _MatBackend | None = None

    def _check_repo(self) -> None:
        if not self.repo_path.exists():
            raise FileNotFoundError(
                f"MAT repo not found at {self.repo_path}. "
                "Clone https://github.com/fenglinglwb/MAT into external/mat."
            )
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"MAT checkpoint not found at {self.checkpoint_path}. "
                "Download Places_512_FullData.pkl into external/mat/pretrained/."
            )

    def _get_model(self) -> _MatBackend:
        if self._model is None:
            self._model = _MatBackend(
                self.repo_path,
                self.checkpoint_path,
                _resolve_device(self.device),
                seed=self.seed,
                truncation_psi=self.truncation_psi,
            )
        return self._model

    def inpaint(
        self,
        masked_image: np.ndarray,
        keep_mask: np.ndarray,
        *,
        reference: np.ndarray | None = None,
    ) -> InpaintResult:
        self._check_repo()
        source = reference if reference is not None else masked_image

        with Timer() as timer:
            restored = self._get_model()(source, keep_mask)

        return InpaintResult(
            image=restored,
            time_sec=timer.elapsed,
            metadata={
                "backend": "external/mat",
                "checkpoint": str(self.checkpoint_path),
            },
        )
