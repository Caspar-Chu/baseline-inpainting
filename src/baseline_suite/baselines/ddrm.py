from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import yaml

from baseline_suite.baselines._repo_isolation import (
    activate_external_repo,
    assert_model_from_repo,
)
from baseline_suite.baselines.base import BaseInpainter, InpaintResult
from baseline_suite.metrics import Timer
from baseline_suite.paths import CONFIGS_DIR, EXTERNAL_DIR, resolve


def _resolve_device(device: str | None):
    import torch

    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _dict2namespace(config: dict) -> argparse.Namespace:
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = _dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def _resize_to_square(
    image: np.ndarray,
    keep_mask: np.ndarray,
    size: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    import cv2

    orig_h, orig_w = image.shape[:2]
    if orig_h == size and orig_w == size:
        return image, keep_mask, (orig_h, orig_w)

    interp = cv2.INTER_AREA if max(orig_h, orig_w) > size else cv2.INTER_LINEAR
    image_rs = cv2.resize(image, (size, size), interpolation=interp)
    mask_rs = cv2.resize(
        keep_mask.astype(np.float32),
        (size, size),
        interpolation=cv2.INTER_NEAREST,
    )
    return image_rs, mask_rs, (orig_h, orig_w)


def _resize_from_square(image: np.ndarray, orig_hw: tuple[int, int]) -> np.ndarray:
    import cv2

    orig_h, orig_w = orig_hw
    if image.shape[0] == orig_h and image.shape[1] == orig_w:
        return image
    return cv2.resize(image, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


def _get_beta_schedule(
    beta_schedule: str,
    *,
    beta_start: float,
    beta_end: float,
    num_diffusion_timesteps: int,
) -> np.ndarray:
    if beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "quad":
        betas = (
            np.linspace(beta_start**0.5, beta_end**0.5, num_diffusion_timesteps, dtype=np.float64)
            ** 2
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported beta schedule: {beta_schedule}")
    return betas


def _missing_indices_from_keep_mask(keep_mask: np.ndarray, image_size: int, device) -> "torch.Tensor":
    import torch

    mask = torch.from_numpy(keep_mask.reshape(image_size, image_size).astype(np.float32)).to(device)
    hole_pixels = torch.nonzero(mask.reshape(-1) < 0.5, as_tuple=False).squeeze(1)
    if hole_pixels.numel() == 0:
        raise ValueError("keep_mask has no missing pixels (all ones).")
    missing_r = hole_pixels.long() * 3
    missing_g = missing_r + 1
    missing_b = missing_g + 1
    return torch.cat([missing_r, missing_g, missing_b], dim=0)


_DDRM_PURGE = ("guided_diffusion", "functions", "datasets", "models", "runners", "conf_mgt", "utils")
class _DdrmBackend:
    """Programmatic DDRM inpainting (ImageNet-256 unconditional DDPM)."""

    def __init__(
        self,
        repo_path: Path,
        conf_path: Path,
        checkpoint_path: Path,
        device,
        *,
        seed: int = 1234,
        timesteps: int = 20,
        eta: float = 0.85,
        eta_b: float = 1.0,
        sigma_0: float = 0.0,
        show_progress: bool = False,
    ) -> None:
        import torch

        self.device = device
        self.repo_path = repo_path
        self.seed = seed
        self.timesteps = timesteps
        self.eta = eta
        self.eta_b = eta_b
        self.sigma_0 = 2.0 * sigma_0
        self.image_size = 256
        self.show_progress = show_progress

        if not show_progress:
            os.environ.setdefault("TQDM_DISABLE", "1")

        activate_external_repo(repo_path, purge_roots=_DDRM_PURGE)

        from datasets import data_transform, inverse_data_transform
        from functions.denoising import efficient_generalized_steps
        from guided_diffusion.script_util import create_model

        with open(conf_path, encoding="utf-8") as f:
            config = _dict2namespace(yaml.safe_load(f))

        if device.type != "cuda":
            config.model.use_fp16 = False

        self.config = config
        self._data_transform = data_transform
        self._inverse_data_transform = inverse_data_transform
        self._efficient_generalized_steps = efficient_generalized_steps

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        betas = _get_beta_schedule(
            config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        self.betas = torch.from_numpy(betas).float().to(device)
        self.num_timesteps = self.betas.shape[0]

        model_kwargs = {
            k: v
            for k, v in vars(config.model).items()
            if k not in {"type", "var_type", "in_channels", "out_channels", "resamp_with_conv"}
        }
        model = create_model(**model_kwargs)
        if config.model.use_fp16:
            model.convert_to_fp16()
        state = torch.load(str(checkpoint_path), map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        assert_model_from_repo(model, repo_path, label="DDRM")
        self.model = model

    def __call__(self, image_rgb: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
        import torch

        activate_external_repo(self.repo_path, purge_roots=_DDRM_PURGE)
        from functions.svd_replacement import Inpainting

        image_rs, mask_rs, orig_hw = _resize_to_square(
            image_rgb, keep_mask, self.image_size
        )

        image_chw = np.clip(image_rs, 0.0, 1.0).astype(np.float32).transpose(2, 0, 1)
        x_orig = torch.from_numpy(image_chw).float().unsqueeze(0).to(self.device)
        x_orig = self._data_transform(self.config, x_orig)

        missing = _missing_indices_from_keep_mask(mask_rs, self.image_size, self.device)
        h_funcs = Inpainting(
            self.config.data.channels,
            self.image_size,
            missing,
            self.device,
        )

        y_0 = h_funcs.H(x_orig)
        if self.sigma_0 > 0:
            y_0 = y_0 + self.sigma_0 * torch.randn_like(y_0)

        x = torch.randn_like(x_orig)
        skip = self.num_timesteps // self.timesteps
        seq = range(0, self.num_timesteps, skip)

        with torch.inference_mode():
            xs, _ = self._efficient_generalized_steps(
                x,
                seq,
                self.model,
                self.betas,
                h_funcs,
                y_0,
                self.sigma_0,
                etaB=self.eta_b,
                etaA=self.eta,
                etaC=self.eta,
                cls_fn=None,
                classes=None,
            )
            restored = self._inverse_data_transform(self.config, xs[-1])[0]
            out = restored.permute(1, 2, 0).detach().cpu().numpy()

        out = _resize_from_square(out, orig_hw)
        return np.clip(out, 0.0, 1.0).astype(np.float32)


class DdrmInpainter(BaseInpainter):
    """DDRM inpainting via external/ddrm (ImageNet-256 unconditional).

    Mask convention: 1=keep/observed, 0=hole — converted to DDRM missing indices.
    Non-256 inputs are resized to 256 for inference, then resized back.
    Requires CUDA for practical runtime (denoising loop is GPU-oriented).
    """

    name = "ddrm"
    default_checkpoint = "exp/logs/imagenet/256x256_diffusion_uncond.pt"
    default_conf = "ddrm_imagenet256.yml"

    def __init__(
        self,
        repo_path: str | Path | None = None,
        conf_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
        *,
        seed: int = 1234,
        timesteps: int = 20,
        eta: float = 0.85,
        eta_b: float = 1.0,
        sigma_0: float = 0.0,
        show_progress: bool = False,
    ) -> None:
        self.repo_path = resolve(repo_path or EXTERNAL_DIR / "ddrm")
        self.conf_path = (
            resolve(conf_path)
            if conf_path is not None
            else CONFIGS_DIR / self.default_conf
        )
        self.checkpoint_path = (
            resolve(checkpoint_path, root=self.repo_path)
            if checkpoint_path is not None
            else self.repo_path / self.default_checkpoint
        )
        self.device = device
        self.seed = seed
        self.timesteps = timesteps
        self.eta = eta
        self.eta_b = eta_b
        self.sigma_0 = sigma_0
        self.show_progress = show_progress
        self._model: _DdrmBackend | None = None

    def _check_repo(self) -> None:
        if not self.repo_path.exists():
            raise FileNotFoundError(
                f"DDRM repo not found at {self.repo_path}. "
                "Clone https://github.com/bahjat-kawar/ddrm into external/ddrm."
            )
        if not self.conf_path.exists():
            raise FileNotFoundError(f"DDRM config not found at {self.conf_path}.")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"DDRM checkpoint not found at {self.checkpoint_path}. "
                "Download OpenAI 256x256_diffusion_uncond.pt into "
                "external/ddrm/exp/logs/imagenet/ (see external/ddrm/README.md)."
            )

    def _get_model(self) -> _DdrmBackend:
        if self._model is None:
            device = _resolve_device(self.device)
            if device.type != "cuda":
                raise RuntimeError(
                    "DDRM inference requires CUDA. Set device='cuda' or run on a GPU server."
                )
            self._model = _DdrmBackend(
                self.repo_path,
                self.conf_path,
                self.checkpoint_path,
                device,
                seed=self.seed,
                timesteps=self.timesteps,
                eta=self.eta,
                eta_b=self.eta_b,
                sigma_0=self.sigma_0,
                show_progress=self.show_progress,
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
                "backend": "external/ddrm",
                "checkpoint": str(self.checkpoint_path),
                "seed": self.seed,
                "timesteps": self.timesteps,
                "eta": self.eta,
                "eta_b": self.eta_b,
                "sigma_0": self.sigma_0,
                "image_size": 256,
            },
        )
