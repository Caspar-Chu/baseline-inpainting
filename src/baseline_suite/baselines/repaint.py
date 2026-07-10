from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from baseline_suite.baselines._repo_isolation import activate_external_repo
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


_REPAINT_PURGE = ("guided_diffusion", "conf_mgt", "utils", "functions", "datasets", "models")


class _RePaintBackend:
    """Programmatic RePaint inference (Places2 / 256 DDPM)."""

    def __init__(
        self,
        repo_path: Path,
        conf_path: Path,
        checkpoint_path: Path,
        device,
        *,
        seed: int = 42,
        show_progress: bool = False,
    ) -> None:
        import torch

        self.device = device
        self.seed = seed
        self.image_size = 256

        activate_external_repo(
            repo_path,
            purge_roots=_REPAINT_PURGE,
        )

        from conf_mgt.conf_base import Default_Conf
        from guided_diffusion import dist_util
        from guided_diffusion.script_util import (
            NUM_CLASSES,
            create_model_and_diffusion,
            model_and_diffusion_defaults,
            select_args,
        )
        from utils import yamlread

        conf = Default_Conf()
        conf.update(yamlread(str(conf_path)))
        conf["show_progress"] = show_progress
        conf["model_path"] = str(checkpoint_path)
        self.conf = conf
        self._torch = torch
        self._num_classes = NUM_CLASSES

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        model, diffusion = create_model_and_diffusion(
            **select_args(conf, model_and_diffusion_defaults().keys()),
            conf=conf,
        )
        model.load_state_dict(
            dist_util.load_state_dict(str(checkpoint_path), map_location="cpu")
        )
        model.to(device)
        if conf.get("use_fp16"):
            model.convert_to_fp16()
        model.eval()

        self.model = model
        self.diffusion = diffusion

    def __call__(self, image_rgb: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
        import torch

        image_rs, mask_rs, orig_hw = _resize_to_square(
            image_rgb, keep_mask, self.image_size
        )

        image_u8 = (np.clip(image_rs, 0.0, 1.0) * 255.0).astype(np.uint8)
        arr_gt = image_u8.astype(np.float32) / 127.5 - 1.0
        gt = (
            torch.from_numpy(arr_gt.transpose(2, 0, 1))
            .float()
            .unsqueeze(0)
            .to(self.device)
        )

        mask_3ch = np.repeat(mask_rs.astype(np.float32)[None, ...], 3, axis=0)
        gt_keep_mask = torch.from_numpy(mask_3ch).float().unsqueeze(0).to(self.device)

        batch_size = 1
        classes = torch.randint(
            low=0,
            high=self._num_classes,
            size=(batch_size,),
            device=self.device,
        )

        model_kwargs = {
            "gt": gt,
            "gt_keep_mask": gt_keep_mask,
            "y": classes,
        }

        def model_fn(x, t, y=None, gt=None, **kwargs):
            assert y is not None
            return self.model(
                x,
                t,
                y if self.conf.get("class_cond") else None,
                gt=gt,
            )

        with torch.inference_mode():
            result = self.diffusion.p_sample_loop(
                model_fn,
                (batch_size, 3, self.image_size, self.image_size),
                clip_denoised=self.conf.get("clip_denoised", True),
                model_kwargs=model_kwargs,
                cond_fn=None,
                device=self.device,
                progress=self.conf.get("show_progress", False),
                return_all=True,
                conf=self.conf,
            )

        sample = result["sample"]
        out_u8 = (
            ((sample + 1) * 127.5)
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .contiguous()
            .detach()
            .cpu()
            .numpy()[0]
        )

        restored = out_u8.astype(np.float32) / 255.0
        restored = _resize_from_square(restored, orig_hw)
        return restored


class RePaintInpainter(BaseInpainter):
    """RePaint inpainting via external/repaint (Places2-256 checkpoint).

    Mask convention: 1=keep/known, 0=hole (RePaint uses 0–1 keep mask, same as keep_mask).
    Non-256 inputs are resized to 256 for inference, then resized back to the original size.
    """

    name = "repaint"
    default_checkpoint = "data/pretrained/places256_300000.pt"
    default_conf = "repaint_places256.yml"

    def __init__(
        self,
        repo_path: str | Path | None = None,
        conf_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
        *,
        seed: int = 42,
        show_progress: bool = False,
    ) -> None:
        self.repo_path = resolve(repo_path or EXTERNAL_DIR / "repaint")
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
        self.show_progress = show_progress
        self._model: _RePaintBackend | None = None

    def _check_repo(self) -> None:
        if not self.repo_path.exists():
            raise FileNotFoundError(
                f"RePaint repo not found at {self.repo_path}. "
                "Clone https://github.com/andreas128/RePaint into external/repaint."
            )
        if not self.conf_path.exists():
            raise FileNotFoundError(f"RePaint config not found at {self.conf_path}.")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"RePaint checkpoint not found at {self.checkpoint_path}. "
                "Run: cd external/repaint && bash download.sh "
                "(or scp places256_300000.pt to data/pretrained/)."
            )

    def _get_model(self) -> _RePaintBackend:
        if self._model is None:
            self._model = _RePaintBackend(
                self.repo_path,
                self.conf_path,
                self.checkpoint_path,
                _resolve_device(self.device),
                seed=self.seed,
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

        schedule = self._model.conf.get("schedule_jump_params", {}) if self._model else {}
        return InpaintResult(
            image=restored,
            time_sec=timer.elapsed,
            metadata={
                "backend": "external/repaint",
                "checkpoint": str(self.checkpoint_path),
                "seed": self.seed,
                "t_T": schedule.get("t_T"),
                "jump_n_sample": schedule.get("jump_n_sample"),
                "image_size": 256,
            },
        )
