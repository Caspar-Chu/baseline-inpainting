from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

_lpips_model = None


def _get_lpips_model():
    global _lpips_model
    if _lpips_model is None:
        import lpips

        _lpips_model = lpips.LPIPS(net="alex")
        _lpips_model.eval()
    return _lpips_model


def to_uint8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


@dataclass
class MetricResult:
    psnr: float | None = None
    ssim: float | None = None
    lpips: float | None = None
    time_sec: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.psnr is not None:
            out["psnr"] = self.psnr
        if self.ssim is not None:
            out["ssim"] = self.ssim
        if self.lpips is not None:
            out["lpips"] = self.lpips
        if self.time_sec is not None:
            out["time_sec"] = self.time_sec
        out.update(self.extra)
        return out


def compute_psnr(reference: np.ndarray, restored: np.ndarray) -> float:
    ref = to_uint8(reference)
    rec = to_uint8(restored)
    return float(peak_signal_noise_ratio(ref, rec, data_range=255))


def compute_ssim(reference: np.ndarray, restored: np.ndarray) -> float:
    ref = to_uint8(reference)
    rec = to_uint8(restored)
    return float(
        structural_similarity(ref, rec, channel_axis=2, data_range=255)
    )


def compute_lpips(reference: np.ndarray, restored: np.ndarray) -> float:
    model = _get_lpips_model()
    ref_t = torch.from_numpy(reference).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
    rec_t = torch.from_numpy(restored).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
    with torch.no_grad():
        value = model(ref_t, rec_t)
    return float(value.item())


def evaluate(
    reference: np.ndarray,
    restored: np.ndarray,
    *,
    metrics: list[str] | None = None,
) -> MetricResult:
    names = {m.lower() for m in (metrics or ["psnr", "ssim", "lpips"])}
    result = MetricResult()
    if "psnr" in names:
        result.psnr = compute_psnr(reference, restored)
    if "ssim" in names:
        result.ssim = compute_ssim(reference, restored)
    if "lpips" in names:
        result.lpips = compute_lpips(reference, restored)
    return result


class Timer:
    def __init__(self) -> None:
        self._start: float | None = None
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        if self._start is not None:
            self.elapsed = time.perf_counter() - self._start
