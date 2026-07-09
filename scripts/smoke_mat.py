#!/usr/bin/env python3
"""Single-image MAT smoke test using cached masks."""

from __future__ import annotations

import argparse
from pathlib import Path

from baseline_suite.config import load_config
from baseline_suite.experiments import build_samples
from baseline_suite.io import save_image
from baseline_suite.metrics import compute_lpips, compute_psnr, compute_ssim
from baseline_suite.registry import create_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="MAT single-sample smoke test")
    parser.add_argument(
        "--experiment",
        default="missing_patterns",
        choices=[
            "random_pixel_masks",
            "block_masks",
            "missing_patterns",
            "scenes_resolutions",
        ],
    )
    parser.add_argument("--case", default="random", help="Sample case name")
    parser.add_argument(
        "--output",
        default="results/smoke_mat/restored.png",
        help="Where to save restored image",
    )
    args = parser.parse_args()

    cfg = load_config()
    samples = build_samples(cfg, args.experiment)
    sample = next(s for s in samples if s.case_name == args.case)

    inpainter = create_baseline("mat")
    result = inpainter.inpaint(
        sample.masked_image,
        sample.keep_mask,
        reference=sample.image,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image(out, result.image)

    psnr = compute_psnr(sample.image, result.image)
    ssim = compute_ssim(sample.image, result.image)
    lpips = compute_lpips(sample.image, result.image)

    print(f"case: {sample.case_name}")
    print(f"image shape: {sample.image.shape}")
    print(f"time_sec: {result.time_sec:.3f}")
    print(f"PSNR: {psnr:.3f}")
    print(f"SSIM: {ssim:.3f}")
    print(f"LPIPS: {lpips:.3f}")
    print(f"saved: {out.resolve()}")


if __name__ == "__main__":
    main()
