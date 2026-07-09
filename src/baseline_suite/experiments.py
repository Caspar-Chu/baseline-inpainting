from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from baseline_suite.config import ExperimentConfig
from baseline_suite.io import apply_mask, load_image, resize_image, save_image
from baseline_suite.masks import block_mask_from_image, mask_to_visual, random_pixel_mask


@dataclass
class Sample:
    experiment: str
    case_name: str
    image: np.ndarray
    keep_mask: np.ndarray
    masked_image: np.ndarray
    trial: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def mask_path_key(self) -> str:
        parts = [self.experiment, self.case_name]
        if self.trial is not None:
            parts.append(f"trial{self.trial:02d}")
        return "/".join(parts)


def _prepare_image(
    cfg: ExperimentConfig,
    rel_path: str,
    target_size: list[int] | None,
) -> np.ndarray:
    image = load_image(cfg.data_path(rel_path))
    if target_size:
        image = resize_image(image, (target_size[0], target_size[1]))
    return image


def iter_random_pixel_masks(cfg: ExperimentConfig) -> list[Sample]:
    exp = cfg.experiments["random_pixel_masks"]
    image = _prepare_image(cfg, exp["image"], exp.get("target_size"))
    h, w = image.shape[:2]
    samples: list[Sample] = []

    for ratio in exp["missing_ratios"]:
        for trial in range(exp.get("num_trials", 1)):
            seed = trial if exp.get("num_trials", 1) > 1 else exp.get("seed", 42)
            keep = random_pixel_mask(h, w, ratio, seed=seed + int(ratio * 1000))
            masked = apply_mask(image, keep)
            samples.append(
                Sample(
                    experiment="random_pixel_masks",
                    case_name=f"ratio_{int(ratio * 100)}",
                    image=image,
                    keep_mask=keep,
                    masked_image=masked,
                    trial=trial,
                    meta={"missing_ratio": ratio, "seed": seed},
                )
            )
    return samples


def iter_block_masks(cfg: ExperimentConfig) -> list[Sample]:
    exp = cfg.experiments["block_masks"]
    target = tuple(exp.get("target_size", [300, 300]))
    samples: list[Sample] = []

    for case in exp["cases"]:
        image_rel = case.get("image")
        mask_rel = case["mask"]
        if image_rel:
            image = _prepare_image(cfg, image_rel, list(target))
        else:
            # Fallback when only mask PNG is available from paper figures.
            image = _prepare_image(cfg, exp.get("fallback_image", "random-mask素材/leida.png"), list(target))

        keep = block_mask_from_image(cfg.data_path(mask_rel), target)
        if keep.shape[:2] != image.shape[:2]:
            raise ValueError(f"Mask/image size mismatch for case {case['name']}")
        masked = apply_mask(image, keep)
        samples.append(
            Sample(
                experiment="block_masks",
                case_name=case["name"],
                image=image,
                keep_mask=keep,
                masked_image=masked,
                meta={"target_size": target},
            )
        )
    return samples


def iter_missing_patterns(cfg: ExperimentConfig) -> list[Sample]:
    exp = cfg.experiments["missing_patterns"]
    target = exp.get("target_size")
    image = _prepare_image(cfg, exp["image"], target)
    h, w = image.shape[:2]
    ratio = exp["missing_ratio"]
    samples: list[Sample] = []

    for pattern in exp["patterns"]:
        if pattern["mask_type"] == "random":
            seed = pattern.get("seed", 42)
            keep = random_pixel_mask(h, w, ratio, seed=seed)
        else:
            keep = block_mask_from_image(
                cfg.data_path(pattern["mask"]),
                (h, w),
            )
        masked = apply_mask(image, keep)
        samples.append(
            Sample(
                experiment="missing_patterns",
                case_name=pattern["name"],
                image=image,
                keep_mask=keep,
                masked_image=masked,
                meta={"missing_ratio": ratio, "pattern": pattern["name"]},
            )
        )
    return samples


def iter_scenes_resolutions(cfg: ExperimentConfig) -> list[Sample]:
    exp = cfg.experiments["scenes_resolutions"]
    ratio = exp["missing_ratio"]
    seed = exp.get("seed", 42)
    samples: list[Sample] = []

    for case in exp["cases"]:
        image = load_image(cfg.data_path(case["image"]))
        h, w = image.shape[:2]
        keep = random_pixel_mask(h, w, ratio, seed=seed)
        masked = apply_mask(image, keep)
        samples.append(
            Sample(
                experiment="scenes_resolutions",
                case_name=case["name"],
                image=image,
                keep_mask=keep,
                masked_image=masked,
                meta={"missing_ratio": ratio, "seed": seed},
            )
        )
    return samples


EXPERIMENT_BUILDERS = {
    "random_pixel_masks": iter_random_pixel_masks,
    "block_masks": iter_block_masks,
    "missing_patterns": iter_missing_patterns,
    "scenes_resolutions": iter_scenes_resolutions,
}


def build_samples(cfg: ExperimentConfig, experiment: str) -> list[Sample]:
    if experiment not in EXPERIMENT_BUILDERS:
        raise KeyError(
            f"Unknown experiment '{experiment}'. "
            f"Available: {list(EXPERIMENT_BUILDERS.keys())}"
        )
    return EXPERIMENT_BUILDERS[experiment](cfg)


def cache_masks(cfg: ExperimentConfig, samples: list[Sample]) -> None:
    for sample in samples:
        out_dir = cfg.mask_cache / sample.experiment / sample.case_name
        if sample.trial is not None:
            out_dir = out_dir / f"trial_{sample.trial:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_image(out_dir / "original.png", sample.image)
        save_image(out_dir / "masked.png", sample.masked_image)
        save_image(out_dir / "mask_vis.png", mask_to_visual(sample.keep_mask))
        np.save(out_dir / "keep_mask.npy", sample.keep_mask)
