from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from baseline_suite.paths import CONFIGS_DIR, resolve


@dataclass
class ExperimentConfig:
    raw: dict[str, Any]
    data_root: Path
    output_root: Path
    mask_cache: Path
    experiments: dict[str, Any]
    baselines: list[str]
    external_repos: dict[str, str]

    def data_path(self, rel: str) -> Path:
        return resolve(self.data_root / rel)


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    config_path = Path(path) if path else CONFIGS_DIR / "experiments.yaml"
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return ExperimentConfig(
        raw=raw,
        data_root=resolve(raw["data_root"]),
        output_root=resolve(raw.get("output_root", "results")),
        mask_cache=resolve(raw.get("mask_cache", "data/masks")),
        experiments=raw["experiments"],
        baselines=raw.get("baselines", []),
        external_repos=raw.get("external_repos", {}),
    )
