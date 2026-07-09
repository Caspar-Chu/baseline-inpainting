from __future__ import annotations

from baseline_suite.baselines.base import BaseInpainter
from baseline_suite.baselines.ddrm import DdrmInpainter
from baseline_suite.baselines.identity import IdentityInpainter
from baseline_suite.baselines.lama import LamaInpainter
from baseline_suite.baselines.mat import MatInpainter
from baseline_suite.baselines.repaint import RePaintInpainter

_REGISTRY: dict[str, type[BaseInpainter]] = {
    "identity": IdentityInpainter,
    "lama": LamaInpainter,
    "mat": MatInpainter,
    "repaint": RePaintInpainter,
    "ddrm": DdrmInpainter,
}


def available_baselines() -> list[str]:
    return sorted(_REGISTRY.keys())


def create_baseline(name: str, **kwargs) -> BaseInpainter:
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown baseline '{name}'. Available: {available_baselines()}")
    return _REGISTRY[key](**kwargs)
