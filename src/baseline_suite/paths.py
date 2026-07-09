from __future__ import annotations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
EXTERNAL_DIR = PROJECT_ROOT / "external"


def resolve(path: str | Path, *, root: Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    base = root or PROJECT_ROOT
    return (base / p).resolve()


def ensure_dirs() -> None:
    for d in (DATA_DIR, RESULTS_DIR, EXTERNAL_DIR):
        d.mkdir(parents=True, exist_ok=True)
