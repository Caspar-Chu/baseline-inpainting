from __future__ import annotations

import os
import sys
from pathlib import Path

from baseline_suite.paths import EXTERNAL_DIR


def purge_modules(purge_roots: tuple[str, ...]) -> None:
    """Remove cached top-level packages that collide across external repos."""
    to_delete = [
        name
        for name in list(sys.modules)
        if name in purge_roots or any(name.startswith(f"{root}.") for root in purge_roots)
    ]
    for name in to_delete:
        if "." in name:
            parent_name, attr = name.rsplit(".", 1)
            parent = sys.modules.get(parent_name)
            if parent is not None:
                parent.__dict__.pop(attr, None)
    for name in sorted(to_delete, key=lambda item: item.count("."), reverse=True):
        sys.modules.pop(name, None)


def activate_external_repo(repo_path: Path, *, purge_roots: tuple[str, ...]) -> str:
    """Put one external repo first on sys.path and purge conflicting imports."""
    repo = str(repo_path.resolve())
    external_root = str(EXTERNAL_DIR.resolve())

    cleaned: list[str] = []
    for entry in sys.path:
        if entry == repo:
            continue
        try:
            resolved = str(Path(entry).resolve())
        except OSError:
            cleaned.append(entry)
            continue
        if resolved.startswith(external_root + os.sep):
            continue
        cleaned.append(entry)

    sys.path[:] = [repo, *cleaned]
    purge_modules(purge_roots)
    return repo


def assert_model_from_repo(model: object, repo_path: Path, *, label: str) -> None:
    """Fail fast when a cached import loaded the wrong UNet implementation."""
    import inspect

    mod_file = Path(inspect.getfile(type(model))).resolve()
    repo = repo_path.resolve()
    try:
        mod_file.relative_to(repo)
    except ValueError as exc:
        raise RuntimeError(
            f"{label} model class loaded from {mod_file}, expected under {repo}. "
            "Another baseline likely polluted sys.modules; try running baselines "
            "one at a time or update baseline_suite."
        ) from exc
