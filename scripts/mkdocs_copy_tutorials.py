"""MkDocs hook: mirror understanding_its2s notebooks into docs/tutorials/ before build.

The canonical home of the vignette notebooks is ``understanding_its2s/`` (versioned
alongside the source). This hook copies them into ``docs/tutorials/`` (gitignored)
right before MkDocs / mkdocs-jupyter renders them. Runs for both ``mkdocs serve``
(local) and ``mkdocs build`` (CI), so editing a notebook in either location stays
in sync with the deployed site.
"""

from __future__ import annotations

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "understanding_its2s"
DST = REPO_ROOT / "docs" / "tutorials"


def _sync_dir(src: Path, dst: Path, suffixes: tuple[str, ...]) -> None:
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for entry in src.iterdir():
        if entry.is_file() and entry.suffix in suffixes:
            shutil.copy2(entry, dst / entry.name)


def on_pre_build(config):  # noqa: D401 -- MkDocs hook signature
    """Copy notebooks and the figures/ directory into docs/tutorials/."""
    # Only ship notebooks. Markdown files (e.g. understanding_its2s/README.md)
    # would otherwise land in docs/tutorials/ and trip mkdocs --strict because
    # they are not listed in the nav.
    _sync_dir(SRC, DST, suffixes=(".ipynb",))
    _sync_dir(SRC / "figures", DST / "figures", suffixes=(".png", ".jpg", ".jpeg", ".svg"))
    return config
