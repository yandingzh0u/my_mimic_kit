"""Resolve paths recorded before the paper assets were consolidated.

Historical experiment manifests, frozen sources and checkpoint metadata remain
unchanged. Only their current filesystem locations are translated here.
"""
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RELOCATIONS = {
    "output/paper_benchmark": "output",
    "PAPER/experiments": "output",
    "output/paper_screenshots": "PAPER/artifacts/screenshots",
    "output/paper_rollouts": "PAPER/artifacts/rollouts",
    "output/paper_builds": "PAPER/build",
    "output/paper_cleanup": "PAPER/backups",
    "output/paper_screenshot_ppt_check": "PAPER/build/screenshot_ppt_check",
    "tools/paper_benchmark": "PAPER/tools/benchmark",
    "tools/paper_eval": "PAPER/tools/evaluation",
    "data/envs/paper_benchmark": "PAPER/configs/envs",
    "args/paper_benchmark": "PAPER/configs/args",
}


def resolve_relocated_path(value: str | Path) -> Path:
    """Return the current absolute location of a recorded repository path."""
    path = Path(value).expanduser()
    if path.is_absolute():
        try:
            relative = path.relative_to(REPO_ROOT)
        except ValueError:
            return path
    else:
        relative = path
    for old, new in RELOCATIONS.items():
        try:
            suffix = relative.relative_to(old)
        except ValueError:
            continue
        relative = Path(new) / suffix
        break
    parts = relative.parts
    if len(parts) > 1 and parts[0] == "output" and parts[1].startswith("dare_"):
        relative = Path(parts[0], "mira_" + parts[1][5:], *parts[2:])
    return REPO_ROOT / relative
