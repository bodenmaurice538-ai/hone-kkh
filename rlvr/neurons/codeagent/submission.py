"""Turn Claude Code's workspace edits into a validated submission.

Repository tasks: the edited copy is diffed against the pristine workspace with
the same Git recipe the model path uses (:meth:`Workbench._patch`), so the bytes
are a ``unified_diff_v1`` the validator's ``git apply -p1`` accepts. Only files
under the task's working directory are considered, and the diff paths are made
relative to it.

Terminal tasks: Claude Code writes its answer script to a fixed path at the
workspace root; its bytes are the ``bash_script_v1`` submission.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

# Where a terminal solve must leave its script, relative to the workspace root.
TERMINAL_SCRIPT_NAME = ".hone_solution.sh"

_SKIP_DIRS = {".git", ".hg", "node_modules", "__pycache__"}
_MAX_CHANGED_FILES = 2_000
_MAX_FILE_BYTES = 8 * 1024 * 1024


def _rel_under(working_directory: str) -> str:
    wd = working_directory.strip().strip("/")
    return "" if wd in ("", ".") else wd


def changed_files(pristine: Path, edited: Path, working_directory: str) -> list[str]:
    """Root-relative paths that are new or modified under the working directory.

    Deleted files are not reported: the diff recipe, like the model path, does not
    express deletions. Binary and oversized files are skipped.
    """
    under = _rel_under(working_directory)
    scope = edited / under if under else edited
    result: list[str] = []
    for current, dirs, files in os.walk(scope):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for name in files:
            path = Path(current) / name
            if path.is_symlink() or not path.is_file():
                continue
            try:
                if path.stat().st_size > _MAX_FILE_BYTES:
                    continue
                new_bytes = path.read_bytes()
            except OSError:
                continue
            relative = path.relative_to(edited).as_posix()
            original = pristine / relative
            try:
                old_bytes = original.read_bytes() if original.is_file() else None
            except OSError:
                old_bytes = None
            if old_bytes == new_bytes:
                continue
            result.append(relative)
            if len(result) > _MAX_CHANGED_FILES:
                raise ValueError("too many changed files; refusing to build a diff")
    return sorted(result)


def git_diff(pristine: Path, edited: Path, touched: list[str], working_directory: str, *, scratch_dir: Path) -> bytes:
    """A ``unified_diff_v1`` of ``touched`` files, paths relative to the working directory.

    Mirrors the model path's Git recipe: build a one-commit repo from the pristine
    versions, swap in the edited versions, and emit the staged diff. Using a fresh
    index forces Git to read every file instead of trusting size and mtime.
    """
    if not touched:
        return b""
    git = shutil.which("git")
    if git is None:
        raise ValueError("git is not available")
    under = _rel_under(working_directory)

    def strip(relative: str) -> str:
        if not under:
            return relative
        prefix = under + "/"
        return relative[len(prefix):] if relative.startswith(prefix) else relative

    with tempfile.TemporaryDirectory(dir=scratch_dir) as temporary:
        repository, home = Path(temporary, "repository"), Path(temporary, "home")
        repository.mkdir()
        home.mkdir()
        environment = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": str(home),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "LC_ALL": "C.UTF-8",
        }

        def run(*arguments: str) -> bytes:
            return subprocess.run(
                [git, "-c", "core.autocrlf=false", "-c", "user.name=miner", "-c", "user.email=miner@invalid", *arguments],
                cwd=repository, env=environment, stdin=subprocess.DEVNULL,
                capture_output=True, timeout=30, check=True,
            ).stdout

        def place(tree: Path) -> None:
            for relative in touched:
                source = tree / relative
                if source.is_file():
                    destination = repository / strip(relative)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)

        run("init", "-q", "--template=")
        place(pristine)
        run("add", "-f", "-A")
        run("commit", "-q", "--allow-empty", "-m", "workspace")
        (repository / ".git" / "index").unlink()
        place(edited)
        run("add", "-f", "-A")
        return run("diff", "--cached", "--no-renames", "--no-color", "--no-ext-diff", "HEAD")


def read_terminal_script(edited: Path) -> bytes:
    """The script a terminal solve was asked to write, or empty if it wrote none."""
    path = edited / TERMINAL_SCRIPT_NAME
    if not path.is_file() or path.is_symlink():
        return b""
    try:
        return path.read_bytes()
    except OSError:
        return b""
