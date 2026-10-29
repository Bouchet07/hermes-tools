"""
snapshot.py — Git provenance, environment freezing, and source code snapshotting.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

from hermes_tools.core import console, write_text_lf

_IGNORED_PARTS: Set[str] = {
    ".git",
    "runs",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    "wandb",
}

MAX_FILE_BYTES = 15 * 1024 * 1024  # 15 MB per file
MAX_TOTAL_BYTES = 100 * 1024 * 1024  # 100 MB total snapshot cap
MAX_SNAPSHOT_FILES = 2500


def get_git_info(cwd: Path) -> Dict[str, object]:
    """Extracts Git repository status, commit hash, branch, and uncommitted diff."""
    info: Dict[str, object] = {
        "is_git": False,
        "commit": None,
        "branch": None,
        "dirty": False,
        "root": None,
        "diff": "",
        "status": "",
    }
    try:
        root_res = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        git_root = Path(root_res.stdout.strip()).resolve()
        info["is_git"] = True
        info["root"] = str(git_root)

        commit_res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=git_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        info["commit"] = (
            commit_res.stdout.strip() if commit_res.returncode == 0 else "UNCOMMITTED"
        )

        branch_res = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=git_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        info["branch"] = (
            branch_res.stdout.strip() if branch_res.returncode == 0 else "detached"
        )

        status_res = subprocess.run(
            ["git", "status", "--short"],
            cwd=git_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        info["status"] = status_res.stdout.strip()
        info["dirty"] = bool(info["status"])

        diff_cmd = (
            ["git", "diff", "HEAD"]
            if info["commit"] != "UNCOMMITTED"
            else ["git", "diff"]
        )
        diff_res = subprocess.run(
            diff_cmd,
            cwd=git_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        info["diff"] = diff_res.stdout
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass

    return info


def _copy_entry_safely(
    src_file: Path, dest_file: Path, max_bytes: int = MAX_FILE_BYTES
) -> Tuple[bool, int]:
    """
    Copies regular files (<= max_bytes) and recreates symlinks without dereferencing.
    Returns (copied, size_bytes).
    """
    try:
        st = src_file.lstat()
    except OSError:
        return False, 0

    if stat.S_ISLNK(st.st_mode):
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            if dest_file.exists() or dest_file.is_symlink():
                dest_file.unlink()
            os.symlink(os.readlink(src_file), dest_file)
            return True, 0
        except OSError:
            return False, 0

    if not stat.S_ISREG(st.st_mode) or st.st_size > max_bytes:
        return False, 0

    dest_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_file, dest_file)
    return True, st.st_size


def create_code_snapshot(
    target_path: Path, cwd: Path, repro_dir: Path, git_info: Dict[str, object]
) -> Tuple[Path, Optional[Path]]:
    """
    Copies git-tracked and untracked source files (<=15MB each, <=100MB total) into repro/code/
    so editing the workspace while a job is queued never breaks the run.
    Used by both `send` (initial snapshot) and `runs` (sync / retry --update-code).
    """
    code_dir = repro_dir / "code"
    if code_dir.exists():
        shutil.rmtree(code_dir, ignore_errors=True)
    code_dir.mkdir(parents=True, exist_ok=True)
    runs_root = (cwd / "runs").resolve()

    if git_info.get("is_git") and git_info.get("root"):
        git_root = Path(str(git_info["root"])).resolve()
        git_runs_root = (git_root / "runs").resolve()
        try:
            # Snapshot tracked files first so source code always takes priority over untracked data
            tracked_res = subprocess.run(
                ["git", "ls-files", "--cached"],
                cwd=git_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=True,
            )
            untracked_res = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard"],
                cwd=git_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=True,
            )
            tracked_files = [f.strip() for f in tracked_res.stdout.splitlines() if f.strip()]
            tracked_set = set(tracked_files)
            untracked_files = [
                f.strip()
                for f in untracked_res.stdout.splitlines()
                if f.strip() and f.strip() not in tracked_set
            ]

            total_bytes = 0
            total_files = 0
            skipped_cap = 0

            for rel_file in tracked_files + untracked_files:
                rel_path = Path(rel_file)
                if any(part in _IGNORED_PARTS for part in rel_path.parts):
                    continue
                src_file = git_root / rel_path
                if (
                    src_file.is_relative_to(runs_root)
                    or src_file.is_relative_to(git_runs_root)
                    or src_file.is_relative_to(repro_dir.resolve())
                ):
                    continue
                if total_files >= MAX_SNAPSHOT_FILES or total_bytes >= MAX_TOTAL_BYTES:
                    skipped_cap += 1
                    continue
                dest_file = code_dir / rel_path
                ok, nbytes = _copy_entry_safely(src_file, dest_file)
                if ok:
                    total_files += 1
                    total_bytes += nbytes

            if skipped_cap > 0:
                console.print(
                    f"[dim yellow]ℹ Snapshot cap reached ({MAX_TOTAL_BYTES // (1024 * 1024)} MB / {MAX_SNAPSHOT_FILES} files); "
                    f"skipped {skipped_cap} extra file(s). Add large data folders to .gitignore.[/dim yellow]"
                )

            if target_path.is_relative_to(git_root):
                rel_target = target_path.relative_to(git_root)
                snap_target = code_dir / rel_target
                snap_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target_path, snap_target)
                return snap_target, code_dir
        except Exception:
            pass

    snap_target = code_dir / target_path.name
    shutil.copy2(target_path, snap_target)
    return snap_target, code_dir


def freeze_python_env(venv_path: Optional[Path], repro_dir: Path) -> None:
    """Writes `repro/env.txt` using `uv pip freeze` or `pip freeze`."""
    if not venv_path:
        return
    py_bin = (
        venv_path / "bin" / "python"
        if (venv_path / "bin" / "python").is_file()
        else venv_path / "Scripts" / "python.exe"
    )
    pip_bin = (
        venv_path / "bin" / "pip"
        if (venv_path / "bin" / "pip").is_file()
        else venv_path / "Scripts" / "pip.exe"
    )
    try:
        if shutil.which("uv") and py_bin.is_file():
            freeze_res = subprocess.run(
                ["uv", "pip", "freeze", "--python", str(py_bin)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        elif pip_bin.is_file():
            freeze_res = subprocess.run(
                [str(pip_bin), "freeze"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        else:
            freeze_res = None

        if freeze_res and freeze_res.returncode == 0:
            write_text_lf(repro_dir / "env.txt", freeze_res.stdout)
    except Exception:
        pass
