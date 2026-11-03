"""
snapshot.py — Fast Git provenance, environment freezing, and source code snapshotting.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from hermes_tools.core import write_text_lf

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

MAX_FILE_BYTES = 15 * 1024 * 1024  # 15 MB per file copied (larger files are symlinked)
MAX_TOTAL_BYTES = 100 * 1024 * 1024  # 100 MB total copied cap (remaining files are symlinked)
MAX_SNAPSHOT_FILES = 2500


def _find_git_root(cwd: Path) -> Optional[Path]:
    """Finds the git repository root in pure Python without spawning a subprocess."""
    curr = cwd.resolve()
    while True:
        if (curr / ".git").exists():
            return curr
        if curr.parent == curr:
            return None
        curr = curr.parent


def get_git_info(cwd: Path) -> Dict[str, object]:
    """
    Extracts Git repository status, commit hash, branch, and uncommitted diff
    using a single `git status --porcelain=v2 --branch` call (and `git diff` only if dirty).
    """
    info: Dict[str, object] = {
        "is_git": False,
        "commit": None,
        "branch": None,
        "dirty": False,
        "root": None,
        "diff": "",
        "status": "",
    }
    git_root = _find_git_root(cwd)
    if git_root is None:
        return info

    try:
        st_res = subprocess.run(
            ["git", "status", "--porcelain=v2", "--branch"],
            cwd=git_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        info["is_git"] = True
        info["root"] = str(git_root)

        commit = "UNCOMMITTED"
        branch = "detached"
        status_lines: List[str] = []
        has_tracked_changes = False

        for line in st_res.stdout.splitlines():
            if line.startswith("# branch.oid "):
                oid = line[13:].strip()
                if oid and oid != "(initial)":
                    commit = oid
            elif line.startswith("# branch.head "):
                head = line[14:].strip()
                if head and head != "(detached)":
                    branch = head
            elif not line.startswith("#"):
                status_lines.append(line)
                if line.startswith(("1 ", "2 ", "u ")):
                    has_tracked_changes = True

        info["commit"] = commit
        info["branch"] = branch
        info["status"] = "\n".join(status_lines)
        info["dirty"] = bool(status_lines)

        if has_tracked_changes:
            diff_cmd = (
                ["git", "diff", "HEAD"] if commit != "UNCOMMITTED" else ["git", "diff"]
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


def _symlink_fallback(src_file: Path, dest_file: Path) -> None:
    """Creates a symlink at dest_file pointing to src_file so relative paths never break."""
    try:
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        if dest_file.exists() or dest_file.is_symlink():
            dest_file.unlink()
        os.symlink(src_file.resolve(), dest_file)
    except OSError:
        pass


def _copy_or_link_entry(
    src_file: Path,
    dest_file: Path,
    allow_copy: bool = True,
    max_bytes: int = MAX_FILE_BYTES,
) -> Tuple[bool, int]:
    """
    Copies regular files (<= max_bytes when allow_copy is True) and symlinks larger/capped
    files or existing symlinks so `Path(__file__).parent` relative lookups always work.
    Returns (copied_bytes_flag, size_bytes).
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
        except OSError:
            pass
        return False, 0

    if not stat.S_ISREG(st.st_mode):
        return False, 0

    if not allow_copy or st.st_size > max_bytes:
        _symlink_fallback(src_file, dest_file)
        return False, 0

    dest_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_file, dest_file)
    return True, st.st_size


def create_code_snapshot(
    target_path: Path, cwd: Path, repro_dir: Path, git_info: Dict[str, object]
) -> Tuple[Path, Optional[Path]]:
    """
    Copies source files (<=15MB each, <=100MB total) into repro/code/ and symlinks larger
    files so editing the workspace while a job is queued never breaks the run, and relative
    `Path(__file__)` lookups to datasets/weights still resolve cleanly.
    """
    code_dir = repro_dir / "code"
    if code_dir.exists():
        shutil.rmtree(code_dir, ignore_errors=True)
    code_dir.mkdir(parents=True, exist_ok=True)
    runs_root = (cwd / "runs").resolve()
    repro_resolved = repro_dir.resolve()

    if git_info.get("is_git") and git_info.get("root"):
        git_root = Path(str(git_info["root"])).resolve()
        git_runs_root = (git_root / "runs").resolve()
        try:
            # Single git ls-files invocation with -t to distinguish tracked ('H') vs untracked ('?')
            ls_res = subprocess.run(
                ["git", "ls-files", "-t", "--cached", "--others", "--exclude-standard"],
                cwd=git_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=True,
            )
            tracked_files: List[str] = []
            untracked_files: List[str] = []
            seen: Set[str] = set()
            for raw_line in ls_res.stdout.splitlines():
                if len(raw_line) < 3:
                    continue
                tag, rel_f = raw_line[0], raw_line[2:].strip()
                if not rel_f or rel_f in seen:
                    continue
                seen.add(rel_f)
                if tag == "?":
                    untracked_files.append(rel_f)
                else:
                    tracked_files.append(rel_f)

            total_bytes = 0
            total_files = 0

            for rel_file in tracked_files + untracked_files:
                rel_path = Path(rel_file)
                if any(part in _IGNORED_PARTS for part in rel_path.parts):
                    continue
                src_file = git_root / rel_path
                if (
                    src_file.is_relative_to(runs_root)
                    or src_file.is_relative_to(git_runs_root)
                    or src_file.is_relative_to(repro_resolved)
                ):
                    continue
                allow_copy = (
                    total_files < MAX_SNAPSHOT_FILES and total_bytes < MAX_TOTAL_BYTES
                )
                dest_file = code_dir / rel_path
                copied, nbytes = _copy_or_link_entry(
                    src_file, dest_file, allow_copy=allow_copy
                )
                if copied:
                    total_files += 1
                    total_bytes += nbytes

            if target_path.is_relative_to(git_root):
                rel_target = target_path.relative_to(git_root)
                snap_target = code_dir / rel_target
                snap_target.parent.mkdir(parents=True, exist_ok=True)
                if snap_target.is_symlink():
                    snap_target.unlink()
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
