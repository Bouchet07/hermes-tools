"""
snapshot.py — Git provenance, environment freezing, and source code snapshotting.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple


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
            check=True,
        )
        git_root = Path(root_res.stdout.strip()).resolve()
        info["is_git"] = True
        info["root"] = str(git_root)

        commit_res = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=git_root, capture_output=True, text=True
        )
        info["commit"] = (
            commit_res.stdout.strip() if commit_res.returncode == 0 else "UNCOMMITTED"
        )

        branch_res = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=git_root,
            capture_output=True,
            text=True,
        )
        info["branch"] = (
            branch_res.stdout.strip() if branch_res.returncode == 0 else "detached"
        )

        status_res = subprocess.run(
            ["git", "status", "--short"], cwd=git_root, capture_output=True, text=True
        )
        info["status"] = status_res.stdout.strip()
        info["dirty"] = bool(info["status"])

        diff_cmd = (
            ["git", "diff", "HEAD"]
            if info["commit"] != "UNCOMMITTED"
            else ["git", "diff"]
        )
        diff_res = subprocess.run(diff_cmd, cwd=git_root, capture_output=True, text=True)
        info["diff"] = diff_res.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return info


def _copy_entry_safely(
    src_file: Path, dest_file: Path, max_bytes: int = 15 * 1024 * 1024
) -> bool:
    """Copies regular files (<= max_bytes) and recreates symlinks without dereferencing."""
    try:
        st = src_file.lstat()
    except OSError:
        return False

    if stat.S_ISLNK(st.st_mode):
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        if dest_file.exists() or dest_file.is_symlink():
            dest_file.unlink()
        os.symlink(os.readlink(src_file), dest_file)
        return True

    if not stat.S_ISREG(st.st_mode) or st.st_size > max_bytes:
        return False

    dest_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_file, dest_file)
    return True


def create_code_snapshot(
    target_path: Path, cwd: Path, repro_dir: Path, git_info: Dict[str, object]
) -> Tuple[Path, Optional[Path]]:
    """
    Copies git-tracked and untracked source files (<15MB each) into repro/code/
    so editing the workspace while a job is queued never breaks the run.
    Used by both `send` (initial snapshot) and `runs` (sync / retry --update-code).
    """
    code_dir = repro_dir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    runs_root = cwd / "runs"

    if git_info.get("is_git") and git_info.get("root"):
        git_root = Path(str(git_info["root"]))
        git_runs_root = git_root / "runs"
        try:
            res = subprocess.run(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                cwd=git_root,
                capture_output=True,
                text=True,
                check=True,
            )
            files = [f.strip() for f in res.stdout.splitlines() if f.strip()]
            for rel_file in files:
                rel_path = Path(rel_file)
                if ".git" in rel_path.parts:
                    continue
                src_file = git_root / rel_path
                if (
                    src_file.is_relative_to(runs_root)
                    or src_file.is_relative_to(git_runs_root)
                    or (rel_path.parts and rel_path.parts[0] == "runs")
                ):
                    continue
                dest_file = code_dir / rel_path
                _copy_entry_safely(src_file, dest_file)

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
    pip_bin = venv_path / "bin" / "pip"
    try:
        if shutil.which("uv"):
            freeze_res = subprocess.run(
                ["uv", "pip", "freeze", "--python", str(venv_path / "bin" / "python")],
                capture_output=True,
                text=True,
            )
        elif pip_bin.is_file():
            freeze_res = subprocess.run(
                [str(pip_bin), "freeze"], capture_output=True, text=True
            )
        else:
            freeze_res = None

        if freeze_res and freeze_res.returncode == 0:
            (repro_dir / "env.txt").write_text(freeze_res.stdout)
    except Exception:
        pass