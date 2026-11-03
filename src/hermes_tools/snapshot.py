"""
snapshot.py — Fast Git provenance, environment freezing, and source code snapshotting.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
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

# Binary, dataset, checkpoint, and media extensions that should always be symlinked (never copied)
_SYMLINK_ONLY_EXTENSIONS: Set[str] = {
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".onnx",
    ".bin",
    ".h5",
    ".hdf5",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
    ".joblib",
    ".parquet",
    ".feather",
    ".arrow",
    ".csv",
    ".tsv",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".tar",
    ".gz",
    ".tgz",
    ".bz2",
    ".xz",
    ".zip",
    ".7z",
    ".zst",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".tiff",
    ".webp",
    ".svg",
    ".pdf",
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".so",
    ".dll",
    ".dylib",
    ".a",
    ".o",
    ".whl",
}

MAX_FILE_BYTES = 2 * 1024 * 1024  # 2 MB per source file copied (larger files are symlinked)
MAX_TOTAL_BYTES = 50 * 1024 * 1024  # 50 MB total copied cap (remaining files are symlinked)
MAX_SNAPSHOT_FILES = 1500


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


def _ensure_parent(dest_file: Path, created_dirs: Set[Path]) -> None:
    parent = dest_file.parent
    if parent not in created_dirs:
        parent.mkdir(parents=True, exist_ok=True)
        created_dirs.add(parent)


def _symlink_fallback(src_file: Path, dest_file: Path, created_dirs: Set[Path]) -> None:
    """Creates a symlink at dest_file pointing to src_file so relative paths never break."""
    try:
        _ensure_parent(dest_file, created_dirs)
        if dest_file.exists() or dest_file.is_symlink():
            dest_file.unlink()
        os.symlink(src_file.resolve(), dest_file)
    except OSError:
        pass


def _copy_or_link_entry(
    src_file: Path,
    dest_file: Path,
    created_dirs: Set[Path],
    allow_copy: bool = True,
    max_bytes: int = MAX_FILE_BYTES,
) -> Tuple[bool, int]:
    """
    Copies source/config files (<= max_bytes when allow_copy is True) using fast copyfile
    (skipping copystat RPCs) and symlinks binary/data/large files so `Path(__file__).parent`
    relative lookups always work.
    """
    if src_file.suffix.lower() in _SYMLINK_ONLY_EXTENSIONS:
        _symlink_fallback(src_file, dest_file, created_dirs)
        return False, 0

    try:
        st = src_file.lstat()
    except OSError:
        return False, 0

    if stat.S_ISLNK(st.st_mode):
        _ensure_parent(dest_file, created_dirs)
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
        _symlink_fallback(src_file, dest_file, created_dirs)
        return False, 0

    _ensure_parent(dest_file, created_dirs)
    shutil.copyfile(src_file, dest_file)
    return True, st.st_size


def create_code_snapshot(
    target_path: Path, cwd: Path, repro_dir: Path, git_info: Dict[str, object]
) -> Tuple[Path, Optional[Path]]:
    """
    Copies source/config files into repro/code/ and symlinks larger or binary/data files
    so editing code while a job is queued never breaks the run, and relative `Path(__file__)`
    lookups to datasets/weights still resolve cleanly.
    """
    code_dir = repro_dir / "code"
    if code_dir.exists():
        shutil.rmtree(code_dir, ignore_errors=True)
    code_dir.mkdir(parents=True, exist_ok=True)
    created_dirs: Set[Path] = {code_dir}
    runs_root = (cwd / "runs").resolve()
    repro_resolved = repro_dir.resolve()

    if git_info.get("is_git") and git_info.get("root"):
        git_root = Path(str(git_info["root"])).resolve()
        git_runs_root = (git_root / "runs").resolve()
        try:
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
                    src_file, dest_file, created_dirs, allow_copy=allow_copy
                )
                if copied:
                    total_files += 1
                    total_bytes += nbytes

            if target_path.is_relative_to(git_root):
                rel_target = target_path.relative_to(git_root)
                snap_target = code_dir / rel_target
                _ensure_parent(snap_target, created_dirs)
                if snap_target.is_symlink():
                    snap_target.unlink()
                shutil.copyfile(target_path, snap_target)
                return snap_target, code_dir
        except Exception:
            pass

    # Non-git fallback: fast scandir walk of cwd
    cwd_resolved = cwd.resolve()
    total_bytes = 0
    total_files = 0
    stack: List[Path] = [cwd_resolved]
    while stack:
        curr_dir = stack.pop()
        try:
            with os.scandir(curr_dir) as it:
                for entry in it:
                    if entry.name in _IGNORED_PARTS or entry.name.startswith(".hermes"):
                        continue
                    entry_path = Path(entry.path)
                    if entry_path == runs_root or entry_path == repro_resolved:
                        continue
                    rel_path = entry_path.relative_to(cwd_resolved)
                    dest_file = code_dir / rel_path
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry_path)
                    else:
                        allow_copy = (
                            total_files < MAX_SNAPSHOT_FILES
                            and total_bytes < MAX_TOTAL_BYTES
                        )
                        copied, nbytes = _copy_or_link_entry(
                            entry_path, dest_file, created_dirs, allow_copy=allow_copy
                        )
                        if copied:
                            total_files += 1
                            total_bytes += nbytes
        except OSError:
            pass

    if target_path.is_relative_to(cwd_resolved):
        snap_target = code_dir / target_path.relative_to(cwd_resolved)
    else:
        snap_target = code_dir / target_path.name
    _ensure_parent(snap_target, created_dirs)
    if snap_target.is_symlink():
        snap_target.unlink()
    shutil.copyfile(target_path, snap_target)
    return snap_target, code_dir


def _venv_state_mtime(venv_path: Path) -> float:
    """Returns the latest mtime of the venv's package directories to validate the env.txt cache."""
    latest = 0.0
    candidates = [venv_path / "pyvenv.cfg", venv_path / "Lib" / "site-packages"]
    lib_dir = venv_path / "lib"
    if lib_dir.is_dir():
        try:
            with os.scandir(lib_dir) as it:
                for entry in it:
                    if entry.name.startswith("python"):
                        candidates.append(Path(entry.path) / "site-packages")
        except OSError:
            pass
    for p in candidates:
        try:
            st = p.stat()
            if st.st_mtime > latest:
                latest = st.st_mtime
        except OSError:
            pass
    return latest


def freeze_python_env(
    venv_path: Optional[Path], repro_dir: Path, detach: bool = False
) -> None:
    """
    Writes `repro/env.txt` using a mtime-validated cache in `.venv/.hermes_env_cache.txt` (0ms on cache hit),
    or spawns `uv pip freeze` / `pip freeze` (optionally detached so `send` exits immediately).
    """
    if not venv_path:
        return
    target_env_txt = repro_dir / "env.txt"
    cache_file = venv_path / ".hermes_env_cache.txt"
    try:
        if cache_file.is_file():
            if cache_file.stat().st_mtime >= _venv_state_mtime(venv_path):
                shutil.copyfile(cache_file, target_env_txt)
                return
    except OSError:
        pass

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
    uv_bin = shutil.which("uv")
    if uv_bin and py_bin.is_file():
        freeze_cmd = [uv_bin, "pip", "freeze", "--python", str(py_bin)]
    elif pip_bin.is_file():
        freeze_cmd = [str(pip_bin), "freeze"]
    else:
        return

    if detach:
        helper_code = (
            "import shutil, subprocess, sys; "
            "r = subprocess.run(sys.argv[3:], capture_output=True, text=True, encoding='utf-8', errors='replace'); "
            "r.returncode == 0 and ("
            "open(sys.argv[1], 'w', encoding='utf-8', newline='\\n').write(r.stdout), "
            "shutil.copyfile(sys.argv[1], sys.argv[2]))"
        )
        kwargs: Dict[str, object] = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if os.name == "posix":
            kwargs["start_new_session"] = True
        elif hasattr(subprocess, "CREATE_NO_WINDOW"):
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            subprocess.Popen(
                [sys.executable, "-c", helper_code, str(target_env_txt), str(cache_file), *freeze_cmd],
                **kwargs,  # type: ignore[arg-type]
            )
            return
        except Exception:
            pass

    try:
        freeze_res = subprocess.run(
            freeze_cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if freeze_res.returncode == 0:
            write_text_lf(target_env_txt, freeze_res.stdout)
            try:
                shutil.copyfile(target_env_txt, cache_file)
            except OSError:
                pass
    except Exception:
        pass
