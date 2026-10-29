"""
core.py — Shared UI constants, atomic JSON locking, and path utilities for hermes-tools.
"""

from __future__ import annotations

import fcntl
import itertools
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from questionary import Style
from rich.console import Console

console = Console()

PROMPT_STYLE = Style(
    [
        ("qmark", "fg:#00d7ff bold"),
        ("question", "bold"),
        ("answer", "fg:#00ff87 bold"),
        ("pointer", "fg:#00d7ff bold"),
        ("highlighted", "fg:#00d7ff bold"),
        ("selected", "fg:#00ff87"),
        ("separator", "fg:#6c6c6c"),
        ("instruction", "fg:#808080 italic"),
    ]
)

STATUS_STYLES: Dict[str, str] = {
    "COMPLETED": "bold green",
    "RUNNING": "bold cyan",
    "QUEUED": "bold yellow",
    "HELD": "bold magenta",
    "FAILED": "bold red",
    "COMPLETED_WITH_ERRORS": "bold yellow",
    "INCOMPLETE": "bold yellow",
    "INTERRUPTED": "bold magenta",
    "ABORTED": "dim red",
    "VANISHED": "bold red",
    "SUBMIT_FAILED": "bold red",
    "DRY_RUN": "dim cyan",
}


def style_status(raw_status: str) -> str:
    base = raw_status.split(" ")[0]
    color = STATUS_STYLES.get(base, "white")
    return f"[{color}]{raw_status}[/{color}]"


# ---------------------------------------------------------------------------
# Atomic JSON & File Locking (NFS / CephFS Safe)
# ---------------------------------------------------------------------------
def atomic_write_json(target_path: Path, data: Dict[str, Any], tmp_dir: Path) -> None:
    """Writes JSON to a same-filesystem temp file and atomically replaces target_path."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f".{target_path.name}.tmp.{os.getpid()}"
    tmp_path.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp_path, target_path)


def locked_mutate_run_json(
    run_json_path: Path,
    repro_dir: Path,
    mutator_fn: Callable[[Dict[str, Any]], None],
) -> Dict[str, Any]:
    """Updates run.json under an exclusive flock without unlinking the lock inode."""
    repro_dir.mkdir(parents=True, exist_ok=True)
    lock_path = repro_dir / ".run_json.lock"
    with open(lock_path, "a") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            data = json.loads(run_json_path.read_text()) if run_json_path.exists() else {}
            mutator_fn(data)
            atomic_write_json(run_json_path, data, repro_dir)
            return data
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def load_run_json(run_dir: Path) -> Dict[str, Any]:
    rj = run_dir / "run.json"
    if not rj.is_file():
        return {"run_id": run_dir.name, "status": "UNKNOWN"}
    try:
        return json.loads(rj.read_text())
    except Exception:
        return {"run_id": run_dir.name, "status": "CORRUPTED"}


# ---------------------------------------------------------------------------
# Run Directory & Symlink Helpers
# ---------------------------------------------------------------------------
def get_runs_root(cwd: Path) -> Path:
    curr = cwd.resolve()
    for _ in range(4):
        if (curr / "runs").is_dir():
            return curr / "runs"
        if (curr / ".git").exists() or curr.parent == curr:
            break
        curr = curr.parent
    return cwd / "runs"


def list_run_dirs(runs_root: Path) -> List[Path]:
    if not runs_root.is_dir():
        return []
    dirs = [
        p
        for p in runs_root.iterdir()
        if p.is_dir() and not p.is_symlink() and not p.name.startswith(".")
    ]
    dirs.sort(key=lambda p: p.name, reverse=True)
    return dirs


def update_latest_symlink(runs_root: Path, target_run_dir: Optional[Path] = None) -> None:
    """Atomically updates or repairs `runs/latest` symlink."""
    latest = runs_root / "latest"
    if target_run_dir is None:
        dirs = list_run_dirs(runs_root)
        if not dirs:
            if latest.is_symlink() or latest.exists():
                latest.unlink(missing_ok=True)
            return
        if latest.is_symlink() and (runs_root / os.readlink(latest)).exists():
            return
        target_run_dir = dirs[0]

    tmp_link = runs_root / f".latest.tmp.{os.getpid()}"
    try:
        tmp_link.unlink(missing_ok=True)
        tmp_link.symlink_to(target_run_dir.name)
        os.replace(tmp_link, latest)
    except OSError:
        tmp_link.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Parsing, Formatting & Sweep Expansion
# ---------------------------------------------------------------------------
def parse_memory_to_mb(mem_str: str) -> int:
    """Converts strings like '16G', '16GB', '1T', '4096M', or '4096' into integer MB."""
    s = str(mem_str).strip().upper()
    if not s:
        raise ValueError("Empty memory specification.")

    multipliers = (
        ("TB", 1024 * 1024),
        ("T", 1024 * 1024),
        ("GB", 1024),
        ("G", 1024),
        ("MB", 1),
        ("M", 1),
    )
    for suffix, mult in multipliers:
        if s.endswith(suffix):
            val_mb = int(float(s[: -len(suffix)]) * mult)
            break
    else:
        val_mb = int(float(s))

    if val_mb <= 0:
        raise ValueError("Memory request must be greater than 0 MB.")
    return val_mb


def is_valid_memory_input(val: str, allow_empty: bool = False) -> bool:
    s = val.strip()
    if not s:
        return allow_empty
    try:
        parse_memory_to_mb(s)
        return True
    except ValueError:
        return False


def sanitize_tag(raw_tag: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_tag.strip())
    return cleaned.strip("_")


def format_duration(seconds: Optional[int]) -> str:
    if seconds is None or seconds < 0:
        return "—"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def dir_size_mb(path: Path) -> float:
    total = 0
    for root, _, files in os.walk(path, followlinks=False):
        for f in files:
            fp = Path(root) / f
            try:
                st = fp.lstat()
                if stat.S_ISREG(st.st_mode):
                    total += st.st_size
            except OSError:
                pass
    return total / (1024 * 1024)


def summarize_metrics(data: Dict[str, Any]) -> Tuple[str, str, str]:
    """Returns (host_summary, runtime_summary, peak_ram_summary) across single or sweep runs."""
    exec_single = data.get("execution")
    exec_list = data.get("executions") or []
    records = [exec_single] if isinstance(exec_single, dict) else exec_list
    if not records:
        return "—", "—", "—"

    hosts = sorted({r.get("hostname", "").split(".")[0] for r in records if r.get("hostname")})
    host_str = ", ".join(hosts[:2]) + (f" (+{len(hosts) - 2})" if len(hosts) > 2 else "")
    if not host_str:
        host_str = "—"

    max_dur = max((int(r.get("runtime_seconds", 0) or 0) for r in records), default=0)
    max_ram = max((float(r.get("peak_ram_mb", 0.0) or 0.0) for r in records), default=0.0)
    ram_str = (
        f"{max_ram / 1024:.2f} GB"
        if max_ram >= 1024
        else (f"{max_ram:.0f} MB" if max_ram > 0 else "—")
    )
    return host_str, format_duration(max_dur), ram_str


def _expand_single_brace_expr(expr: str) -> List[str]:
    pattern = re.compile(r"\{([^{}]+)\}")
    matches = list(pattern.finditer(expr))
    if not matches:
        return [expr]

    options_per_match: List[List[str]] = []
    for m in matches:
        inner = m.group(1).strip()
        range_match = re.match(r"^(-?\d+)\.\.(-?\d+)(?:\.\.(-?\d+))?$", inner)
        if range_match:
            start, end = int(range_match.group(1)), int(range_match.group(2))
            raw_step = int(range_match.group(3) or 1)
            step = max(1, abs(raw_step))
            if start <= end:
                options_per_match.append([str(i) for i in range(start, end + 1, step)])
            else:
                options_per_match.append([str(i) for i in range(start, end - 1, -step)])
        elif "," in inner:
            options_per_match.append([x.strip() for x in inner.split(",")])
        else:
            options_per_match.append([m.group(0)])

    expanded: List[str] = []
    for combo in itertools.product(*options_per_match):
        res = expr
        for m, val in reversed(list(zip(matches, combo))):
            res = res[: m.start()] + val + res[m.end() :]
        expanded.append(res)
    return expanded


def expand_arg_sweeps(expr: str) -> List[str]:
    """Expands brace expressions ('{0..4}', '{a,b}') and/or ';' separators into a sweep list."""
    if not expr.strip():
        return [""]
    segments = [part.strip() for part in expr.split(";") if part.strip()]
    out: List[str] = []
    for seg in segments:
        out.extend(_expand_single_brace_expr(seg))
    return out or [""]


def discover_project_venv(cwd: Path, explicit_venv: Optional[str] = None) -> Optional[Path]:
    """Auto-discovers the active or local project venv, ignoring the CLI tool's own venv."""
    tool_venv = Path(sys.executable).resolve().parent.parent

    if explicit_venv:
        p = Path(explicit_venv).expanduser().resolve()
        if (p / "bin" / "activate").is_file():
            return p
        console.print(f"[bold red]✘ Explicit venv not valid:[/bold red] {p}")
        sys.exit(1)

    env_var = os.environ.get("VIRTUAL_ENV")
    if env_var:
        p = Path(env_var).resolve()
        if p != tool_venv and (p / "bin" / "activate").is_file():
            return p

    curr = cwd.resolve()
    for _ in range(4):
        for candidate in (".venv", "venv", "env"):
            venv_candidate = curr / candidate
            if (
                venv_candidate.is_dir()
                and venv_candidate.resolve() != tool_venv
                and (venv_candidate / "bin" / "activate").is_file()
            ):
                return venv_candidate.resolve()
        if (curr / ".git").exists() or curr.parent == curr:
            break
        curr = curr.parent

    return None