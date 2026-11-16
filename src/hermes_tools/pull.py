"""
pull.py — Incremental single-SSH compressed streaming sync (`runs pull` / `fetch` / `download`).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich import box
from rich.table import Table

from hermes_tools.core import (
    console,
    extract_run_display_parts,
    format_bytes as _format_bytes,
    list_run_dirs,
    load_project_config,
    load_run_json,
    style_status,
    update_latest_symlink,
)

_PULL_MANIFEST_PREFIX = b"HERMES_PULL_MANIFEST:"

_PULL_WEIGHT_EXTENSIONS = {
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".bin",
    ".onnx",
    ".h5",
    ".hdf5",
    ".tar",
    ".gz",
    ".tgz",
    ".bz2",
    ".xz",
    ".zip",
    ".7z",
    ".zst",
}

_REMOTE_PULL_SCRIPT = r'''
import fnmatch
import glob
import io
import json
import os
import socket
import stat
import sys
import tarfile
from pathlib import Path

MANIFEST_PREFIX = b"HERMES_PULL_MANIFEST:"
WEIGHT_EXTS = {
    ".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".onnx",
    ".h5", ".hdf5", ".tar", ".gz", ".tgz", ".bz2", ".xz",
    ".zip", ".7z", ".zst",
}
SKIP_DIRS = {
    ".git", ".venv", "venv", "env", "conda", "miniconda3", "anaconda3",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "node_modules", "wandb", ".cache", ".local", ".vscode", ".vscode-server",
}


class _SafeReader:
    def __init__(self, fh, size):
        self.fh = fh
        self.rem = size

    def read(self, n=-1):
        if self.rem <= 0:
            return b""
        to_read = self.rem if (n is None or n < 0 or n > self.rem) else n
        chunk = self.fh.read(to_read)
        if not chunk:
            pad = b"\x00" * to_read
            self.rem -= to_read
            return pad
        self.rem -= len(chunk)
        return chunk


def _emit_manifest(obj):
    sys.stdout.buffer.write(MANIFEST_PREFIX + json.dumps(obj).encode("utf-8") + b"\n")
    sys.stdout.buffer.flush()


def _strip_git_suffix(url):
    s = (url or "").strip().rstrip("/")
    if s.lower().endswith(".git"):
        s = s[:-4]
    return s.lower()


def _read_git_origin(repo_dir):
    cfg = repo_dir / ".git" / "config"
    if not cfg.is_file():
        return ""
    try:
        for line in cfg.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.strip()
            if s.startswith("url =") or s.startswith("url="):
                return _strip_git_suffix(s.split("=", 1)[1])
    except OSError:
        pass
    return ""


def _has_run_subdirs(runs_dir):
    if not runs_dir.is_dir():
        return False
    try:
        with os.scandir(runs_dir) as it:
            for e in it:
                if not e.name.startswith(".") and not e.is_symlink() and e.is_dir(follow_symlinks=False):
                    if (Path(e.path) / "run.json").is_file():
                        return True
    except OSError:
        pass
    return False


def _resolve_runs_root(req):
    explicit = (req.get("remote_dir") or "").strip()
    if explicit:
        expanded = os.path.expandvars(os.path.expanduser(explicit))
        p = Path(expanded)
        if not p.is_absolute():
            p = Path.home() / p
        if p.is_dir():
            if (p / "run.json").is_file():
                return p.parent.resolve(), [p.name], False
            if p.name == "runs" or _has_run_subdirs(p):
                return p.resolve(), None, False
            if (p / "runs").is_dir():
                return (p / "runs").resolve(), None, False
            return (p / "runs").resolve(), None, False
        return None, None, False

    for hint in req.get("hint_dirs") or []:
        if not hint:
            continue
        hp = Path(os.path.expandvars(os.path.expanduser(str(hint))))
        if not hp.is_absolute():
            hp = Path.home() / hp
        if hp.is_dir():
            if hp.name == "runs" or _has_run_subdirs(hp):
                return hp.resolve(), None, True
            if (hp / "runs").is_dir():
                return (hp / "runs").resolve(), None, True

    proj_name = (req.get("project_name") or "").strip().lower()
    git_origin = _strip_git_suffix(req.get("git_remote_url") or "")

    search_roots = []
    home = Path.home()
    search_roots.append(home)
    try:
        home_real = home.resolve()
        if home_real != home and home_real not in search_roots:
            search_roots.append(home_real)
    except OSError:
        pass

    candidates = []
    visited = set()

    def _walk(curr, depth):
        if depth > 4:
            return
        try:
            real_c = str(curr.resolve())
        except OSError:
            return
        if real_c in visited:
            return
        visited.add(real_c)

        runs_sub = curr / "runs"
        if runs_sub.is_dir() and _has_run_subdirs(runs_sub):
            score = 0
            if git_origin and _read_git_origin(curr) == git_origin:
                score += 100
            if proj_name and curr.name.lower() == proj_name:
                score += 50
            elif proj_name and proj_name in curr.name.lower():
                score += 20
            if score > 0:
                try:
                    newest_mtime = max(
                        (e.stat(follow_symlinks=False).st_mtime for e in os.scandir(runs_sub) if not e.name.startswith(".")),
                        default=0.0,
                    )
                except OSError:
                    newest_mtime = 0.0
                candidates.append((score, newest_mtime, runs_sub.resolve()))
                return

        try:
            with os.scandir(curr) as it:
                for entry in it:
                    if entry.name.startswith(".") or entry.name in SKIP_DIRS or entry.name == "runs":
                        continue
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            _walk(Path(entry.path), depth + 1)
                    except OSError:
                        pass
        except OSError:
            pass

    for sr in search_roots:
        if proj_name:
            direct = sr / req.get("project_name", proj_name) / "runs"
            if direct.is_dir() and _has_run_subdirs(direct):
                return direct.resolve(), None, True
        _walk(sr, 0)

    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return candidates[0][2], None, True

    return None, None, False


def _get_all_cids(data):
    sub = data.get("submission") or {}
    cids = []
    for raw in sub.get("retry_clusters") or []:
        try:
            v = int(raw)
            if v not in cids:
                cids.append(v)
        except Exception:
            pass
    mc = sub.get("cluster_id")
    if mc is not None:
        try:
            v = int(mc)
            if v not in cids:
                cids.append(v)
        except Exception:
            pass
    dep = data.get("dependency") or {}
    if isinstance(dep, dict):
        for k in ("dagman_cluster_id", "job_cluster_id"):
            rc = dep.get(k)
            if rc is not None:
                try:
                    v = int(rc)
                    if v not in cids:
                        cids.append(v)
                except Exception:
                    pass
    return cids


def _load_and_reconcile(run_dir, ht_condor, live_clusters_box):
    rj_path = run_dir / "run.json"
    try:
        data = json.loads(rj_path.read_text(encoding="utf-8"))
    except Exception:
        data = {"run_id": run_dir.name, "status": "UNKNOWN"}

    if ht_condor is not None:
        try:
            if ht_condor.run_needs_live_reconcile(data):
                if not live_clusters_box[0]:
                    live_clusters_box[1] = ht_condor.query_live_condor_clusters()
                    live_clusters_box[0] = True
                data = ht_condor.reconcile_run(run_dir, data, live_clusters_box[1])
            return data
        except Exception:
            pass

    repro_dir = run_dir / "repro"
    if repro_dir.is_dir():
        node_files = sorted(repro_dir.glob("node_*.json"))
        if node_files:
            loaded = []
            for nf in node_files:
                try:
                    loaded.append(json.loads(nf.read_text(encoding="utf-8")))
                except Exception:
                    pass
            if loaded:
                loaded.sort(key=lambda x: int(x.get("process_id", 0)))
                data["executions"] = loaded
                data.pop("execution", None)
    return data


def _extract_label(run_dir, data):
    sub = data.get("submission") or {}
    target_str = str(sub.get("target") or "").strip()
    tag_str = str(sub.get("tag") or "").strip()
    if target_str:
        stem = Path(target_str).stem
        return f"{stem} ({tag_str})" if tag_str else stem
    return run_dir.name


def main():
    try:
        raw_in = sys.stdin.read()
        req = json.loads(raw_in) if raw_in.strip() else {}
    except Exception as exc:
        _emit_manifest({"ok": False, "error": f"Invalid pull request payload: {exc}"})
        return

    for p in ("/usr/bin", "/bin", "/usr/local/bin", "/usr/lib/condor/bin", "/usr/local/condor/bin"):
        if os.path.isdir(p) and p not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{p}:{os.environ.get('PATH', '')}"

    try:
        import hermes_tools.condor as _ht_condor
    except Exception:
        _ht_condor = None
        py_ver = f"python{sys.version_info[0]}.{sys.version_info[1]}"
        for sp in glob.glob(
            os.path.expanduser(f"~/.local/share/uv/tools/hermes-tools/lib/{py_ver}/site-packages")
        ):
            if sp not in sys.path:
                sys.path.insert(0, sp)
        try:
            import hermes_tools.condor as _ht_condor
        except Exception:
            _ht_condor = None

    runs_root, forced_queries, auto_discovered = _resolve_runs_root(req)
    host_label = req.get("host") or socket.gethostname()
    if runs_root is None:
        if req.get("remote_dir"):
            _emit_manifest({
                "ok": False,
                "error": f"Remote directory not found on {socket.gethostname()}: {req['remote_dir']}",
            })
        else:
            proj = req.get("project_name") or "current project"
            _emit_manifest({
                "ok": False,
                "error": (
                    f"Could not auto-discover 'runs/' for '{proj}' on {socket.gethostname()}.\n"
                    f"Pass `-r {host_label}:/path/to/project` or set `remote_dir = \"/path/to/project\"` in .hermes.toml."
                ),
            })
        return

    try:
        with os.scandir(runs_root) as it:
            dirs = [
                Path(e.path)
                for e in it
                if not e.name.startswith(".") and not e.is_symlink() and e.is_dir(follow_symlinks=False)
            ]
    except OSError:
        dirs = []
    dirs.sort(key=lambda p: p.name, reverse=True)

    latest_dir = None
    latest_link = runs_root / "latest"
    try:
        if latest_link.is_symlink():
            t = (runs_root / os.readlink(latest_link)).resolve()
            if t.is_dir():
                latest_dir = t
    except OSError:
        pass
    if latest_dir is None:
        dot_latest = runs_root / ".latest"
        try:
            if dot_latest.is_file():
                t = (runs_root / dot_latest.read_text(encoding="utf-8").strip()).resolve()
                if t.is_dir():
                    latest_dir = t
        except OSError:
            pass
    if latest_dir is None and dirs:
        latest_dir = dirs[0]
    latest_run_id = latest_dir.name if latest_dir is not None else None

    queries = forced_queries if forced_queries is not None else list(req.get("queries") or [])
    if not dirs:
        if queries:
            _emit_manifest({"ok": False, "error": f"No runs found in {runs_root}"})
        else:
            _emit_manifest({
                "ok": True,
                "remote_host": socket.gethostname(),
                "remote_runs_root": str(runs_root),
                "auto_discovered": bool(auto_discovered),
                "latest_run_id": None,
                "selected_runs": [],
                "up_to_date_count": 0,
                "total_matched": 0,
                "total_files": 0,
                "total_bytes": 0,
                "skipped_weights_files": 0,
                "skipped_weights_bytes": 0,
                "skipped_code_files": 0,
                "any_active": False,
            })
        return

    live_clusters_box = [False, None]
    loaded_cache = {}

    def _get_data(d):
        if d.name not in loaded_cache:
            loaded_cache[d.name] = _load_and_reconcile(d, _ht_condor, live_clusters_box)
        return loaded_cache[d.name]

    candidate_dirs = []
    seen_names = set()

    if queries:
        for raw_q in queries:
            q = str(raw_q).strip()
            if not q:
                continue
            q_matches = []
            if q == "latest":
                q_matches = [latest_dir if latest_dir is not None else dirs[0]]
            elif q.isdigit() and 1 <= int(q) <= len(dirs):
                q_matches = [dirs[int(q) - 1]]
            else:
                exact = [d for d in dirs if d.name == q]
                if exact:
                    q_matches = exact
                else:
                    ql = q.lower()
                    for d in dirs:
                        if ql in d.name.lower():
                            q_matches.append(d)
                        else:
                            rj = _get_data(d)
                            lbl = _extract_label(d, rj).lower()
                            cids = [str(c) for c in _get_all_cids(rj)]
                            if ql in lbl or q in cids:
                                q_matches.append(d)
            if not q_matches:
                _emit_manifest({
                    "ok": False,
                    "error": f"No remote run matching '{q}' in {runs_root}",
                })
                return
            for m in q_matches:
                if m.name not in seen_names:
                    seen_names.add(m.name)
                    candidate_dirs.append(m)
    else:
        candidate_dirs = list(dirs)

    status_filter = (req.get("status") or "").strip().upper()
    tag_filter = (req.get("tag") or "").strip().lower()
    pinned_only = bool(req.get("pinned", False))
    limit = int(req.get("limit") or 0)

    matched_runs = []
    for d in candidate_dirs:
        data = _get_data(d)
        if pinned_only and not bool(data.get("pinned", False)):
            continue
        disp_st = str(data.get("_live_badge") or data.get("status", "UNKNOWN"))
        if status_filter and status_filter not in disp_st.upper():
            continue
        lbl = _extract_label(d, data)
        if tag_filter and tag_filter not in d.name.lower() and tag_filter not in lbl.lower():
            continue
        matched_runs.append((d, data, disp_st))
        if not queries and limit > 0 and len(matched_runs) >= limit:
            break

    include_weights = bool(req.get("include_weights", False))
    include_code = bool(req.get("include_code", False))
    force = bool(req.get("force", False))
    include_globs = [str(x).strip() for x in (req.get("include_globs") or []) if str(x).strip()]
    exclude_globs = [str(x).strip() for x in (req.get("exclude_globs") or []) if str(x).strip()]
    local_state = req.get("local_state") or {}

    files_to_send = []
    selected_runs_meta = []
    up_to_date_count = 0
    total_bytes = 0
    skipped_weights_files = 0
    skipped_weights_bytes = 0
    skipped_code_files = 0
    any_active = False

    for d, data, disp_st in matched_runs:
        is_active = disp_st.startswith(("QUEUED", "RUNNING", "HELD", "WAITING"))
        if is_active:
            any_active = True

        loc_run = local_state.get(d.name) or {}
        loc_files = loc_run.get("files") or {}
        loc_status = str(loc_run.get("status") or "")
        loc_has_code = bool(loc_run.get("has_code", False))

        run_items = []
        run_bytes = 0

        for cur_root, subdirs, files in os.walk(d, followlinks=False):
            cur_path = Path(cur_root)

            kept_subdirs = []
            for sd in sorted(subdirs):
                sdp = cur_path / sd
                if sdp.is_symlink():
                    continue
                sd_rel = sdp.relative_to(d).as_posix()
                if sd_rel == "repro/code" and not include_code and not include_globs:
                    try:
                        for _, _, cfiles in os.walk(sdp, followlinks=False):
                            skipped_code_files += len(cfiles)
                    except OSError:
                        pass
                    continue
                if sd_rel == "repro/code" and include_code and not force and not is_active and loc_has_code:
                    continue
                kept_subdirs.append(sd)
            subdirs[:] = kept_subdirs

            for fname in sorted(files):
                if fname == ".run_json.lock" or ".tmp." in fname or ".pull_tmp." in fname:
                    continue
                fp = cur_path / fname
                try:
                    st = fp.lstat()
                except OSError:
                    continue
                if not stat.S_ISREG(st.st_mode):
                    continue

                rel_posix = fp.relative_to(d).as_posix()
                if any(fnmatch.fnmatch(rel_posix, pat) or fnmatch.fnmatch(fname, pat) for pat in exclude_globs):
                    continue

                explicit_inc = any(
                    fnmatch.fnmatch(rel_posix, pat) or fnmatch.fnmatch(fname, pat)
                    for pat in include_globs
                )
                if not explicit_inc:
                    if rel_posix == "repro/code" or rel_posix.startswith("repro/code/"):
                        if not include_code:
                            skipped_code_files += 1
                            continue
                    if fp.suffix.lower() in WEIGHT_EXTS and not include_weights:
                        skipped_weights_files += 1
                        skipped_weights_bytes += st.st_size
                        continue

                mtime_int = int(st.st_mtime)
                if rel_posix == "run.json":
                    rj_bytes = (json.dumps(data, indent=2) + "\n").encode("utf-8")
                    if (
                        not force
                        and not is_active
                        and rel_posix in loc_files
                        and loc_files[rel_posix] == [len(rj_bytes), mtime_int]
                        and loc_status == disp_st
                    ):
                        continue
                    run_items.append(("bytes", d.name, rel_posix, rj_bytes, mtime_int))
                    run_bytes += len(rj_bytes)
                else:
                    if (
                        not force
                        and rel_posix in loc_files
                        and loc_files[rel_posix] == [st.st_size, mtime_int]
                    ):
                        continue
                    run_items.append(("file", d.name, rel_posix, str(fp), st.st_size, mtime_int))
                    run_bytes += st.st_size

        if not run_items:
            up_to_date_count += 1
            continue

        files_to_send.extend(run_items)
        total_bytes += run_bytes
        sub = data.get("submission") or {}
        req_res = data.get("resources_requested") or {}
        cids = _get_all_cids(data)
        selected_runs_meta.append({
            "run_id": d.name,
            "status": disp_st,
            "arch": str(req_res.get("architecture") or "—"),
            "target": str(sub.get("target") or ""),
            "tag": str(sub.get("tag") or ""),
            "submitted_at": str(sub.get("submitted_at") or ""),
            "cluster_id": cids[-1] if cids else None,
            "files_count": len(run_items),
            "bytes_count": run_bytes,
            "is_update": bool(d.name in local_state),
            "is_active": is_active,
        })

    manifest = {
        "ok": True,
        "remote_host": socket.gethostname(),
        "remote_runs_root": str(runs_root),
        "auto_discovered": bool(auto_discovered),
        "latest_run_id": latest_run_id,
        "selected_runs": selected_runs_meta,
        "up_to_date_count": up_to_date_count,
        "total_matched": len(matched_runs),
        "total_files": len(files_to_send),
        "total_bytes": total_bytes,
        "skipped_weights_files": skipped_weights_files,
        "skipped_weights_bytes": skipped_weights_bytes,
        "skipped_code_files": skipped_code_files,
        "any_active": any_active,
    }
    _emit_manifest(manifest)

    if not req.get("dry_run") and files_to_send:
        with tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz", compresslevel=4) as tf:
            for item in files_to_send:
                kind = item[0]
                if kind == "bytes":
                    _, run_name, rel_posix, raw_bytes, mtime_int = item
                    ti = tarfile.TarInfo(name=f"{run_name}/{rel_posix}")
                    ti.size = len(raw_bytes)
                    ti.mtime = mtime_int
                    ti.mode = 0o644
                    tf.addfile(ti, io.BytesIO(raw_bytes))
                else:
                    _, run_name, rel_posix, fpath_str, fsize, mtime_int = item
                    try:
                        if fsize <= 8 * 1024 * 1024:
                            with open(fpath_str, "rb") as fh:
                                data_bytes = fh.read()
                            ti = tarfile.TarInfo(name=f"{run_name}/{rel_posix}")
                            ti.size = len(data_bytes)
                            ti.mtime = mtime_int
                            ti.mode = 0o644
                            tf.addfile(ti, io.BytesIO(data_bytes))
                        else:
                            with open(fpath_str, "rb") as fh:
                                ti = tarfile.TarInfo(name=f"{run_name}/{rel_posix}")
                                ti.size = fsize
                                ti.mtime = mtime_int
                                ti.mode = 0o644
                                tf.addfile(ti, _SafeReader(fh, fsize))
                    except OSError:
                        pass
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
'''


def _parse_remote_spec(spec: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parses a `--remote` / `remote` specification into `(host, remote_dir)`.
    Supports `[user@]host:/path/to/project`, `/remote/path`, `~/remote/path`, or `host`.
    """
    s = (spec or "").strip()
    if not s:
        return None, None
    if s.startswith(("/", "~", ".")):
        return None, s
    if ":" in s and not re.match(r"^[A-Za-z]:[\\/]", s):
        host_part, path_part = s.split(":", 1)
        return (host_part.strip() or None), (path_part.strip() or None)
    return s, None


def _read_local_git_origin(cwd: Path) -> Tuple[str, str]:
    curr = cwd.resolve()
    git_root: Optional[Path] = None
    for _ in range(5):
        if (curr / ".git").exists():
            git_root = curr
            break
        if curr.parent == curr:
            break
        curr = curr.parent

    proj_dir = git_root if git_root is not None else cwd.resolve()
    proj_name = proj_dir.name
    git_origin = ""
    if git_root is not None:
        cfg_file = git_root / ".git" / "config"
        if cfg_file.is_file():
            try:
                for line in cfg_file.read_text(encoding="utf-8", errors="replace").splitlines():
                    s = line.strip()
                    if s.startswith("url =") or s.startswith("url="):
                        git_origin = s.split("=", 1)[1].strip()
                        break
            except OSError:
                pass
    return proj_name, git_origin


def _resolve_pull_target(
    args: argparse.Namespace, cwd: Path, runs_root: Path
) -> Tuple[str, Optional[str], List[str], str, str]:
    """Resolves `(host, remote_dir, hint_dirs, project_name, git_remote_url)`."""
    proj_cfg = load_project_config(cwd)

    cli_remote_host, cli_remote_dir = _parse_remote_spec(str(getattr(args, "remote", "") or ""))
    cfg_remote_host, cfg_remote_dir = _parse_remote_spec(str(proj_cfg.get("remote", "") or ""))

    explicit_host = (
        getattr(args, "host", None)
        or cli_remote_host
        or proj_cfg.get("remote_host")
        or proj_cfg.get("host")
        or cfg_remote_host
    )
    host = str(explicit_host).strip() if explicit_host else "hermes"

    explicit_dir = (
        getattr(args, "remote_dir", None)
        or cli_remote_dir
        or proj_cfg.get("remote_dir")
        or proj_cfg.get("remote_path")
        or cfg_remote_dir
    )
    remote_dir = str(explicit_dir).strip() if explicit_dir else None

    hint_dirs: List[str] = []
    for d in list_run_dirs(runs_root)[:10]:
        rj = load_run_json(d)
        wd = str((rj.get("submission") or {}).get("working_directory") or "").strip()
        if wd and wd not in hint_dirs:
            hint_dirs.append(f"{wd.rstrip('/')}/runs")
            hint_dirs.append(wd)

    proj_name, git_origin = _read_local_git_origin(cwd)
    return host, remote_dir, hint_dirs, proj_name, git_origin


def _collect_local_pull_state(runs_root: Path, force: bool = False) -> Dict[str, Any]:
    if force or not runs_root.is_dir():
        return {}
    state: Dict[str, Any] = {}
    for d in list_run_dirs(runs_root):
        rj = load_run_json(d)
        status = str(rj.get("_live_badge") or rj.get("status", "UNKNOWN"))
        is_active = status.startswith(("QUEUED", "RUNNING", "HELD", "WAITING"))
        files_map: Dict[str, List[int]] = {}
        code_dir = d / "repro" / "code"
        has_code = False
        try:
            if code_dir.is_dir():
                with os.scandir(code_dir) as it:
                    has_code = any(True for _ in it)
        except OSError:
            pass

        for cur_root, subdirs, files in os.walk(d, followlinks=False):
            cur_path = Path(cur_root)
            rel_dir = cur_path.relative_to(d).as_posix()
            if rel_dir == "repro" and has_code and not is_active:
                subdirs[:] = [sd for sd in subdirs if sd != "code"]
            for fname in files:
                if fname == ".run_json.lock" or ".tmp." in fname or ".pull_tmp." in fname:
                    continue
                fp = cur_path / fname
                try:
                    st = fp.lstat()
                    rel_posix = fp.relative_to(d).as_posix()
                    files_map[rel_posix] = [st.st_size, int(st.st_mtime)]
                except OSError:
                    pass
        state[d.name] = {
            "status": status,
            "has_code": has_code,
            "files": files_map,
        }
    return state


def _spawn_pull_process(host: str, req: Dict[str, Any]) -> subprocess.Popen[bytes]:
    script_b64 = base64.b64encode(_REMOTE_PULL_SCRIPT.encode("utf-8")).decode("ascii")
    py_one_liner = f"import base64;exec(base64.b64decode('{script_b64}'))"

    if host.lower() in ("local", "localhost-local"):
        cmd = [sys.executable, "-u", "-c", py_one_liner]
    else:
        ssh_bin = shutil.which("ssh")
        if not ssh_bin:
            console.print("[bold red]✘ Error:[/bold red] 'ssh' client not found in PATH.")
            sys.exit(1)
        remote_sh = (
            'export PATH="/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin:$PATH"; '
            'PY_BIN="$HOME/.local/share/uv/tools/hermes-tools/bin/python"; '
            'if [ ! -x "$PY_BIN" ]; then PY_BIN="python3"; fi; '
            f'"$PY_BIN" -u -c "{py_one_liner}"'
        )
        cmd = [ssh_bin, host, remote_sh]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None
    try:
        proc.stdin.write(json.dumps(req).encode("utf-8"))
        proc.stdin.close()
    except OSError:
        pass
    return proc


def _extract_pull_tar_stream(
    stream: Any,
    runs_root: Path,
    total_bytes: int,
    show_progress: bool = True,
) -> Tuple[int, int]:
    runs_root.mkdir(parents=True, exist_ok=True)
    runs_root_resolved = runs_root.resolve()
    extracted_files = 0
    extracted_bytes = 0

    def _do_extract(progress: Optional[Any], task_id: Optional[Any]) -> None:
        nonlocal extracted_files, extracted_bytes
        with tarfile.open(fileobj=stream, mode="r|gz") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                rel_parts = [
                    p
                    for p in member.name.replace("\\", "/").split("/")
                    if p and p != "."
                ]
                if not rel_parts or ".." in rel_parts:
                    continue
                dest_path = runs_root.joinpath(*rel_parts)
                try:
                    if not dest_path.resolve().is_relative_to(runs_root_resolved):
                        continue
                except OSError:
                    continue

                src_f = tf.extractfile(member)
                if src_f is None:
                    continue

                dest_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_dest = dest_path.parent / f".{dest_path.name}.pull_tmp.{os.getpid()}"
                try:
                    with open(tmp_dest, "wb") as out_f:
                        while True:
                            chunk = src_f.read(65536)
                            if not chunk:
                                break
                            out_f.write(chunk)
                            extracted_bytes += len(chunk)
                            if progress is not None and task_id is not None:
                                progress.update(task_id, advance=len(chunk))
                    os.replace(tmp_dest, dest_path)
                    if member.mtime:
                        mtime_int = int(member.mtime)
                        try:
                            os.utime(dest_path, (mtime_int, mtime_int))
                        except OSError:
                            pass
                    extracted_files += 1
                finally:
                    if tmp_dest.exists():
                        try:
                            tmp_dest.unlink()
                        except OSError:
                            pass

    if show_progress and console.is_terminal and total_bytes > 0:
        from rich.progress import (
            BarColumn,
            DownloadColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TransferSpeedColumn,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            console=console,
            transient=True,
        ) as progress:
            tid = progress.add_task("Downloading run artifacts...", total=total_bytes)
            _do_extract(progress, tid)
    else:
        _do_extract(None, None)

    return extracted_files, extracted_bytes


def _execute_single_pull(
    args: argparse.Namespace, runs_root: Path, quiet_if_up_to_date: bool = False
) -> Tuple[int, bool]:
    cwd = Path.cwd()
    host, remote_dir, hint_dirs, proj_name, git_origin = _resolve_pull_target(
        args, cwd, runs_root
    )
    full_flag = bool(getattr(args, "full", False))
    include_weights = bool(getattr(args, "weights", False) or full_flag)
    include_code = bool(getattr(args, "code", False) or full_flag)
    force = bool(getattr(args, "force", False))
    dry_run = bool(getattr(args, "dry_run", False))
    all_runs = bool(getattr(args, "all", False))
    raw_limit = int(getattr(args, "limit", 25) or 0)
    limit = 0 if all_runs else raw_limit

    raw_queries = getattr(args, "run", None)
    if isinstance(raw_queries, str):
        queries = [raw_queries] if raw_queries.strip() else []
    elif isinstance(raw_queries, list):
        queries = [str(q).strip() for q in raw_queries if str(q).strip()]
    else:
        queries = []

    local_state = _collect_local_pull_state(runs_root, force=force)
    req: Dict[str, Any] = {
        "host": host,
        "remote_dir": remote_dir,
        "hint_dirs": hint_dirs,
        "project_name": proj_name,
        "git_remote_url": git_origin,
        "queries": queries,
        "limit": limit,
        "status": getattr(args, "status", None),
        "tag": getattr(args, "tag", None),
        "pinned": bool(getattr(args, "pinned", False)),
        "include_weights": include_weights,
        "include_code": include_code,
        "include_globs": list(getattr(args, "include", None) or []),
        "exclude_globs": list(getattr(args, "exclude", None) or []),
        "force": force,
        "dry_run": dry_run,
        "local_state": local_state,
    }

    proc = _spawn_pull_process(host, req)
    assert proc.stdout is not None and proc.stderr is not None

    manifest: Optional[Dict[str, Any]] = None
    pre_lines: List[str] = []
    try:
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            if line.startswith(_PULL_MANIFEST_PREFIX):
                raw_json = line[len(_PULL_MANIFEST_PREFIX) :].decode("utf-8", errors="replace")
                manifest = json.loads(raw_json)
                break
            pre_lines.append(line.decode("utf-8", errors="replace").rstrip())

        if manifest is None:
            stderr_txt = proc.stderr.read().decode("utf-8", errors="replace").strip()
            proc.wait()
            err_detail = stderr_txt or "\n".join(pre_lines).strip() or "Connection closed unexpectedly."
            console.print(f"[bold red]✘ Failed to pull from '{host}':[/bold red]\n{err_detail}")
            sys.exit(1)

        if not manifest.get("ok"):
            proc.wait()
            console.print(f"[bold red]✘ {manifest.get('error', 'Remote pull error.')}[/bold red]")
            sys.exit(1)

        remote_runs_root = str(manifest.get("remote_runs_root") or "")
        auto_discovered = bool(manifest.get("auto_discovered", False))
        selected_runs: List[Dict[str, Any]] = list(manifest.get("selected_runs") or [])
        up_to_date_count = int(manifest.get("up_to_date_count") or 0)
        total_matched = int(manifest.get("total_matched") or 0)
        total_files = int(manifest.get("total_files") or 0)
        total_bytes = int(manifest.get("total_bytes") or 0)
        skipped_weights_files = int(manifest.get("skipped_weights_files") or 0)
        skipped_weights_bytes = int(manifest.get("skipped_weights_bytes") or 0)
        any_active = bool(manifest.get("any_active", False))
        latest_run_id = manifest.get("latest_run_id")

        if auto_discovered and not quiet_if_up_to_date:
            console.print(
                f"[dim cyan]ℹ Auto-discovered remote runs:[/dim cyan] [bold]{host}:{remote_runs_root}[/bold]"
            )

        if total_matched == 0:
            proc.wait()
            if not quiet_if_up_to_date:
                console.print(f"[yellow]No matching runs found on {host}:{remote_runs_root}.[/yellow]")
            return 0, False

        if not selected_runs:
            proc.wait()
            if not quiet_if_up_to_date:
                console.print(
                    f"[bold green]✔ All {up_to_date_count} matching run(s) are already up-to-date locally in {runs_root}.[/bold green]"
                )
                if skipped_weights_files > 0:
                    console.print(
                        f"[dim]  ℹ Skipped {skipped_weights_files} checkpoint/weight file(s) "
                        f"({_format_bytes(skipped_weights_bytes)}) — pass [bold]--weights[/bold] to include them.[/dim]"
                    )
            return 0, any_active

        tbl_title = (
            f"🔍 Dry Run — Runs to Pull from {host}"
            if dry_run
            else f"📥 Pulling Runs from {host}"
        )
        tbl = Table(
            title=tbl_title,
            title_style="bold cyan",
            box=box.ROUNDED,
            header_style="bold cyan",
            padding=(0, 1),
        )
        tbl.add_column("Run", no_wrap=True)
        tbl.add_column("Submitted", style="dim", no_wrap=True)
        tbl.add_column("Cluster", style="cyan", justify="right", no_wrap=True)
        tbl.add_column("Arch", style="magenta", no_wrap=True)
        tbl.add_column("Status", no_wrap=True)
        tbl.add_column("Sync", justify="center", no_wrap=True)
        tbl.add_column("Files", justify="right", no_wrap=True)
        tbl.add_column("Size", justify="right", style="cyan", no_wrap=True)

        for r in selected_runs[:30]:
            fake_data = {
                "submission": {
                    "target": r.get("target"),
                    "tag": r.get("tag"),
                    "submitted_at": r.get("submitted_at"),
                },
                "resources_requested": {"architecture": r.get("arch")},
            }
            _, rich_label, short_date = extract_run_display_parts(
                Path(str(r["run_id"])), fake_data
            )
            star = (
                "[bold yellow]★[/bold yellow] "
                if latest_run_id and r.get("run_id") == latest_run_id
                else ""
            )
            sync_badge = (
                "[cyan]update[/cyan]" if r.get("is_update") else "[bold green]new[/bold green]"
            )
            cid_val = r.get("cluster_id")
            tbl.add_row(
                f"{star}{rich_label}",
                short_date,
                str(cid_val) if cid_val else "—",
                str(r.get("arch") or "—"),
                style_status(str(r.get("status") or "UNKNOWN")),
                sync_badge,
                str(r.get("files_count", 0)),
                _format_bytes(int(r.get("bytes_count", 0))),
            )

        if len(selected_runs) > 30:
            tbl.add_row(
                f"[dim]... (+{len(selected_runs) - 30} more runs)[/dim]",
                "—",
                "—",
                "—",
                "—",
                "—",
                "—",
                "—",
            )

        console.print(tbl)

        if dry_run:
            proc.wait()
            console.print(
                f"[bold cyan]Dry run:[/bold cyan] Would pull [bold]{len(selected_runs)}[/bold] run(s) "
                f"({total_files} files, {_format_bytes(total_bytes)}) into [bold]{runs_root}[/bold]"
            )
            if skipped_weights_files > 0:
                console.print(
                    f"[dim]  ℹ {skipped_weights_files} checkpoint/weight file(s) "
                    f"({_format_bytes(skipped_weights_bytes)}) excluded by default (use [bold]--weights[/bold] to include).[/dim]"
                )
            return len(selected_runs), any_active

        extracted_files, extracted_bytes = _extract_pull_tar_stream(
            proc.stdout, runs_root, total_bytes
        )
        proc.wait()

        if latest_run_id and (runs_root / str(latest_run_id)).is_dir():
            update_latest_symlink(runs_root, runs_root / str(latest_run_id))
        else:
            update_latest_symlink(runs_root)

        up_note = (
            f" [dim]({up_to_date_count} already up-to-date)[/dim]"
            if up_to_date_count > 0
            else ""
        )
        console.print(
            f"[bold green]✔ Pulled {len(selected_runs)} run(s)[/bold green] "
            f"({extracted_files} files, {_format_bytes(extracted_bytes)}) into [bold]{runs_root}[/bold]{up_note}"
        )
        if skipped_weights_files > 0:
            console.print(
                f"[dim]  ℹ Skipped {skipped_weights_files} checkpoint/weight file(s) "
                f"({_format_bytes(skipped_weights_bytes)}) — pass [bold]--weights[/bold] to include them.[/dim]"
            )
        return len(selected_runs), any_active
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def cmd_pull(args: argparse.Namespace, runs_root: Path) -> None:
    watch_mode = bool(getattr(args, "watch", False))
    interval = max(2, int(getattr(args, "interval", 15) or 15))

    _, any_active = _execute_single_pull(args, runs_root, quiet_if_up_to_date=False)
    if not watch_mode or getattr(args, "dry_run", False):
        return

    if not any_active:
        console.print("[dim]All matched remote runs have reached a terminal status.[/dim]")
        return

    console.print(
        f"[dim cyan]Watching remote runs every {interval}s for new outputs/telemetry (Ctrl+C to stop)...[/dim cyan]"
    )
    try:
        while any_active:
            time.sleep(interval)
            _, any_active = _execute_single_pull(args, runs_root, quiet_if_up_to_date=True)
            if not any_active:
                console.print(
                    "[bold green]✔ All watched remote runs have finished and synced locally![/bold green]"
                )
                break
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped watching remote runs.[/dim]")
