"""
runs_cli.py — HTCondor Run Manager, Telemetry Inspector & Recovery CLI entrypoint (`runs`).
"""

from __future__ import annotations

import argparse
import datetime
import difflib
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import questionary
from rich import box
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from hermes_tools.condor import (
    generate_retry_sub_file,
    get_all_cluster_ids,
    query_live_condor_clusters,
    reconcile_run,
)
from hermes_tools.core import (
    PROMPT_STYLE,
    console,
    dir_size_mb,
    format_duration,
    get_runs_root,
    list_run_dirs,
    load_run_json,
    locked_mutate_run_json,
    resolve_latest_symlink,
    style_status,
    summarize_metrics,
    update_latest_symlink,
    write_text_lf,
)
from hermes_tools.snapshot import create_code_snapshot, freeze_python_env, get_git_info


def resolve_run_dir(
    runs_root: Path, query: Optional[str], prompt_title: str = "Select a run:"
) -> Path:
    dirs = list_run_dirs(runs_root)
    if not dirs:
        console.print(f"[yellow]No runs found in {runs_root}[/yellow]")
        sys.exit(1)

    if query:
        q = query.strip()
        if q == "latest":
            latest_target = resolve_latest_symlink(runs_root)
            return latest_target if latest_target is not None else dirs[0]

        if q.isdigit():
            idx = int(q)
            if 1 <= idx <= len(dirs):
                return dirs[idx - 1]

        exact = [d for d in dirs if d.name == q]
        if len(exact) == 1:
            return exact[0]

        matches = []
        for d in dirs:
            if q.lower() in d.name.lower():
                matches.append(d)
            else:
                rj = load_run_json(d)
                all_cids = [str(c) for c in get_all_cluster_ids(rj)]
                if q in all_cids:
                    matches.append(d)

        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            dirs = matches
        else:
            console.print(f"[bold red]✘ No run matching '{query}'.[/bold red]")
            sys.exit(1)

    if not sys.stdin.isatty():
        return dirs[0]

    choices = []
    for i, d in enumerate(dirs[:35], start=1):
        rj = load_run_json(d)
        st = rj.get("status", "UNKNOWN")
        arch = rj.get("resources_requested", {}).get("architecture", "?")
        cid = rj.get("submission", {}).get("cluster_id") or "—"
        choices.append(
            questionary.Choice(
                title=f"[{i:02d}] {d.name:<42} │ {st:<22} │ {arch:<8} │ Cluster: {cid}",
                value=d,
            )
        )
    selected = questionary.select(prompt_title, choices=choices, style=PROMPT_STYLE).ask()
    if not selected:
        sys.exit(0)
    return selected


# ---------------------------------------------------------------------------
# Subcommand: `runs ls` & `runs watch`
# ---------------------------------------------------------------------------
def _build_ls_table(args: argparse.Namespace, runs_root: Path) -> Table:
    dirs = list_run_dirs(runs_root)
    live_clusters = query_live_condor_clusters()
    latest_target = resolve_latest_symlink(runs_root)

    table = Table(
        title=f"[bold cyan]📊 Hermes 2 Runs ({runs_root})[/bold cyan]",
        box=box.ROUNDED,
        header_style="bold cyan",
    )
    table.add_column("#", style="dim", justify="right")
    table.add_column("Run ID", style="bold white")
    table.add_column("Cluster", style="cyan", justify="right")
    table.add_column("Arch", style="magenta")
    table.add_column("Status")
    table.add_column("Tasks", justify="center")
    table.add_column("Host(s)", style="dim")
    table.add_column("Wall Time", justify="right")
    table.add_column("Peak RAM", justify="right")

    shown = 0
    limit = getattr(args, "limit", 25)
    status_filter = getattr(args, "status", None)
    tag_filter = getattr(args, "tag", None)

    for idx, d in enumerate(dirs, start=1):
        data = reconcile_run(d, load_run_json(d), live_clusters)
        display_status = str(data.get("_live_badge") or data.get("status", "UNKNOWN"))

        if status_filter and status_filter.upper() not in display_status.upper():
            continue
        if tag_filter and tag_filter.lower() not in d.name.lower():
            continue

        req = data.get("resources_requested", {})
        arch = str(req.get("architecture", "—"))
        sweep_cnt = int(req.get("sweep_count", 1) or 1)
        cids = get_all_cluster_ids(data)
        cid_str = (
            str(cids[-1]) + (f" (+{len(cids) - 1})" if len(cids) > 1 else "") if cids else "—"
        )

        if sweep_cnt > 1:
            execs = data.get("executions", [])
            ok = sum(1 for x in execs if x.get("exit_code", -1) == 0)
            fail = sum(1 for x in execs if x.get("exit_code", 0) != 0)
            tasks_str = (
                f"[green]{ok}[/green]/[red]{fail}[/red]/{sweep_cnt}"
                if fail
                else f"{ok}/{sweep_cnt}"
            )
        else:
            tasks_str = "1"

        host_str, dur_str, ram_str = summarize_metrics(data)
        name_label = f"{d.name} [bold yellow]★[/bold yellow]" if d == latest_target else d.name

        table.add_row(
            str(idx),
            name_label,
            cid_str,
            arch,
            style_status(display_status),
            tasks_str,
            host_str,
            dur_str,
            ram_str,
        )
        shown += 1
        if limit and shown >= limit:
            break

    return table


def cmd_ls(args: argparse.Namespace, runs_root: Path) -> None:
    dirs = list_run_dirs(runs_root)
    if not dirs:
        console.print(f"[dim]No runs found in {runs_root}[/dim]")
        return

    if getattr(args, "watch", False):
        interval = max(1, int(getattr(args, "interval", 2) or 2))
        try:
            with Live(
                _build_ls_table(args, runs_root),
                console=console,
                refresh_per_second=1,
            ) as live:
                while True:
                    time.sleep(interval)
                    live.update(_build_ls_table(args, runs_root))
        except KeyboardInterrupt:
            return
    else:
        console.print(_build_ls_table(args, runs_root))


# ---------------------------------------------------------------------------
# Subcommand: `runs inspect`
# ---------------------------------------------------------------------------
def cmd_inspect(args: argparse.Namespace, runs_root: Path) -> None:
    run_dir = resolve_run_dir(runs_root, args.run)
    live_clusters = query_live_condor_clusters()
    data = reconcile_run(run_dir, load_run_json(run_dir), live_clusters)

    sub = data.get("submission", {})
    req = data.get("resources_requested", {})
    git = data.get("git", {})
    display_status = str(data.get("_live_badge") or data.get("status", "UNKNOWN"))
    cids = get_all_cluster_ids(data)

    meta = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
    meta.add_column("Key", style="bold cyan", width=20)
    meta.add_column("Val", style="white")

    meta.add_row("Run ID", f"[bold]{run_dir.name}[/bold]")
    meta.add_row("Status", style_status(display_status))
    meta.add_row("HTCondor Cluster(s)", ", ".join(str(c) for c in cids) if cids else "—")
    meta.add_row(
        "Submitted At",
        f"{sub.get('submitted_at', '—')} by {sub.get('user', '—')}@{sub.get('submit_host', '—')}",
    )
    meta.add_row(
        "Target Script",
        f"{sub.get('target', '—')} (Snapshot: [cyan]{sub.get('snapshot_executable', '—')}[/cyan])",
    )
    meta.add_row(
        "Architecture",
        f"[bold magenta]{req.get('architecture', '—')}[/bold magenta] (CPUs: {req.get('cpus') or 'Auto'}, RAM: {req.get('memory_mb') or 'Auto'} MB, GPUs: {req.get('gpus') or 0})",
    )
    if git.get("is_git"):
        dirty_str = (
            f"[yellow]dirty ({git.get('patch_file')})[/yellow]"
            if git.get("dirty")
            else "[green]clean[/green]"
        )
        sync_note = f" [dim](synced {git.get('synced_at')})[/dim]" if git.get("synced_at") else ""
        meta.add_row(
            "Git Provenance",
            f"{git.get('branch')} @ {str(git.get('commit'))[:8]} ({dirty_str}){sync_note}",
        )
    meta.add_row("Disk Footprint", f"{dir_size_mb(run_dir):.2f} MB in [dim]{run_dir}[/dim]")
    meta.add_row("Repro CLI", f"[bold green]{sub.get('equivalent_cli', '—')}[/bold green]")

    console.print(
        Panel(
            meta,
            title="[bold cyan]🔍 Run Provenance & Telemetry[/bold cyan]",
            border_style="cyan",
            expand=False,
        )
    )

    sweep_cnt = int(req.get("sweep_count", 1) or 1)
    records: List[Dict[str, Any]] = []
    if sweep_cnt == 1 and isinstance(data.get("execution"), dict):
        records = [data["execution"]]
    elif sweep_cnt > 1:
        records = list(data.get("executions") or [])

    if records:
        et = Table(
            title="Process Execution Breakdown",
            box=box.SIMPLE_HEAVY,
            header_style="bold cyan",
        )
        et.add_column("Proc", justify="right")
        et.add_column("Exit", justify="center")
        et.add_column("Host", style="cyan")
        et.add_column("GPU / Hardware", style="dim")
        et.add_column("Queue Wait", justify="right")
        et.add_column("Runtime", justify="right")
        et.add_column("Peak RAM", justify="right")
        et.add_column("CPU %", justify="right")
        et.add_column("Arguments")

        for r in records:
            pid = r.get("process_id", 0)
            ec = r.get("exit_code", 0)
            sig = r.get("signal")
            ec_str = (
                "[bold green]0[/bold green]"
                if ec == 0
                else f"[bold red]{ec}[/bold red]{f' ({sig})' if sig else ''}"
            )
            hw = (
                r.get("gpu_model")
                if r.get("gpu_model") not in (None, "None", "")
                else r.get("cpu_model", "—")
            )
            ram_mb = float(r.get("peak_ram_mb", 0.0) or 0.0)
            et.add_row(
                str(pid),
                ec_str,
                str(r.get("hostname", "—")).split(".")[0],
                str(hw)[:32],
                format_duration(r.get("queue_wait_seconds")),
                format_duration(r.get("runtime_seconds")),
                f"{ram_mb:.1f} MB",
                str(r.get("cpu_utilization", "—")),
                str(r.get("arguments", "")),
            )
        console.print(et)

    err_files = sorted(run_dir.glob("*.err"))
    for ef in err_files:
        try:
            content = ef.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if content:
            tail_lines = "\n".join(content.splitlines()[-25:])
            console.print(
                Panel(
                    Syntax(tail_lines, "pytb", theme="monokai", word_wrap=True),
                    title=f"[bold red]⚠ Stderr Tail: {ef.name}[/bold red]",
                    border_style="red",
                )
            )


# ---------------------------------------------------------------------------
# Subcommand: `runs logs`
# ---------------------------------------------------------------------------
def _follow_file_python(log_file: Path, initial_lines: int) -> None:
    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            for line in lines[-initial_lines:]:
                sys.stdout.write(line)
            sys.stdout.flush()
            while True:
                line = f.readline()
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                else:
                    time.sleep(0.25)
    except KeyboardInterrupt:
        return


def cmd_logs(args: argparse.Namespace, runs_root: Path) -> None:
    run_dir = resolve_run_dir(runs_root, args.run)
    data = load_run_json(run_dir)
    target_stem = Path(data.get("submission", {}).get("target", "job")).stem
    sweep_cnt = int(data.get("resources_requested", {}).get("sweep_count", 1) or 1)

    if args.condor:
        log_file = run_dir / "repro" / "condor.log"
    else:
        ext = ".err" if args.err else ".out"
        if sweep_cnt > 1:
            proc = args.proc
            if proc is None:
                if args.err:
                    failed_execs = [
                        int(x.get("process_id", 0))
                        for x in (data.get("executions") or [])
                        if int(x.get("exit_code", 0)) != 0
                    ]
                    if failed_execs:
                        proc = failed_execs[0]
                        console.print(
                            f"[dim yellow]ℹ Auto-selected failing process #{proc} (use `-p <id>` to override)[/dim yellow]"
                        )
                    else:
                        for idx in range(sweep_cnt):
                            cand = run_dir / f"{target_stem}_{idx}.err"
                            if cand.is_file() and cand.stat().st_size > 0:
                                proc = idx
                                break
                if proc is None:
                    proc = 0
            log_file = run_dir / f"{target_stem}_{proc}{ext}"
        else:
            log_file = run_dir / f"{target_stem}{ext}"

    if not log_file.is_file():
        console.print(f"[bold red]✘ Log file not found:[/bold red] {log_file}")
        sys.exit(1)

    if args.follow:
        console.print(f"[dim cyan]Following {log_file} (Ctrl+C to exit)...[/dim cyan]")
        if os.name == "posix" and shutil.which("tail"):
            os.execvp("tail", ["tail", "-n", str(args.lines), "-f", str(log_file)])
        else:
            _follow_file_python(log_file, args.lines)
    else:
        lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        snippet = "\n".join(lines[-args.lines :])
        rel_title = (
            log_file.relative_to(runs_root.parent)
            if log_file.is_relative_to(runs_root.parent)
            else log_file
        )
        console.print(
            Panel(
                snippet,
                title=f"[bold cyan]{rel_title}[/bold cyan]",
                border_style="cyan",
            )
        )


# ---------------------------------------------------------------------------
# Subcommand: `runs stop`
# ---------------------------------------------------------------------------
def cmd_stop(args: argparse.Namespace, runs_root: Path) -> None:
    run_dir = resolve_run_dir(
        runs_root, args.run, prompt_title="Select run to stop (condor_rm):"
    )
    repro_dir = run_dir / "repro"
    rj_path = run_dir / "run.json"
    data = load_run_json(run_dir)
    cids = get_all_cluster_ids(data)

    if not cids:
        console.print(f"[yellow]Run {run_dir.name} has no HTCondor cluster_id.[/yellow]")
        return

    live_clusters = query_live_condor_clusters()
    target_cids = [c for c in cids if live_clusters is None or c in live_clusters] or cids

    if not args.yes:
        if not questionary.confirm(
            f"Kill HTCondor cluster(s) {target_cids} ({run_dir.name})?",
            default=True,
            style=PROMPT_STYLE,
        ).ask():
            return

    for cid in target_cids:
        res = subprocess.run(
            ["condor_rm", str(cid)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if res.returncode == 0:
            console.print(f"[bold green]✔ {res.stdout.strip()}[/bold green]")
        else:
            console.print(
                f"[yellow]⚠ condor_rm {cid}: {(res.stderr or res.stdout).strip()}[/yellow]"
            )

    locked_mutate_run_json(rj_path, repro_dir, lambda d: d.update({"status": "ABORTED"}))
    console.print(f"[cyan]Updated {run_dir.name}/run.json status -> ABORTED[/cyan]")


# ---------------------------------------------------------------------------
# Subcommand: `runs sync` & Code Snapshot Refreshing
# ---------------------------------------------------------------------------
def refresh_code_snapshot(run_dir: Path, data: Dict[str, Any]) -> None:
    """
    Refreshes `repro/code/`, updates `repro/uncommitted.patch`, re-freezes `repro/env.txt`,
    and updates `run.json["git"]` so provenance stays 100% accurate.
    """
    cwd = Path(data.get("submission", {}).get("working_directory", Path.cwd()))
    target_rel = data.get("submission", {}).get("target", "")
    target_path = (cwd / target_rel).resolve()
    repro_dir = run_dir / "repro"
    rj_path = run_dir / "run.json"

    if not target_path.is_file():
        console.print(f"[bold red]✘ Source file not found at {target_path}[/bold red]")
        sys.exit(1)

    git_info = get_git_info(cwd)
    create_code_snapshot(target_path, cwd, repro_dir, git_info)

    patch_file = repro_dir / "uncommitted.patch"
    patch_rel_path: Optional[str] = None
    if git_info.get("is_git") and git_info.get("diff"):
        write_text_lf(patch_file, str(git_info["diff"]))
        patch_rel_path = "repro/uncommitted.patch"
    elif patch_file.is_file():
        patch_file.unlink()

    venv_str = data.get("submission", {}).get("venv")
    if venv_str:
        freeze_python_env(Path(venv_str), repro_dir)

    sync_iso = datetime.datetime.now().isoformat(timespec="seconds")

    def _update_git_record(d: Dict[str, Any]) -> None:
        g = d.setdefault("git", {})
        if g.get("commit") and "initial_commit" not in g:
            g["initial_commit"] = g.get("commit")
        g["is_git"] = git_info["is_git"]
        g["branch"] = git_info["branch"]
        g["commit"] = git_info["commit"]
        g["dirty"] = git_info["dirty"]
        g["patch_file"] = patch_rel_path
        g["synced_at"] = sync_iso

    locked_mutate_run_json(rj_path, repro_dir, _update_git_record)
    console.print(
        f"[bold green]✔ Synchronized working tree & git provenance into {run_dir.name}/repro/[/bold green]"
    )


def cmd_sync(args: argparse.Namespace, runs_root: Path) -> None:
    run_dir = resolve_run_dir(
        runs_root, args.run, prompt_title="Select run to sync code snapshot into:"
    )
    data = load_run_json(run_dir)
    refresh_code_snapshot(run_dir, data)


# ---------------------------------------------------------------------------
# Subcommand: `runs retry`
# ---------------------------------------------------------------------------
def get_failed_or_missing_procs(data: Dict[str, Any]) -> List[Tuple[int, str]]:
    req = data.get("resources_requested", {})
    sweep_args: List[str] = list(req.get("sweep_arguments") or [""])
    sweep_cnt = len(sweep_args)

    if sweep_cnt == 1:
        exec_rec = data.get("execution") or {}
        ec = exec_rec.get("exit_code")
        sig = exec_rec.get("signal")
        if ec != 0 or sig:
            return [(0, sweep_args[0])]
        return []

    execs = {int(x.get("process_id", -1)): x for x in (data.get("executions") or [])}
    failed: List[Tuple[int, str]] = []
    for pid, arg_line in enumerate(sweep_args):
        rec = execs.get(pid)
        if rec is None or int(rec.get("exit_code", -1)) != 0 or rec.get("signal"):
            failed.append((pid, arg_line))
    return failed


def cmd_retry(args: argparse.Namespace, runs_root: Path) -> None:
    run_dir = resolve_run_dir(
        runs_root, args.run, prompt_title="Select run to retry/continue:"
    )
    live_clusters = query_live_condor_clusters()
    data = reconcile_run(run_dir, load_run_json(run_dir), live_clusters)
    repro_dir = run_dir / "repro"
    rj_path = run_dir / "run.json"

    req = data.get("resources_requested", {})
    sweep_args: List[str] = list(req.get("sweep_arguments") or [""])
    is_sweep = len(sweep_args) > 1

    if args.all:
        targets = list(enumerate(sweep_args))
    else:
        targets = get_failed_or_missing_procs(data)
        if not targets:
            console.print(
                "[yellow]All processes in this run completed with exit_code 0. Use `--all` to force re-run.[/yellow]"
            )
            return

    mode = "in_place" if args.in_place else ("new_run" if args.new_run else None)
    update_code = args.update_code

    if mode is None and sys.stdin.isatty():
        choice = questionary.select(
            f"Found {len(targets)}/{len(sweep_args)} process(es) to retry. Choose recovery mode:",
            choices=[
                questionary.Choice(
                    "1. In-Place Continuation + Update Code Snapshot (Fixes bug & keeps same $HERMES_RUN_DIR)",
                    value=("in_place", True),
                ),
                questionary.Choice(
                    "2. In-Place Continuation with Frozen Snapshot (Transient node/OOM failure)",
                    value=("in_place", False),
                ),
                questionary.Choice(
                    "3. Launch New Run via `send` (Creates fresh timestamped run folder)",
                    value=("new_run", True),
                ),
            ],
            style=PROMPT_STYLE,
        ).ask()
        if not choice:
            sys.exit(0)
        mode, update_code = choice
    elif mode is None:
        mode = "in_place"

    if mode == "new_run":
        send_bin = shutil.which("send") or str(Path(sys.executable).parent / "send")
        cwd = Path(data.get("submission", {}).get("working_directory", Path.cwd()))
        sub_meta = data.get("submission", {})
        cmd = [send_bin, str(sub_meta.get("target")), "-a", str(req.get("architecture"))]
        if req.get("cpus"):
            cmd.extend(["-c", str(req["cpus"])])
        if req.get("memory_mb"):
            cmd.extend(["-m", f"{req['memory_mb']}M"])
        if req.get("gpus") and req.get("architecture") != "GPU3090":
            cmd.extend(["-g", str(req["gpus"])])
        if sub_meta.get("tag"):
            cmd.extend(["-t", str(sub_meta["tag"])])
        if sub_meta.get("no_venv"):
            cmd.append("--no-venv")
        elif sub_meta.get("venv"):
            cmd.extend(["--venv", str(sub_meta["venv"])])
        for n in req.get("node_include") or []:
            cmd.extend(["--node", n])
        for ex in req.get("node_exclude") or []:
            cmd.extend(["--exclude", ex])
        if req.get("project_code"):
            cmd.extend(["--project", str(req["project_code"])])
        if req.get("scratch_files"):
            cmd.append("--scratch")
            cmd.extend(str(sf) for sf in req["scratch_files"])
        if is_sweep:
            for _, arg_line in targets:
                cmd.extend(["-s", arg_line])
        elif targets[0][1]:
            cmd.append(f"--args={targets[0][1]}")
        if args.yes:
            cmd.append("-y")

        console.print(
            f"[bold cyan]▶ Relaunching via send:[/bold cyan] {' '.join(shlex.quote(c) for c in cmd)}"
        )
        os.chdir(cwd)
        os.execv(send_bin, cmd)

    # --- Guard against colliding with still-running jobs in condor_q ---
    if live_clusters is not None and not args.force:
        active_cids = [c for c in get_all_cluster_ids(data) if c in live_clusters]
        if active_cids:
            console.print(
                f"[bold red]✘ Run still has active jobs in HTCondor cluster(s) {active_cids}.[/bold red]\n"
                f"  Wait for them to finish, run `runs stop {run_dir.name}` first, or pass `--force`."
            )
            sys.exit(1)

    # --- In-Place Continuation Mode ---
    if update_code:
        refresh_code_snapshot(run_dir, data)

    script_stem = Path(data.get("submission", {}).get("target", "job")).stem
    now = datetime.datetime.now()
    retry_epoch = int(now.timestamp())
    backup_tag = now.strftime("prev_%H%M%S")

    for pid, _ in targets:
        suffix = f"_{pid}" if is_sweep else ""
        for ext in (".out", ".err"):
            p = run_dir / f"{script_stem}{suffix}{ext}"
            if p.is_file():
                shutil.move(str(p), str(repro_dir / f"{p.name}.{backup_tag}"))
        node_f = repro_dir / f"node_{pid}.json"
        if node_f.is_file():
            node_f.unlink()

    retry_sub_content = generate_retry_sub_file(
        orig_sub_path=repro_dir / "job.sub",
        run_dir=run_dir,
        repro_dir=repro_dir,
        script_stem=script_stem,
        is_sweep=is_sweep,
        targets=targets,
        retry_epoch=retry_epoch,
    )
    retry_sub_path = repro_dir / "job_retry.sub"
    write_text_lf(retry_sub_path, retry_sub_content)

    res = subprocess.run(
        ["condor_submit", str(retry_sub_path)],
        cwd=run_dir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if res.returncode != 0:
        console.print(
            f"[bold red]✘ condor_submit failed:[/bold red]\n{(res.stderr or res.stdout).strip()}"
        )
        sys.exit(1)

    condor_out = res.stdout.strip()
    cluster_match = re.search(r"submitted to cluster\s+(\d+)", condor_out)
    new_cid = int(cluster_match.group(1)) if cluster_match else None

    def _record_retry(d: Dict[str, Any]) -> None:
        d["status"] = "QUEUED (RETRY)"
        if not is_sweep:
            d["execution"] = None
        else:
            retried_ids = {pid for pid, _ in targets}
            d["executions"] = [
                x for x in d.get("executions", []) if x.get("process_id") not in retried_ids
            ]
        sub_block = d.setdefault("submission", {})
        history = sub_block.setdefault("retry_clusters", [])
        old_cid = sub_block.get("cluster_id")
        if old_cid is not None and old_cid not in history:
            history.append(old_cid)
        sub_block["cluster_id"] = new_cid

    locked_mutate_run_json(rj_path, repro_dir, _record_retry)
    console.print(
        Panel(
            f"[bold green]✔ {condor_out}[/bold green]\n"
            f"  • Retried Processes : [cyan]{[pid for pid, _ in targets]}[/cyan]\n"
            f"  • Code Refreshed    : [cyan]{'Yes' if update_code else 'No (Frozen snapshot)'}[/cyan]\n"
            f"  • Archived Old Logs : [dim]repro/*.{backup_tag}[/dim]",
            title=f"[bold green]🔄 In-Place Recovery Submitted ({run_dir.name})[/bold green]",
            border_style="green",
            expand=False,
        )
    )


# ---------------------------------------------------------------------------
# Subcommand: `runs clean`
# ---------------------------------------------------------------------------
def cmd_clean(args: argparse.Namespace, runs_root: Path) -> None:
    dirs = list_run_dirs(runs_root)
    if not dirs:
        console.print("[dim]No runs to clean.[/dim]")
        return

    live_clusters = query_live_condor_clusters()
    now = datetime.datetime.now()
    candidates: List[Tuple[Path, str, float]] = []

    for d in dirs:
        data = reconcile_run(d, load_run_json(d), live_clusters)
        st = str(data.get("_live_badge") or data.get("status", "UNKNOWN"))
        if st.startswith(("RUNNING", "QUEUED", "HELD")):
            continue

        match = False
        if args.failed and (
            st in ("FAILED", "SUBMIT_FAILED", "VANISHED") or st.startswith("FAILED")
        ):
            match = True
        if args.aborted and (st in ("ABORTED", "INTERRUPTED") or st.startswith("INTERRUPTED")):
            match = True
        if args.dry and st == "DRY_RUN":
            match = True
        if args.older_than:
            m = re.match(r"^(\d+)([dhm])$", args.older_than.strip().lower())
            if m:
                val, unit = int(m.group(1)), m.group(2)
                delta = (
                    datetime.timedelta(days=val)
                    if unit == "d"
                    else (
                        datetime.timedelta(hours=val)
                        if unit == "h"
                        else datetime.timedelta(minutes=val)
                    )
                )
                mtime = datetime.datetime.fromtimestamp(d.stat().st_mtime)
                if now - mtime > delta:
                    match = True

        if match:
            candidates.append((d, st, dir_size_mb(d)))

    if not (args.failed or args.aborted or args.dry or args.older_than):
        if not sys.stdin.isatty():
            console.print(
                "[yellow]Specify --failed, --aborted, --dry, or --older-than (e.g. 7d) in non-interactive mode.[/yellow]"
            )
            return
        choices = []
        for d in dirs:
            data = load_run_json(d)
            st = str(data.get("status", "UNKNOWN"))
            if st.startswith(("RUNNING", "QUEUED", "HELD")):
                continue
            mb = dir_size_mb(d)
            choices.append(
                questionary.Choice(
                    f"{d.name:<42} │ {st:<20} │ {mb:6.2f} MB", value=(d, st, mb)
                )
            )
        if not choices:
            console.print("[green]No inactive runs eligible for deletion.[/green]")
            return
        selected = questionary.checkbox(
            "Select runs to permanently delete (Space to toggle, Enter to confirm):",
            choices=choices,
            style=PROMPT_STYLE,
        ).ask()
        if not selected:
            return
        candidates = selected

    if not candidates:
        console.print("[green]No matching runs found to clean.[/green]")
        return

    total_mb = sum(x[2] for x in candidates)
    for d, st, mb in candidates:
        console.print(f"  • [bold]{d.name}[/bold] ({style_status(st)} — {mb:.2f} MB)")

    if not args.yes:
        if not questionary.confirm(
            f"Delete {len(candidates)} run(s) and free {total_mb:.2f} MB?",
            default=False,
            style=PROMPT_STYLE,
        ).ask():
            return

    for d, _, _ in candidates:
        shutil.rmtree(d, ignore_errors=True)
    update_latest_symlink(runs_root)
    console.print(
        f"[bold green]✔ Deleted {len(candidates)} run(s) ({total_mb:.2f} MB freed).[/bold green]"
    )


# ---------------------------------------------------------------------------
# Subcommand: `runs diff`
# ---------------------------------------------------------------------------
def _diff_directories(dir_a: Path, dir_b: Path) -> str:
    """Computes a unified diff across all text files in two directories."""
    files_a = {
        p.relative_to(dir_a).as_posix() for p in dir_a.rglob("*") if p.is_file()
    }
    files_b = {
        p.relative_to(dir_b).as_posix() for p in dir_b.rglob("*") if p.is_file()
    }
    all_rel = sorted(files_a | files_b)
    chunks: List[str] = []

    for rel in all_rel:
        pa, pb = dir_a / rel, dir_b / rel
        lines_a = (
            pa.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
            if pa.is_file()
            else []
        )
        lines_b = (
            pb.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
            if pb.is_file()
            else []
        )
        if lines_a == lines_b:
            continue
        diff_lines = list(
            difflib.unified_diff(
                lines_a,
                lines_b,
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
            )
        )
        if diff_lines:
            chunks.append("".join(diff_lines))
    return "\n".join(chunks)


def cmd_diff(args: argparse.Namespace, runs_root: Path) -> None:
    dir_a = resolve_run_dir(runs_root, args.run_a, prompt_title="Select first run (A):")
    dir_b = resolve_run_dir(runs_root, args.run_b, prompt_title="Select second run (B):")
    da, db = load_run_json(dir_a), load_run_json(dir_b)

    t = Table(
        title="[bold cyan]⚖️  Run Provenance Comparison[/bold cyan]",
        box=box.ROUNDED,
        header_style="bold cyan",
    )
    t.add_column("Field", style="bold cyan")
    t.add_column(f"Run A: {dir_a.name}", style="white")
    t.add_column(f"Run B: {dir_b.name}", style="white")

    fields = [
        ("Status", da.get("status"), db.get("status")),
        (
            "Target",
            da.get("submission", {}).get("target"),
            db.get("submission", {}).get("target"),
        ),
        (
            "Architecture",
            da.get("resources_requested", {}).get("architecture"),
            db.get("resources_requested", {}).get("architecture"),
        ),
        (
            "Arguments / Sweep",
            str(da.get("resources_requested", {}).get("sweep_arguments")),
            str(db.get("resources_requested", {}).get("sweep_arguments")),
        ),
        (
            "Git Commit",
            str(da.get("git", {}).get("commit"))[:10],
            str(db.get("git", {}).get("commit"))[:10],
        ),
        ("Git Dirty", str(da.get("git", {}).get("dirty")), str(db.get("git", {}).get("dirty"))),
        ("Peak RAM", summarize_metrics(da)[2], summarize_metrics(db)[2]),
        ("Wall Time", summarize_metrics(da)[1], summarize_metrics(db)[1]),
    ]
    for label, va, vb in fields:
        style_b = "bold yellow" if va != vb else "white"
        t.add_row(label, str(va), f"[{style_b}]{vb}[/{style_b}]")
    console.print(t)

    code_a, code_b = dir_a / "repro" / "code", dir_b / "repro" / "code"
    if code_a.is_dir() and code_b.is_dir():
        diff_text = _diff_directories(code_a, code_b).strip()
        if diff_text:
            console.print(
                Panel(
                    Syntax(diff_text[:4000], "diff", theme="monokai"),
                    title="[bold yellow]Code Snapshot Diff (repro/code/)[/bold yellow]",
                    border_style="yellow",
                )
            )
        else:
            console.print(
                "[bold green]✔ Code snapshots (`repro/code/`) are byte-for-byte identical.[/bold green]"
            )


# ---------------------------------------------------------------------------
# CLI Argument Parser & Entrypoint
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="runs",
        description="Hermes 2 HTCondor Run Manager, Telemetry Inspector & Recovery CLI.",
    )
    sub = p.add_subparsers(dest="cmd")

    p_ls = sub.add_parser("ls", help="List runs with live HTCondor status reconciliation")
    p_ls.add_argument("-n", "--limit", type=int, default=25, help="Max runs to display (default: 25)")
    p_ls.add_argument("-s", "--status", type=str, help="Filter by status (e.g. RUNNING, FAILED)")
    p_ls.add_argument("-t", "--tag", type=str, help="Filter by substring/tag in run ID")
    p_ls.add_argument("-w", "--watch", action="store_true", help="Live-refresh table every N seconds")
    p_ls.add_argument("-i", "--interval", type=int, default=2, help="Refresh interval in seconds for --watch")

    p_watch = sub.add_parser("watch", help="Live-watch runs and HTCondor queue progress")
    p_watch.add_argument("-n", "--limit", type=int, default=25, help="Max runs to display (default: 25)")
    p_watch.add_argument("-s", "--status", type=str, help="Filter by status (e.g. RUNNING, FAILED)")
    p_watch.add_argument("-t", "--tag", type=str, help="Filter by substring/tag in run ID")
    p_watch.add_argument("-i", "--interval", type=int, default=2, help="Refresh interval in seconds (default: 2)")

    p_ins = sub.add_parser(
        "inspect", aliases=["info", "show"], help="Inspect run telemetry and stderr tails"
    )
    p_ins.add_argument("run", nargs="?", help="Run ID, index (#), cluster ID, or 'latest'")

    p_log = sub.add_parser("logs", aliases=["tail"], help="View or follow stdout/stderr/condor logs")
    p_log.add_argument("run", nargs="?", default="latest", help="Run ID, index (#), or 'latest'")
    p_log.add_argument(
        "-e", "--err", action="store_true", help="Show stderr (.err) instead of stdout (.out)"
    )
    p_log.add_argument(
        "-c", "--condor", action="store_true", help="Show HTCondor scheduler log (repro/condor.log)"
    )
    p_log.add_argument(
        "-p",
        "--proc",
        type=int,
        help="Process index for parallel sweeps (auto-selects failed proc on -e)",
    )
    p_log.add_argument("-f", "--follow", action="store_true", help="Follow live log output (tail -f)")
    p_log.add_argument("-n", "--lines", type=int, default=60, help="Number of tail lines to display")

    p_ret = sub.add_parser(
        "retry", aliases=["resume", "continue"], help="Retry failed runs or partial sweeps"
    )
    p_ret.add_argument("run", nargs="?", help="Run ID, index (#), or 'latest'")
    p_ret.add_argument("--in-place", action="store_true", help="Resume inside the same run directory")
    p_ret.add_argument(
        "--new-run", action="store_true", help="Launch a brand-new run directory via `send`"
    )
    p_ret.add_argument(
        "-u",
        "--update-code",
        action="store_true",
        help="Refresh repro/code/ snapshot from working tree before retrying",
    )
    p_ret.add_argument(
        "--all",
        action="store_true",
        help="Re-run all sweep processes (even those with exit_code 0)",
    )
    p_ret.add_argument(
        "--force",
        action="store_true",
        help="Allow in-place retry even if jobs are still active in condor_q",
    )
    p_ret.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompts")

    p_sync = sub.add_parser(
        "sync", help="Hot-patch a run's repro/code/ snapshot with current workspace files"
    )
    p_sync.add_argument("run", nargs="?", help="Run ID, index (#), or 'latest'")

    p_stop = sub.add_parser("stop", aliases=["kill", "rm"], help="Abort active HTCondor job (condor_rm)")
    p_stop.add_argument("run", nargs="?", help="Run ID, index (#), cluster ID, or 'latest'")
    p_stop.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")

    p_cln = sub.add_parser(
        "clean", aliases=["prune"], help="Delete failed, aborted, dry-run, or old runs"
    )
    p_cln.add_argument(
        "--failed", action="store_true", help="Remove FAILED, SUBMIT_FAILED, and VANISHED runs"
    )
    p_cln.add_argument("--aborted", action="store_true", help="Remove ABORTED and INTERRUPTED runs")
    p_cln.add_argument("--dry", action="store_true", help="Remove DRY_RUN capsules")
    p_cln.add_argument(
        "--older-than", type=str, help="Remove runs older than e.g. '7d', '24h', '30m'"
    )
    p_cln.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")

    p_diff = sub.add_parser("diff", help="Compare provenance and code snapshots between two runs")
    p_diff.add_argument("run_a", nargs="?", help="First run ID or index")
    p_diff.add_argument("run_b", nargs="?", help="Second run ID or index")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    runs_root = get_runs_root(Path.cwd())

    cmd = args.cmd or "ls"
    if cmd == "ls":
        if not hasattr(args, "limit"):
            args.limit, args.status, args.tag, args.watch, args.interval = 25, None, None, False, 2
        cmd_ls(args, runs_root)
    elif cmd == "watch":
        args.watch = True
        cmd_ls(args, runs_root)
    elif cmd in ("inspect", "info", "show"):
        cmd_inspect(args, runs_root)
    elif cmd in ("logs", "tail"):
        cmd_logs(args, runs_root)
    elif cmd in ("retry", "resume", "continue"):
        cmd_retry(args, runs_root)
    elif cmd == "sync":
        cmd_sync(args, runs_root)
    elif cmd in ("stop", "kill", "rm"):
        cmd_stop(args, runs_root)
    elif cmd in ("clean", "prune"):
        cmd_clean(args, runs_root)
    elif cmd == "diff":
        cmd_diff(args, runs_root)


if __name__ == "__main__":
    main()
