"""
runs_cli.py — HTCondor Run Manager, Telemetry Inspector & Recovery CLI entrypoint (`runs`).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich import box
from rich.panel import Panel
from rich.table import Table

from hermes_tools.condor import (
    build_send_command_tokens,
    compute_telemetry_summary,
    generate_retry_sub_file,
    get_all_cluster_ids,
    get_failed_or_missing_procs,
    materialize_dag_files,
    query_condor_better_analyze,
    query_condor_nodes,
    query_job_live_telemetry,
    query_live_condor_clusters,
    read_telemetry_csv,
    reconcile_run,
    run_needs_live_reconcile,
    sanitize_dagman_condor_sub,
    stream_job_live_telemetry,
    summarize_condor_pools,
)
from hermes_tools.rich_help import RichArgumentParser
from hermes_tools.core import (
    console,
    dir_size_mb,
    extract_run_display_parts as _extract_run_display_parts,
    format_duration,
    format_run_short as _format_run_short,
    get_execution_records,
    get_prompt_style,
    get_runs_root,
    get_username,
    get_watch_row_limit,
    list_run_dirs,
    load_run_json,
    locked_mutate_run_json,
    resolve_latest_symlink,
    set_execution_records,
    status_spinner,
    step_progress,
    style_status,
    summarize_metrics,
    tail_file_lines,
    update_latest_symlink,
    write_text_lf,
)
from hermes_tools.pull import (  # noqa: F401
    _REMOTE_PULL_SCRIPT,
    _parse_remote_spec,
    _resolve_pull_target,
    cmd_pull,
)
from hermes_tools.snapshot import (
    diff_directories as _diff_directories,
    refresh_code_snapshot,
)
from hermes_tools.telemetry import (
    _build_live_telemetry_panel,
    _build_telemetry_plot_panel,
    _save_telemetry_matplotlib,
)


def _format_waiting_status_with_run_index(
    display_status: str,
    data: Dict[str, Any],
    run_index_by_name: Dict[str, int],
) -> str:
    """Replaces cluster IDs in `WAITING (...)` with the parent run's `#N` table index when known."""
    if not display_status.startswith("WAITING"):
        return display_status
    dep = data.get("dependency")
    if isinstance(dep, dict):
        parent_name = dep.get("parent_run_id") or (
            Path(str(dep["parent_run_dir"])).name if dep.get("parent_run_dir") else None
        )
        if parent_name and str(parent_name) in run_index_by_name:
            return f"WAITING (#{run_index_by_name[str(parent_name)]})"
    return display_status


def resolve_run_dir(
    runs_root: Path,
    query: Optional[str],
    prompt_title: str = "Select a run:",
    exit_on_fail: bool = True,
) -> Optional[Path]:
    dirs = list_run_dirs(runs_root)
    if not dirs:
        if exit_on_fail:
            console.print(f"[yellow]No runs found in {runs_root}[/yellow]")
            sys.exit(1)
        return None

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
        q_lower = q.lower()
        for d in dirs:
            if q_lower in d.name.lower():
                matches.append(d)
            else:
                rj = load_run_json(d)
                all_cids = [str(c) for c in get_all_cluster_ids(rj)]
                if q in all_cids:
                    matches.append(d)

        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            if not exit_on_fail:
                return matches[0]
            dirs = matches
        else:
            if exit_on_fail:
                console.print(f"[bold red]✘ No run matching '{query}'.[/bold red]")
                sys.exit(1)
            return None

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return dirs[0]

    import questionary

    latest_target = resolve_latest_symlink(runs_root)
    run_index_by_name = {d.name: idx for idx, d in enumerate(dirs, start=1)}
    live_clusters_queried = False
    live_clusters: Optional[Dict[int, Dict[str, Any]]] = None
    rows: List[Tuple[int, Path, str, str, str, str, str, str]] = []
    for i, d in enumerate(dirs[:35], start=1):
        raw_rj = load_run_json(d)
        if run_needs_live_reconcile(raw_rj):
            if not live_clusters_queried:
                live_clusters = query_live_condor_clusters()
                live_clusters_queried = True
            rj = reconcile_run(d, raw_rj, live_clusters)
        else:
            rj = raw_rj
        plain_label, _, short_date = _extract_run_display_parts(d, rj)
        st = _format_waiting_status_with_run_index(
            str(rj.get("_live_badge") or rj.get("status", "UNKNOWN")),
            rj,
            run_index_by_name,
        )
        arch = str(rj.get("resources_requested", {}).get("architecture", "—"))
        _, dur_str, _ = summarize_metrics(rj)
        raw_cid = rj.get("submission", {}).get("cluster_id")
        cid_str = f"#{raw_cid}" if raw_cid else "—"
        star = " ★" if d == latest_target else ""
        pin_mark = " 📌" if rj.get("pinned") else ""
        rows.append((i, d, plain_label, arch, st, dur_str, short_date, f"{cid_str}{star}{pin_mark}"))

    name_w = max((len(r[2]) for r in rows), default=14)
    arch_w = max((len(r[3]) for r in rows), default=6)
    st_w = max((len(r[4]) for r in rows), default=8)
    dur_w = max((len(r[5]) for r in rows), default=6)
    date_w = max((len(r[6]) for r in rows), default=16)

    choices = [
        questionary.Choice(
            title=(
                f"[{i:02d}] {label:<{name_w}} │ {arch:<{arch_w}} │ {st:<{st_w}} │ "
                f"{dur:>{dur_w}} │ {dt:<{date_w}} │ {cid}"
            ),
            value=d,
        )
        for i, d, label, arch, st, dur, dt, cid in rows
    ]
    selected = questionary.select(
        prompt_title, choices=choices, style=get_prompt_style()
    ).ask()
    if not selected:
        sys.exit(0)
    return selected


# ---------------------------------------------------------------------------
# Subcommand: `runs ls` & `runs watch`
# ---------------------------------------------------------------------------
def _build_ls_table_with_state(
    args: argparse.Namespace,
    runs_root: Path,
    terminal_cache: Optional[Dict[Path, Tuple[Tuple[int, int, int], Dict[str, Any]]]] = None,
) -> Tuple[Table, Tuple[Tuple[str, ...], ...]]:
    dirs = list_run_dirs(runs_root)
    latest_target = resolve_latest_symlink(runs_root)

    table = Table(
        title="📊 Hermes 2 Runs",
        title_style="bold cyan",
        box=box.ROUNDED,
        header_style="bold cyan",
        padding=(0, 1),
    )
    max_run_width = max(14, console.width - 102)
    table.add_column("#", style="dim", justify="right", no_wrap=True)
    table.add_column("Run", no_wrap=True, overflow="ellipsis", max_width=max_run_width)
    table.add_column("Submitted", style="dim", no_wrap=True)
    table.add_column("Cluster", style="cyan", justify="right", no_wrap=True)
    table.add_column("Arch", style="magenta", no_wrap=True)
    table.add_column("Status", no_wrap=True, overflow="ellipsis")
    table.add_column("Tasks", justify="center", no_wrap=True)
    table.add_column("Host(s)", style="dim", no_wrap=True)
    table.add_column("Wall Time", justify="right", no_wrap=True)
    table.add_column("Peak RAM", justify="right", no_wrap=True)

    shown = 0
    limit = getattr(args, "limit", 25)
    if getattr(args, "watch", False) and limit:
        limit = get_watch_row_limit(limit, chrome_lines=6)
    status_filter = getattr(args, "status", None)
    status_upper = status_filter.upper() if status_filter else None
    tag_filter = getattr(args, "tag", None)
    tag_lower = tag_filter.lower() if tag_filter else None
    pinned_only = bool(getattr(args, "pinned", False))

    live_clusters_queried = False
    live_clusters: Optional[Dict[int, Dict[str, int]]] = None
    row_signatures: List[Tuple[str, ...]] = []
    run_index_by_name = {d.name: idx for idx, d in enumerate(dirs, start=1)}

    for idx, d in enumerate(dirs, start=1):
        rj_stat_key: Optional[Tuple[int, int, int]] = None
        if terminal_cache is not None:
            try:
                st_rj = (d / "run.json").stat()
                rj_stat_key = (st_rj.st_ino, st_rj.st_mtime_ns, st_rj.st_size)
            except OSError:
                rj_stat_key = None

        cached_entry = terminal_cache.get(d) if terminal_cache is not None else None
        if cached_entry is not None and rj_stat_key is not None and cached_entry[0] == rj_stat_key:
            data = cached_entry[1]
        else:
            raw_data = load_run_json(d)
            if run_needs_live_reconcile(raw_data):
                if not live_clusters_queried:
                    live_clusters = query_live_condor_clusters()
                    live_clusters_queried = True
                data = reconcile_run(d, raw_data, live_clusters)
            else:
                data = raw_data
            if (
                terminal_cache is not None
                and not run_needs_live_reconcile(data)
                and "_live_badge" not in data
            ):
                try:
                    st_after = (d / "run.json").stat()
                    terminal_cache[d] = (
                        (st_after.st_ino, st_after.st_mtime_ns, st_after.st_size),
                        data,
                    )
                except OSError:
                    pass
            elif terminal_cache is not None:
                terminal_cache.pop(d, None)

        if pinned_only and not bool(data.get("pinned", False)):
            continue

        display_status = _format_waiting_status_with_run_index(
            str(data.get("_live_badge") or data.get("status", "UNKNOWN")),
            data,
            run_index_by_name,
        )
        if status_upper and status_upper not in display_status.upper():
            continue

        plain_label, rich_label, short_date = _extract_run_display_parts(d, data)
        if tag_lower and tag_lower not in d.name.lower() and tag_lower not in plain_label.lower():
            continue

        req = data.get("resources_requested", {})
        arch = str(req.get("architecture", "—"))
        sweep_cnt = int(req.get("sweep_count", 1) or 1)
        cids = get_all_cluster_ids(data)
        cid_str = (
            str(cids[-1]) + (f" (+{len(cids) - 1})" if len(cids) - 1 > 0 else "")
            if cids
            else "—"
        )

        if sweep_cnt > 1:
            execs = get_execution_records(data)
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
        is_pinned = bool(data.get("pinned", False))
        prefix_badges: List[str] = []
        if d == latest_target:
            prefix_badges.append("[bold yellow]★[/bold yellow]")
        if is_pinned:
            prefix_badges.append("[bold magenta]📌[/bold magenta]")
        name_label = f"{' '.join(prefix_badges)} {rich_label}" if prefix_badges else rich_label

        row_cells = (
            str(idx),
            name_label,
            short_date,
            cid_str,
            arch,
            style_status(display_status),
            tasks_str,
            host_str,
            dur_str,
            ram_str,
        )
        table.add_row(*row_cells)
        row_signatures.append(row_cells)
        shown += 1
        if limit and shown >= limit:
            break

    return table, tuple(row_signatures)


def _build_ls_table(args: argparse.Namespace, runs_root: Path) -> Table:
    table, _ = _build_ls_table_with_state(args, runs_root)
    return table


def cmd_ls(args: argparse.Namespace, runs_root: Path) -> None:
    dirs = list_run_dirs(runs_root)
    if not dirs:
        console.print("[dim]No runs found.[/dim]")
        return

    if getattr(args, "watch", False):
        from rich.live import Live

        interval = max(1, int(getattr(args, "interval", 2) or 2))
        terminal_cache: Dict[Path, Tuple[Tuple[int, int, int], Dict[str, Any]]] = {}
        initial_table, last_sig = _build_ls_table_with_state(
            args, runs_root, terminal_cache
        )
        current_view = initial_table
        try:
            with Live(
                initial_table,
                console=console,
                screen=True,
                auto_refresh=False,
            ) as live:
                while True:
                    time.sleep(interval)
                    new_table, new_sig = _build_ls_table_with_state(
                        args, runs_root, terminal_cache
                    )
                    if new_sig != last_sig:
                        last_sig = new_sig
                        current_view = new_table
                        live.update(new_table, refresh=True)
        except KeyboardInterrupt:
            console.print(current_view)
            return
    else:
        console.print(_build_ls_table(args, runs_root))


# ---------------------------------------------------------------------------
# Subcommand: `runs inspect`
# ---------------------------------------------------------------------------
def cmd_inspect(args: argparse.Namespace, runs_root: Path) -> None:
    run_dir = resolve_run_dir(runs_root, args.run)
    raw_data = load_run_json(run_dir)
    if run_needs_live_reconcile(raw_data):
        live_clusters = query_live_condor_clusters()
        data = reconcile_run(run_dir, raw_data, live_clusters)
    else:
        data = raw_data

    sub = data.get("submission", {})
    req = data.get("resources_requested", {})
    git = data.get("git", {})
    display_status = str(data.get("_live_badge") or data.get("status", "UNKNOWN"))
    cids = get_all_cluster_ids(data)

    plain_label, rich_label, short_date = _extract_run_display_parts(run_dir, data)

    meta = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
    meta.add_column("Key", style="bold cyan", width=20)
    meta.add_column("Val", style="white")

    meta.add_row("Run", f"{rich_label} [dim]({run_dir.name})[/dim]")
    cloned_from = sub.get("cloned_from_run")
    if cloned_from:
        cloned_dir = runs_root / str(cloned_from)
        if cloned_dir.is_dir():
            cloned_disp = _format_run_short(cloned_dir)
        else:
            _, c_rich, _ = _extract_run_display_parts(Path(str(cloned_from)), {})
            cloned_disp = c_rich
        meta.add_row("Cloned From", cloned_disp)
    status_val = style_status(display_status)
    if bool(data.get("pinned", False)):
        status_val = f"{status_val} [bold magenta][📌 PINNED][/bold magenta]"
    meta.add_row("Status", status_val)
    if data.get("note"):
        meta.add_row("Note", f"[italic bright_white]{data['note']}[/italic bright_white]")
    dep = data.get("dependency")
    if isinstance(dep, dict) and dep.get("parent_run_id"):
        p_id = dep.get("parent_run_id")
        p_dir = dep.get("parent_run_dir")
        p_cid = dep.get("parent_cluster_id")
        r_ok = dep.get("require_ok", False)
        if p_dir and Path(p_dir).is_dir():
            p_label = _format_run_short(Path(p_dir))
        else:
            _, p_rich, _ = _extract_run_display_parts(Path(str(p_id)), {})
            p_label = p_rich
        cid_part = f" (#{p_cid})" if p_cid else ""
        cond_part = " [bold yellow][exit 0 required][/bold yellow]" if r_ok else ""
        meta.add_row("Dependency", f"After {p_label}{cid_part}{cond_part}")
    hold_reasons = data.get("_hold_reasons") or []
    if hold_reasons:
        meta.add_row(
            "Hold Reason",
            f"[bold magenta]{' ; '.join(str(r) for r in hold_reasons)}[/bold magenta]",
        )
    meta.add_row("HTCondor Cluster(s)", ", ".join(str(c) for c in cids) if cids else "—")
    raw_sub_at = str(sub.get("submitted_at", "") or "").strip()
    submitted_fmt = raw_sub_at.replace("T", " ")[:19] if raw_sub_at else short_date
    meta.add_row(
        "Submitted At",
        f"{submitted_fmt} by {sub.get('user', '—')}@{sub.get('submit_host', '—')}",
    )
    meta.add_row(
        "Target Script",
        f"[bold]{sub.get('target', '—')}[/bold]",
    )
    meta.add_row(
        "Architecture",
        f"[bold magenta]{req.get('architecture', '—')}[/bold magenta] (CPUs: {req.get('cpus') or 'Auto'}, RAM: {req.get('memory_mb') or 'Auto'} MB, GPUs: {req.get('gpus') or 0})",
    )
    if req.get("max_parallel"):
        meta.add_row("Max Parallel", f"{req['max_parallel']} concurrent job(s)")
    env_map = req.get("env") or {}
    if isinstance(env_map, dict) and env_map:
        env_str = ", ".join(f"[cyan]{k}[/cyan]={v}" for k, v in env_map.items())
        meta.add_row("Env Exports", env_str)
    if git.get("is_git"):
        dirty_str = (
            f"[yellow]dirty ({git.get('patch_file')})[/yellow]"
            if git.get("dirty")
            else "[green]clean[/green]"
        )
        synced_fmt = str(git.get("synced_at", "")).replace("T", " ")
        sync_note = f" [dim](synced {synced_fmt})[/dim]" if synced_fmt else ""
        meta.add_row(
            "Git Provenance",
            f"{git.get('branch')} @ {str(git.get('commit'))[:8]} ({dirty_str}){sync_note}",
        )
    meta.add_row("Disk Footprint", f"{dir_size_mb(run_dir):.2f} MB")
    meta.add_row("Repro CLI", f"[bold green]{sub.get('equivalent_cli', '—')}[/bold green]")

    console.print(
        Panel(
            meta,
            title="[bold cyan]🔍 Run Provenance & Telemetry[/bold cyan]",
            border_style="cyan",
            expand=False,
        )
    )

    if cids and display_status.startswith(("QUEUED", "HELD")):
        _render_queue_analysis_panel(f"{cids[-1]}.0", quiet_if_empty=True)

    records = get_execution_records(data)

    if records:
        et = Table(
            title="Process Execution Breakdown",
            title_style="bold cyan",
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

        telem_rows = []
        for r in records:
            pid = r.get("process_id", 0)
            tsum = r.get("telemetry_summary")
            if not isinstance(tsum, dict):
                tsum = compute_telemetry_summary(run_dir / "repro" / f"telemetry_{pid}.csv")
            if isinstance(tsum, dict) and tsum.get("samples", 0) > 0:
                telem_rows.append((pid, tsum))

        if telem_rows:
            tt = Table(
                title="Telemetry Summary",
                title_style="bold cyan",
                box=box.SIMPLE_HEAVY,
                header_style="bold cyan",
            )
            tt.add_column("Proc", justify="right")
            tt.add_column("Samples", justify="right")
            tt.add_column("GPU Util (Avg / Max)", justify="right")
            tt.add_column("Peak VRAM", justify="right")
            tt.add_column("PSS / RSS Max", justify="right")
            tt.add_column("I/O Wait %", justify="right")
            for pid, tsum in telem_rows:
                vtot = float(tsum.get("vram_total_mb", 0.0) or 0.0)
                vmax = float(tsum.get("vram_mb_max", 0.0) or 0.0)
                gavg = float(tsum.get("gpu_util_avg", 0.0) or 0.0)
                gmax = float(tsum.get("gpu_util_max", 0.0) or 0.0)
                gpu_cell = f"{gavg:.0f}% / {gmax:.0f}%" if (vtot > 0 or gmax > 0) else "—"
                vram_cell = (
                    f"{vmax / 1024:.2f} / {vtot / 1024:.1f} GB"
                    if vtot > 0
                    else (f"{vmax:.0f} MB" if vmax > 0 else "—")
                )
                pss_m = float(tsum.get("pss_mb_max", 0.0) or 0.0)
                rss_m = float(tsum.get("rss_mb_max", 0.0) or 0.0)
                ram_cell = f"{pss_m:.0f} / {rss_m:.0f} MB"
                iow = float(tsum.get("io_wait_pct", 0.0) or 0.0)
                iow_cell = f"[yellow]{iow:.1f}%[/yellow]" if iow > 15.0 else f"{iow:.1f}%"
                tt.add_row(str(pid), str(tsum.get("samples", 0)), gpu_cell, vram_cell, ram_cell, iow_cell)
            console.print(tt)

    err_files = sorted(run_dir.glob("*.err"))
    syntax_cls = None
    shown_any_err = False
    for ef in err_files:
        try:
            if ef.stat().st_size == 0:
                continue
        except OSError:
            continue
        lines = [line for line in tail_file_lines(ef, 25) if line.strip()]
        if lines:
            shown_any_err = True
            if syntax_cls is None:
                from rich.syntax import Syntax as syntax_cls
            console.print(
                Panel(
                    syntax_cls("\n".join(lines), "pytb", theme="monokai", word_wrap=True),
                    title=f"[bold red]⚠ Stderr Tail: {ef.name}[/bold red]",
                    border_style="red",
                )
            )

    if not shown_any_err and display_status.startswith(
        ("FAILED", "COMPLETED_WITH_ERRORS", "INTERRUPTED")
    ):
        for of in sorted(run_dir.glob("*.out")):
            try:
                if of.stat().st_size == 0:
                    continue
            except OSError:
                continue
            out_lines = [line for line in tail_file_lines(of, 25) if line.strip()]
            if out_lines:
                if syntax_cls is None:
                    from rich.syntax import Syntax as syntax_cls
                console.print(
                    Panel(
                        syntax_cls("\n".join(out_lines), "text", theme="monokai", word_wrap=True),
                        title=f"[bold yellow]⚠ Stdout Tail (stderr empty): {of.name}[/bold yellow]",
                        border_style="yellow",
                    )
                )


# ---------------------------------------------------------------------------
# Subcommand: `runs logs`
# ---------------------------------------------------------------------------
def follow_run_log(run_dir: Path, log_file: Path, initial_lines: int = 60) -> None:
    """
    Waits if the job is still QUEUED, prints the last `initial_lines` in O(1) memory,
    streams live output, and exits automatically once the run reaches a terminal status.
    """
    try:
        if not log_file.is_file():
            raw_rj = load_run_json(run_dir)
            live_cls = query_live_condor_clusters()
            if live_cls and any(c in live_cls for c in get_all_cluster_ids(raw_rj)):
                raw_rj = reconcile_run(run_dir, raw_rj, live_cls)
            if not run_needs_live_reconcile(raw_rj):
                st = str(raw_rj.get("status", "UNKNOWN"))
                if st.startswith("CANCELLED"):
                    reason = raw_rj.get("cancelled_reason", "")
                    r_str = f" ({reason})" if reason else ""
                    console.print(f"[yellow]ℹ Run was cancelled{r_str}. Log file was not created.[/yellow]")
                    return
                console.print(f"[bold red]✘ Log file not found:[/bold red] {log_file}")
                sys.exit(1)

            st = str(raw_rj.get("status", "QUEUED"))
            if st.startswith("WAITING"):
                dep = raw_rj.get("dependency", {})
                p_label = dep.get("parent_run_id") or dep.get("parent_cluster_id") or "parent"
                console.print(
                    f"[dim yellow]⏳ Run is WAITING on {p_label} to finish (Ctrl+C to detach)...[/dim yellow]"
                )
            else:
                console.print(
                    f"[dim yellow]⏳ Waiting for job to start and create {log_file.name} (Ctrl+C to detach)...[/dim yellow]"
                )

            wait_ticks = 0
            last_waiting = st.startswith("WAITING")
            while not log_file.is_file():
                time.sleep(0.4)
                wait_ticks += 1
                if wait_ticks % 8 == 0:
                    raw_rj = load_run_json(run_dir)
                    rec = reconcile_run(run_dir, raw_rj, query_live_condor_clusters())
                    st = str(rec.get("_live_badge") or rec.get("status", "UNKNOWN"))
                    if st.startswith("HELD"):
                        reasons = rec.get("_hold_reasons") or []
                        r_str = f": {' ; '.join(reasons)}" if reasons else ""
                        console.print(
                            f"[bold magenta]⚠ Job entered HELD state{r_str}[/bold magenta]"
                        )
                        return
                    if not run_needs_live_reconcile(rec) and "_live_badge" not in rec:
                        if log_file.is_file():
                            break
                        term_st = str(rec.get("status", st))
                        if term_st.startswith("CANCELLED"):
                            reason = rec.get("cancelled_reason", "")
                            r_str = f": {reason}" if reason else ""
                            console.print(
                                f"[yellow]ℹ Run {style_status(term_st)}{r_str}. Log file was not created.[/yellow]"
                            )
                        else:
                            console.print(
                                f"[yellow]ℹ Job finished ({style_status(term_st)}) without creating {log_file.name}.[/yellow]"
                            )
                        return
                    is_waiting = st.startswith("WAITING")
                    if last_waiting and not is_waiting:
                        console.print(
                            f"[dim green]✔ Parent job finished! Starting log stream for {log_file.name}...[/dim green]"
                        )
                        last_waiting = False

        console.print(
            f"[dim cyan]Following {log_file.name} (auto-exits when job finishes, Ctrl+C to detach)...[/dim cyan]"
        )
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            f.seek(0, os.SEEK_END)
            end_offset = f.tell()
            tail = tail_file_lines(log_file, initial_lines)
            if tail:
                sys.stdout.write("\n".join(tail) + "\n")
                sys.stdout.flush()
            f.seek(end_offset, os.SEEK_SET)

            idle_ticks = 0
            while True:
                line = f.readline()
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    idle_ticks = 0
                    continue

                time.sleep(0.25)
                idle_ticks += 1
                if idle_ticks >= 4:
                    raw_rj = load_run_json(run_dir)
                    if not run_needs_live_reconcile(raw_rj):
                        time.sleep(0.15)
                        rem = f.read()
                        if rem:
                            sys.stdout.write(rem)
                            sys.stdout.flush()
                        st = str(raw_rj.get("status", "COMPLETED"))
                        console.rule(
                            f"[dim]Run finished:[/dim] {style_status(st)}",
                            style="dim cyan",
                        )
                        return
                    if idle_ticks % 16 == 0:
                        rec = reconcile_run(
                            run_dir, raw_rj, query_live_condor_clusters()
                        )
                        if not run_needs_live_reconcile(rec) and "_live_badge" not in rec:
                            time.sleep(0.15)
                            rem = f.read()
                            if rem:
                                sys.stdout.write(rem)
                                sys.stdout.flush()
                            st = str(rec.get("status", "UNKNOWN"))
                            console.rule(
                                f"[dim]Run finished:[/dim] {style_status(st)}",
                                style="dim cyan",
                            )
                            return
    except KeyboardInterrupt:
        return


def _resolve_run_log_file(
    run_dir: Path,
    data: Dict[str, Any],
    err: bool = False,
    condor: bool = False,
    proc: Optional[int] = None,
    quiet: bool = False,
) -> Path:
    target_stem = Path(data.get("submission", {}).get("target", "job")).stem
    sweep_cnt = int(data.get("resources_requested", {}).get("sweep_count", 1) or 1)
    if condor:
        return run_dir / "repro" / "condor.log"
    ext = ".err" if err else ".out"
    if sweep_cnt > 1:
        resolved_proc = proc
        if resolved_proc is None:
            if err:
                failed_execs = [
                    int(x.get("process_id", 0))
                    for x in get_execution_records(data)
                    if int(x.get("exit_code", 0)) != 0
                ]
                if failed_execs:
                    resolved_proc = failed_execs[0]
                    if not quiet:
                        console.print(
                            f"[dim yellow]ℹ Auto-selected failing process #{resolved_proc} (use `-p <id>` to override)[/dim yellow]"
                        )
                else:
                    for idx in range(sweep_cnt):
                        cand = run_dir / f"{target_stem}_{idx}.err"
                        if cand.is_file() and cand.stat().st_size > 0:
                            resolved_proc = idx
                            break
            if resolved_proc is None:
                resolved_proc = 0
        return run_dir / f"{target_stem}_{resolved_proc}{ext}"
    return run_dir / f"{target_stem}{ext}"


def cmd_logs(args: argparse.Namespace, runs_root: Path) -> None:
    run_dir = resolve_run_dir(runs_root, args.run)
    data = load_run_json(run_dir)
    log_file = _resolve_run_log_file(
        run_dir,
        data,
        err=bool(getattr(args, "err", False)),
        condor=bool(getattr(args, "condor", False)),
        proc=getattr(args, "proc", None),
    )

    if args.follow:
        follow_run_log(run_dir, log_file, args.lines)
        return

    if not log_file.is_file():
        st = str(data.get("status", "UNKNOWN"))
        if st.startswith("QUEUED"):
            console.print(
                f"[yellow]⏳ Run [bold]{run_dir.name}[/bold] is currently {st} ({log_file.name} not created yet).[/yellow]\n"
                f"  [dim]Tip: Use [bold]runs logs -f[/bold] to wait for start and stream live.[/dim]"
            )
            return
        console.print(f"[bold red]✘ Log file not found:[/bold red] {log_file}")
        sys.exit(1)

    snippet = "\n".join(tail_file_lines(log_file, args.lines))
    rel_title = (
        log_file.relative_to(runs_root.parent)
        if log_file.is_relative_to(runs_root.parent)
        else log_file
    )
    console.rule(f"[bold cyan]{rel_title}[/bold cyan]", style="cyan")
    if snippet:
        sys.stdout.write(snippet + "\n")
        sys.stdout.flush()
    console.rule(style="dim cyan")


# ---------------------------------------------------------------------------
# Subcommand: `runs stop`
# ---------------------------------------------------------------------------
def _stop_single_run(
    run_dir: Path,
    target_cids: List[int],
    is_active_status: bool,
) -> None:
    repro_dir = run_dir / "repro"
    rj_path = run_dir / "run.json"
    any_removed = False
    for cid in target_cids:
        res = subprocess.run(
            ["condor_rm", str(cid)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if res.returncode == 0:
            any_removed = True
            console.print(f"[bold green]✔ {res.stdout.strip()}[/bold green]")
        else:
            console.print(
                f"[yellow]⚠ condor_rm {cid}: {(res.stderr or res.stdout).strip()}[/yellow]"
            )

    if any_removed or is_active_status:
        locked_mutate_run_json(rj_path, repro_dir, lambda d: d.update({"status": "ABORTED"}))
        console.print(f"[cyan]Updated {run_dir.name}/run.json status -> ABORTED[/cyan]")


def _find_downstream_runs(
    runs_root: Path,
    root_run_dir: Path,
    root_cids: List[int],
    live_clusters: Optional[Dict[int, Dict[str, Any]]],
    *,
    active_only: bool,
    include_completed: bool = False,
) -> List[Tuple[Path, List[int], Dict[str, Any]]]:
    """
    Finds transitive downstream runs depending on `root_run_dir` or `root_cids`.
    When `active_only=True` (for `runs stop`), returns active/waiting downstream runs.
    When `active_only=False` (for `runs retry`), returns inactive downstream runs in pipeline order.
    """
    dirs = list_run_dirs(runs_root)
    if not dirs:
        return []
    if not active_only:
        dirs = list(reversed(dirs))

    loaded: List[Tuple[Path, Dict[str, Any], List[int], List[int], str, bool]] = []
    for d in dirs:
        if d == root_run_dir:
            continue
        raw_data = load_run_json(d)
        if not raw_data:
            continue
        dep = raw_data.get("dependency")
        if not isinstance(dep, dict):
            continue
        data = (
            reconcile_run(d, raw_data, live_clusters)
            if run_needs_live_reconcile(raw_data)
            else raw_data
        )
        st = str(data.get("_live_badge") or data.get("status", "UNKNOWN"))
        cids = get_all_cluster_ids(data)
        act_cids = [
            c
            for c in cids
            if (live_clusters is None if active_only else False)
            or (live_clusters is not None and c in live_clusters)
        ]
        is_active = st.startswith(("QUEUED", "RUNNING", "HELD", "WAITING")) or bool(act_cids)
        if active_only and not is_active:
            continue
        loaded.append((d, data, cids, act_cids or cids, st, is_active))

    downstream: List[Tuple[Path, List[int], Dict[str, Any]]] = []
    frontier_names = {root_run_dir.name}
    frontier_cids = set(root_cids)
    visited: set[Path] = {root_run_dir}

    changed = True
    while changed:
        changed = False
        for d, data, cids, tcids, st, is_active in loaded:
            if d in visited:
                continue
            dep = data.get("dependency") or {}
            p_run_id = dep.get("parent_run_id")
            p_cid = dep.get("parent_cluster_id")
            matches_parent = (p_run_id and str(p_run_id) in frontier_names) or (
                p_cid is not None and int(p_cid) in frontier_cids
            )
            if matches_parent:
                visited.add(d)
                frontier_names.add(d.name)
                frontier_cids.update(tcids if active_only else cids)
                changed = True
                if not active_only:
                    if is_active or (st == "COMPLETED" and not include_completed):
                        continue
                downstream.append((d, tcids, data))

    return downstream


def _find_downstream_waiting_runs(
    runs_root: Path,
    root_run_dir: Path,
    root_cids: List[int],
    live_clusters: Optional[Dict[int, Dict[str, Any]]],
) -> List[Tuple[Path, List[int], Dict[str, Any]]]:
    """Finds transitive downstream active/waiting runs depending on root_run_dir or root_cids."""
    return _find_downstream_runs(
        runs_root, root_run_dir, root_cids, live_clusters, active_only=True
    )


def cmd_stop(args: argparse.Namespace, runs_root: Path) -> None:
    if getattr(args, "all", False):
        dirs = list_run_dirs(runs_root)
        if not dirs:
            console.print("[dim]No runs found.[/dim]")
            return
        live_clusters = query_live_condor_clusters()
        active_targets: List[Tuple[Path, List[int], bool, str, Dict[str, Any]]] = []
        for d in dirs:
            raw_data = load_run_json(d)
            if not run_needs_live_reconcile(raw_data):
                continue
            data = reconcile_run(d, raw_data, live_clusters)
            st = str(data.get("_live_badge") or data.get("status", "UNKNOWN"))
            cids = get_all_cluster_ids(data)
            if not cids:
                continue
            is_act = st.startswith(("QUEUED", "RUNNING", "HELD", "WAITING"))
            act_cids = [c for c in cids if live_clusters is None or c in live_clusters]
            if act_cids or is_act:
                active_targets.append((d, act_cids or cids, is_act, st, data))

        if not active_targets:
            console.print("[green]No active (QUEUED/RUNNING/HELD/WAITING) runs found.[/green]")
            return

        for d, tcids, _, st, rdata in active_targets:
            console.print(
                f"  • {_format_run_short(d, rdata)} ({style_status(st)} — cluster(s) {tcids})"
            )
        if not args.yes:
            import questionary

            if not questionary.confirm(
                f"Abort all {len(active_targets)} active run(s)?",
                default=True,
                style=get_prompt_style(),
            ).ask():
                return

        for d, tcids, is_act, _, _ in active_targets:
            _stop_single_run(d, tcids, is_act)
        return

    run_dir = resolve_run_dir(
        runs_root, args.run, prompt_title="Select run to stop (condor_rm):"
    )
    with status_spinner("Checking HTCondor queue & reconciling run state..."):
        live_clusters = query_live_condor_clusters()
        data = reconcile_run(run_dir, load_run_json(run_dir), live_clusters)
    status = str(data.get("_live_badge") or data.get("status", "UNKNOWN"))
    cids = get_all_cluster_ids(data)

    if not cids:
        console.print(f"[yellow]Run {run_dir.name} has no HTCondor cluster_id.[/yellow]")
        return

    is_active_status = status.startswith(("QUEUED", "RUNNING", "HELD", "WAITING"))
    active_cids = [c for c in cids if live_clusters is None or c in live_clusters]

    if not active_cids and not is_active_status:
        if status.startswith("CANCELLED"):
            console.print(
                f"[dim cyan]ℹ Run [bold]{run_dir.name}[/bold] is already {style_status(status)}.[/dim cyan]"
            )
            return
        console.print(
            f"[dim cyan]ℹ Run [bold]{run_dir.name}[/bold] has already finished ({style_status(status)}) "
            f"and has no active jobs in condor_q.[/dim cyan]\n"
            f"  [dim]Tip: Use [bold]runs clean[/bold] to delete run directories from disk.[/dim]"
        )
        return

    target_cids = active_cids or cids
    downstream = (
        []
        if getattr(args, "single", False)
        else _find_downstream_waiting_runs(runs_root, run_dir, cids, live_clusters)
    )

    stop_downstream = bool(downstream)
    if downstream:
        console.print(
            f"[yellow]🔗 Found {len(downstream)} downstream chained run(s) waiting on {_format_run_short(run_dir, data)}:[/yellow]"
        )
        for dep_dir, dep_cids, dep_data in downstream:
            dep_st = str(dep_data.get("_live_badge") or dep_data.get("status", "UNKNOWN"))
            console.print(
                f"  • {_format_run_short(dep_dir, dep_data)} ({style_status(dep_st)} — cluster(s) {dep_cids})"
            )

        if not args.yes and sys.stdin.isatty():
            import questionary

            choice = questionary.select(
                "How would you like to stop this chain?",
                choices=[
                    questionary.Choice(
                        f"1. Stop entire chain ({1 + len(downstream)} runs)",
                        value="chain",
                    ),
                    questionary.Choice(
                        f"2. Stop only this run ({run_dir.name})",
                        value="single",
                    ),
                    questionary.Choice("3. Cancel", value="cancel"),
                ],
                style=get_prompt_style(),
            ).ask()
            if not choice or choice == "cancel":
                return
            stop_downstream = choice == "chain"
    elif not args.yes:
        import questionary

        if not questionary.confirm(
            f"Kill HTCondor cluster(s) {target_cids} ({run_dir.name})?",
            default=True,
            style=get_prompt_style(),
        ).ask():
            return

    _stop_single_run(run_dir, target_cids, is_active_status)
    if stop_downstream:
        for dep_dir, dep_cids, _ in downstream:
            _stop_single_run(dep_dir, dep_cids, True)


# ---------------------------------------------------------------------------
# Subcommand: `runs sync`
# ---------------------------------------------------------------------------
def cmd_sync(args: argparse.Namespace, runs_root: Path) -> None:
    run_dir = resolve_run_dir(
        runs_root, args.run, prompt_title="Select run to sync code snapshot into:"
    )
    data = load_run_json(run_dir)
    status = str(data.get("status", "UNKNOWN"))
    if status.startswith("CANCELLED"):
        console.print(
            f"[yellow]⚠ Run {run_dir.name} is {style_status(status)}. Cannot sync code snapshot to cancelled runs.[/yellow]"
        )
        return
    refresh_code_snapshot(run_dir, data)


# ---------------------------------------------------------------------------
# Subcommand: `runs retry`
# ---------------------------------------------------------------------------
def _find_downstream_retry_chain(
    runs_root: Path,
    root_run_dir: Path,
    root_cids: List[int],
    live_clusters: Optional[Dict[int, Dict[str, Any]]],
    include_completed: bool = False,
) -> List[Tuple[Path, Dict[str, Any]]]:
    """
    Finds transitive downstream runs (in chronological pipeline order) that depended on
    `root_run_dir` or `root_cids` and are inactive (e.g. CANCELLED, ABORTED, FAILED).
    """
    return [
        (d, data)
        for d, _, data in _find_downstream_runs(
            runs_root,
            root_run_dir,
            root_cids,
            live_clusters,
            active_only=False,
            include_completed=include_completed,
        )
    ]


def resolve_after_parent(
    runs_root: Path,
    after_arg: str,
    after_ok: bool,
    live_clusters: Optional[Dict[int, Dict[str, Any]]],
    *,
    self_run_dir: Optional[Path] = None,
    dry_run: bool = False,
) -> Tuple[Optional[Path], Optional[int], bool, str]:
    """
    Resolves `--after <after_arg>` to `(parent_run_dir, parent_cid, submit_via_dag, resolved_after_id)`.
    Shared by `send` and `runs retry`.
    """
    parent_run_dir = resolve_run_dir(runs_root, after_arg, exit_on_fail=False)
    if self_run_dir is not None and parent_run_dir == self_run_dir:
        console.print(
            "[bold red]✘ Error:[/bold red] A run cannot be chained after itself (--after)."
        )
        sys.exit(1)

    if parent_run_dir and parent_run_dir.is_dir():
        parent_data = load_run_json(parent_run_dir)
        if not parent_data:
            console.print(
                f"[bold red]✘ Error:[/bold red] Could not load run.json from parent run '{parent_run_dir.name}'."
            )
            sys.exit(1)
        if run_needs_live_reconcile(parent_data):
            parent_data = reconcile_run(parent_run_dir, parent_data, live_clusters)
        p_cids = get_all_cluster_ids(parent_data)
        parent_cid = p_cids[-1] if p_cids else None
        parent_status = str(
            parent_data.get("_live_badge") or parent_data.get("status", "UNKNOWN")
        )
        is_parent_active = parent_status.startswith(
            ("QUEUED", "RUNNING", "HELD", "WAITING")
        ) or (
            live_clusters is not None
            and parent_cid is not None
            and parent_cid in live_clusters
        )
        if not is_parent_active:
            if parent_status == "COMPLETED":
                console.print(
                    f"[dim]Parent run [bold]{parent_run_dir.name}[/bold] has already completed successfully. "
                    f"Submitting directly without DAGMan delay...[/dim]"
                )
                return parent_run_dir, parent_cid, False, parent_run_dir.name
            if after_ok:
                console.print(
                    f"[bold red]✘ Cannot submit:[/bold red] Parent run '{parent_run_dir.name}' has already finished "
                    f"with status '{parent_status}'. Aborting because --after-ok was specified."
                )
                sys.exit(1)
            console.print(
                f"[dim]Parent run [bold]{parent_run_dir.name}[/bold] has already finished ({parent_status}). "
                f"Proceeding with direct submission...[/dim]"
            )
            return parent_run_dir, parent_cid, False, parent_run_dir.name
        return parent_run_dir, parent_cid, True, parent_run_dir.name

    if str(after_arg).strip().isdigit():
        raw_cid = int(str(after_arg).strip())
        if dry_run or (live_clusters is not None and raw_cid in live_clusters):
            if not dry_run:
                console.print(
                    f"[dim]Parent job [bold]#{raw_cid}[/bold] detected as active raw HTCondor cluster in condor_q.[/dim]"
                )
            return None, raw_cid, True, str(raw_cid)
        console.print(
            f"[bold red]✘ Error:[/bold red] Cluster ID #{raw_cid} is not active in HTCondor (condor_q) "
            f"and does not match any local run in {runs_root}."
        )
        sys.exit(1)

    console.print(
        f"[bold red]✘ Error:[/bold red] Parent run '{after_arg}' could not be resolved."
    )
    sys.exit(1)


def _resolve_retry_dependency(
    runs_root: Path,
    run_dir: Path,
    data: Dict[str, Any],
    after_arg: Optional[str],
    after_ok: bool,
    live_clusters: Optional[Dict[int, Dict[str, Any]]],
) -> Tuple[Optional[Path], Optional[int], bool, bool, Optional[str]]:
    """
    Resolves `(parent_run_dir, parent_cid, submit_via_dag, require_ok, resolved_after_id)`
    for `runs retry` when `--after` is passed or when the target run's original parent is currently active.
    """
    if after_arg:
        p_dir, p_cid, via_dag, res_id = resolve_after_parent(
            runs_root,
            after_arg,
            after_ok,
            live_clusters,
            self_run_dir=run_dir,
        )
        return p_dir, p_cid, via_dag, after_ok, res_id

    dep = data.get("dependency")
    if isinstance(dep, dict):
        orig_p_dir: Optional[Path] = None
        if dep.get("parent_run_dir"):
            cand = Path(str(dep["parent_run_dir"]))
            if cand.is_dir():
                orig_p_dir = cand
        if orig_p_dir is None and dep.get("parent_run_id"):
            cand = runs_root / str(dep["parent_run_id"])
            if cand.is_dir():
                orig_p_dir = cand

        if orig_p_dir is not None and orig_p_dir != run_dir:
            orig_p_data = load_run_json(orig_p_dir)
            if run_needs_live_reconcile(orig_p_data):
                orig_p_data = reconcile_run(orig_p_dir, orig_p_data, live_clusters)
            orig_p_cids = get_all_cluster_ids(orig_p_data)
            orig_p_cid = orig_p_cids[-1] if orig_p_cids else dep.get("parent_cluster_id")
            orig_p_st = str(
                orig_p_data.get("_live_badge") or orig_p_data.get("status", "UNKNOWN")
            )
            is_orig_active = orig_p_st.startswith(
                ("QUEUED", "RUNNING", "HELD", "WAITING")
            ) or (
                live_clusters is not None
                and orig_p_cid is not None
                and int(orig_p_cid) in live_clusters
            )
            if is_orig_active:
                req_ok = bool(dep.get("require_ok", False) or after_ok)
                console.print(
                    f"[dim cyan]ℹ Parent run {_format_run_short(orig_p_dir, orig_p_data)} is currently active "
                    f"({style_status(orig_p_st)}) — queueing retry to wait after #{orig_p_cid}.[/dim cyan]"
                )
                return (
                    orig_p_dir,
                    int(orig_p_cid) if orig_p_cid is not None else None,
                    True,
                    req_ok,
                    orig_p_dir.name,
                )

    return None, None, False, after_ok, None


def _retry_single_run_in_place(
    run_dir: Path,
    data: Dict[str, Any],
    targets: List[Tuple[int, str]],
    *,
    update_code: bool,
    submit_via_dag: bool = False,
    parent_run_dir: Optional[Path] = None,
    parent_cid: Optional[int] = None,
    require_ok: bool = False,
) -> Optional[int]:
    """Retries a single run capsule in-place (directly via condor_submit or chained via DAGMan)."""
    repro_dir = run_dir / "repro"
    rj_path = run_dir / "run.json"
    req = data.get("resources_requested", {})
    sweep_args: List[str] = list(req.get("sweep_arguments") or [""])
    is_sweep = len(sweep_args) > 1

    if update_code:
        with status_spinner(f"Refreshing code snapshot from working tree ({run_dir.name})..."):
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
        telem_f = repro_dir / f"telemetry_{pid}.csv"
        if telem_f.is_file():
            shutil.move(str(telem_f), str(repro_dir / f"telemetry_{pid}.csv.{backup_tag}"))

    condor_log = repro_dir / "condor.log"
    if condor_log.is_file():
        try:
            shutil.move(str(condor_log), str(repro_dir / f"condor.log.{backup_tag}"))
        except OSError:
            pass
    for stale_dag_f in repro_dir.glob("job.dag.*"):
        try:
            stale_dag_f.unlink()
        except OSError:
            pass

    retry_sub_content = generate_retry_sub_file(
        orig_sub_path=repro_dir / "job.sub",
        run_dir=run_dir,
        repro_dir=repro_dir,
        script_stem=script_stem,
        is_sweep=is_sweep,
        targets=targets,
        retry_epoch=retry_epoch,
        run_data=data,
    )
    retry_sub_path = repro_dir / "job_retry.sub"
    write_text_lf(retry_sub_path, retry_sub_content)

    if submit_via_dag:
        dag_path, _ = materialize_dag_files(
            run_dir=run_dir,
            repro_dir=repro_dir,
            sub_path=retry_sub_path,
            parent_dir=parent_run_dir,
            parent_cluster_id=parent_cid,
            require_ok=require_ok,
            condor_submit_exe=shutil.which("condor_submit"),
        )

        with status_spinner(f"Submitting chained retry to HTCondor DAGMan ({run_dir.name})..."):
            dag_res = subprocess.run(
                ["condor_submit_dag", "-f", "-no_submit", str(dag_path)],
                cwd=run_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if dag_res.returncode != 0:
                console.print(
                    f"[bold red]✘ condor_submit_dag failed:[/bold red]\n{(dag_res.stderr or dag_res.stdout).strip()}"
                )
                sys.exit(1)

            condor_sub_path = Path(f"{dag_path}.condor.sub")
            if condor_sub_path.is_file():
                sub_txt = condor_sub_path.read_text(encoding="utf-8", errors="replace")
                write_text_lf(condor_sub_path, sanitize_dagman_condor_sub(sub_txt))

            res = subprocess.run(
                ["condor_submit", str(condor_sub_path)],
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

        def _record_dag_retry(d: Dict[str, Any]) -> None:
            d["status"] = "WAITING"
            d.pop("cancelled_reason", None)
            retried_ids = {pid for pid, _ in targets}
            remaining = [
                x for x in get_execution_records(d) if x.get("process_id") not in retried_ids
            ]
            set_execution_records(d, remaining)
            sub_block = d.setdefault("submission", {})
            history = sub_block.setdefault("retry_clusters", [])
            old_cid = sub_block.get("cluster_id")
            if old_cid is not None and old_cid not in history:
                history.append(old_cid)
            sub_block["cluster_id"] = new_cid
            sub_block["condor_submit_output"] = condor_out
            dep_block = d.setdefault("dependency", {})
            dep_block["parent_run_id"] = parent_run_dir.name if parent_run_dir else None
            dep_block["parent_run_dir"] = (
                str(parent_run_dir.resolve()) if parent_run_dir else None
            )
            dep_block["parent_cluster_id"] = parent_cid
            dep_block["require_ok"] = require_ok
            dep_block["dagman_cluster_id"] = new_cid
            dep_block["job_cluster_id"] = None

        locked_mutate_run_json(rj_path, repro_dir, _record_dag_retry)
        parent_label = (
            f"{parent_run_dir.name} (#{parent_cid})"
            if (parent_run_dir and parent_cid)
            else (parent_run_dir.name if parent_run_dir else f"#{parent_cid}")
        )
        ok_badge = " [bold yellow][exit 0 required][/bold yellow]" if require_ok else ""
        console.print(
            Panel(
                f"[bold blue]⏳ {condor_out}[/bold blue]\n"
                f"  • Waiting After     : [bold white]{parent_label}[/bold white]{ok_badge}\n"
                f"  • Retried Processes : [cyan]{[pid for pid, _ in targets]}[/cyan]\n"
                f"  • Code Refreshed    : [cyan]{'Yes' if update_code else 'No (Frozen snapshot)'}[/cyan]\n"
                f"  • Archived Old Logs : [dim]repro/*.{backup_tag}[/dim]",
                title=f"[bold blue]🔗 Chained In-Place Recovery Queued ({run_dir.name})[/bold blue]",
                border_style="blue",
                expand=False,
            )
        )
        return new_cid

    with status_spinner(f"Submitting retry to HTCondor queue ({run_dir.name})..."):
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
        d.pop("cancelled_reason", None)
        retried_ids = {pid for pid, _ in targets}
        remaining = [
            x for x in get_execution_records(d) if x.get("process_id") not in retried_ids
        ]
        set_execution_records(d, remaining)
        sub_block = d.setdefault("submission", {})
        history = sub_block.setdefault("retry_clusters", [])
        old_cid = sub_block.get("cluster_id")
        if old_cid is not None and old_cid not in history:
            history.append(old_cid)
        sub_block["cluster_id"] = new_cid
        if isinstance(d.get("dependency"), dict):
            d["dependency"]["job_cluster_id"] = new_cid

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
    return new_cid


def _build_send_cmd_for_run(
    send_bin: str,
    run_dir: Path,
    data: Dict[str, Any],
    targets: List[Tuple[int, str]],
    *,
    update_code: bool,
    after: Optional[str] = None,
    after_ok: bool = False,
    yes: bool = False,
) -> Tuple[List[str], Path]:
    req = data.get("resources_requested", {})
    sub_meta = data.get("submission", {})
    cwd = Path(sub_meta.get("working_directory", Path.cwd()))
    sweep_args: List[str] = list(req.get("sweep_arguments") or [""])
    is_sweep = len(sweep_args) > 1
    cmd = build_send_command_tokens(
        send_bin=send_bin,
        target=str(sub_meta.get("target")),
        arch=str(req.get("architecture")),
        cpus=req.get("cpus"),
        mem_mb=req.get("memory_mb"),
        gpus=req.get("gpus"),
        tag=str(sub_meta["tag"]) if sub_meta.get("tag") else None,
        no_venv=bool(sub_meta.get("no_venv")),
        venv=str(sub_meta["venv"]) if sub_meta.get("venv") else None,
        nodes_include=list(req.get("node_include") or []),
        nodes_exclude=list(req.get("node_exclude") or []),
        project_code=str(req["project_code"]) if req.get("project_code") else None,
        scratch_files=[str(sf) for sf in (req.get("scratch_files") or [])],
        env_items=[f"{ek}={ev}" for ek, ev in (req.get("env") or {}).items()],
        sweep_specs=[arg_line for _, arg_line in targets] if is_sweep else None,
        args_flag_val=targets[0][1] if (not is_sweep and targets[0][1]) else None,
        after=after,
        after_ok=after_ok,
        from_run=run_dir.name if not update_code else None,
        max_parallel=req.get("max_parallel"),
        yes=yes,
    )
    return cmd, cwd


def cmd_retry(args: argparse.Namespace, runs_root: Path) -> None:
    run_dir = resolve_run_dir(
        runs_root, args.run, prompt_title="Select run to retry/continue:"
    )
    with status_spinner("Checking HTCondor queue & reconciling run state..."):
        live_clusters = query_live_condor_clusters()
        data = reconcile_run(run_dir, load_run_json(run_dir), live_clusters)
    status = str(data.get("_live_badge") or data.get("status", "UNKNOWN"))
    if status.startswith("WAITING"):
        console.print(
            f"[yellow]⚠ Run '{run_dir.name}' is currently {style_status(status)}. Cannot retry a waiting run. Use `runs stop {run_dir.name}` to cancel it.[/yellow]"
        )
        return

    after_arg = getattr(args, "after", None)
    after_ok_flag = bool(getattr(args, "after_ok", False))
    chain_flag = bool(getattr(args, "chain", False))
    single_flag = bool(getattr(args, "single", False))

    parent_run_dir, parent_cid, submit_via_dag, require_ok, resolved_after_id = (
        _resolve_retry_dependency(
            runs_root, run_dir, data, after_arg, after_ok_flag, live_clusters
        )
    )

    req = data.get("resources_requested", {})
    sweep_args: List[str] = list(req.get("sweep_arguments") or [""])
    is_sweep = len(sweep_args) > 1

    if args.all:
        targets = list(enumerate(sweep_args))
    else:
        targets = get_failed_or_missing_procs(data)
        if not targets:
            if args.new_run:
                targets = list(enumerate(sweep_args))
            elif not args.in_place and sys.stdin.isatty():
                import questionary

                if questionary.confirm(
                    "All processes in this run completed with exit 0. Re-run all processes?",
                    default=True,
                    style=get_prompt_style(),
                ).ask():
                    targets = list(enumerate(sweep_args))
                else:
                    return
            else:
                console.print(
                    "[yellow]All processes in this run completed with exit_code 0. Use `--all` or `--new-run` to force re-run.[/yellow]"
                )
                return

    root_cids = get_all_cluster_ids(data)
    downstream = (
        []
        if single_flag
        else _find_downstream_retry_chain(
            runs_root, run_dir, root_cids, live_clusters, include_completed=bool(args.all)
        )
    )
    retry_downstream = bool(downstream)
    if downstream:
        console.print(
            f"[yellow]🔗 Found {len(downstream)} downstream chained run(s) depending on {_format_run_short(run_dir, data)}:[/yellow]"
        )
        for dep_dir, dep_data in downstream:
            dep_st = str(dep_data.get("_live_badge") or dep_data.get("status", "UNKNOWN"))
            console.print(
                f"  • {_format_run_short(dep_dir, dep_data)} ({style_status(dep_st)})"
            )

        if not chain_flag and not args.yes and sys.stdin.isatty():
            import questionary

            c_choice = questionary.select(
                "How would you like to retry this chain?",
                choices=[
                    questionary.Choice(
                        f"1. Retry entire chain ({1 + len(downstream)} runs)",
                        value="chain",
                    ),
                    questionary.Choice(
                        f"2. Retry only this run ({run_dir.name})",
                        value="single",
                    ),
                    questionary.Choice("3. Cancel", value="cancel"),
                ],
                style=get_prompt_style(),
            ).ask()
            if not c_choice or c_choice == "cancel":
                return
            retry_downstream = c_choice == "chain"

    mode = "in_place" if args.in_place else ("new_run" if args.new_run else None)
    update_code = args.update_code

    if mode is None and sys.stdin.isatty():
        import questionary

        choice = questionary.select(
            f"Found {len(targets)}/{len(sweep_args)} process(es) to retry. Choose recovery mode:",
            choices=[
                questionary.Choice(
                    "1. New Run with Frozen Snapshot (Re-runs exact historical code in a new run folder)",
                    value=("new_run", False),
                ),
                questionary.Choice(
                    "2. New Run with Current Working Tree (Snapshots current workspace files into a new run folder)",
                    value=("new_run", True),
                ),
                questionary.Choice(
                    "3. In-Place Continuation with Frozen Snapshot (Re-runs inside same $HERMES_RUN_DIR)",
                    value=("in_place", False),
                ),
                questionary.Choice(
                    "4. In-Place Continuation + Update Code Snapshot (Syncs current workspace into same $HERMES_RUN_DIR)",
                    value=("in_place", True),
                ),
            ],
            style=get_prompt_style(),
        ).ask()
        if not choice:
            sys.exit(0)
        mode, update_code = choice
    elif mode is None:
        mode = "in_place"

    if mode == "new_run":
        send_bin = shutil.which("send") or str(Path(sys.executable).parent / "send")
        cmd, cwd = _build_send_cmd_for_run(
            send_bin,
            run_dir,
            data,
            targets,
            update_code=update_code,
            after=resolved_after_id,
            after_ok=require_ok if resolved_after_id else False,
            yes=bool(args.yes or retry_downstream),
        )
        console.print(
            f"[bold cyan]▶ Relaunching via send:[/bold cyan] {' '.join(shlex.quote(c) for c in cmd)}"
        )
        if not retry_downstream:
            os.chdir(cwd)
            os.execv(send_bin, cmd)
            return

        res_first = subprocess.run(cmd, cwd=cwd)
        if res_first.returncode != 0:
            sys.exit(res_first.returncode)
        latest_after_first = resolve_latest_symlink(runs_root) or (
            list_run_dirs(runs_root)[0] if list_run_dirs(runs_root) else None
        )
        prev_new_name = latest_after_first.name if latest_after_first else run_dir.name

        for dep_dir, dep_data in downstream:
            dep_req = dep_data.get("resources_requested") or {}
            dep_sweep = list(dep_req.get("sweep_arguments") or [""])
            dep_targets = (
                list(enumerate(dep_sweep))
                if args.all
                else (get_failed_or_missing_procs(dep_data) or list(enumerate(dep_sweep)))
            )
            dep_ok = bool(
                (dep_data.get("dependency") or {}).get("require_ok", False) or after_ok_flag
            )
            dep_cmd, dep_cwd = _build_send_cmd_for_run(
                send_bin,
                dep_dir,
                dep_data,
                dep_targets,
                update_code=update_code,
                after=prev_new_name,
                after_ok=dep_ok,
                yes=True,
            )
            console.print(
                f"[bold cyan]▶ Relaunching downstream via send:[/bold cyan] {' '.join(shlex.quote(c) for c in dep_cmd)}"
            )
            res_dep = subprocess.run(dep_cmd, cwd=dep_cwd)
            if res_dep.returncode != 0:
                sys.exit(res_dep.returncode)
            latest_now = resolve_latest_symlink(runs_root) or (
                list_run_dirs(runs_root)[0] if list_run_dirs(runs_root) else None
            )
            if latest_now:
                prev_new_name = latest_now.name
        return

    if live_clusters is not None and not args.force:
        active_cids = [c for c in get_all_cluster_ids(data) if c in live_clusters]
        if active_cids:
            console.print(
                f"[bold red]✘ Run still has active jobs in HTCondor cluster(s) {active_cids}.[/bold red]\n"
                f"  Wait for them to finish, run `runs stop {run_dir.name}` first, or pass `--force`."
            )
            sys.exit(1)

    if retry_downstream:
        total_chain = 1 + len(downstream)
        with step_progress(f"Retrying chain ({total_chain} runs)...", total_chain) as advance_chain:
            first_cid = _retry_single_run_in_place(
                run_dir,
                data,
                targets,
                update_code=update_code,
                submit_via_dag=submit_via_dag,
                parent_run_dir=parent_run_dir,
                parent_cid=parent_cid,
                require_ok=require_ok,
            )
            advance_chain(f"Retried step 1/{total_chain} ({run_dir.name})...")
            prev_dir = run_dir
            prev_cid = first_cid
            for step_idx, (dep_dir, dep_data) in enumerate(downstream, start=2):
                dep_req = dep_data.get("resources_requested") or {}
                dep_sweep = list(dep_req.get("sweep_arguments") or [""])
                dep_targets = (
                    list(enumerate(dep_sweep))
                    if args.all
                    else (get_failed_or_missing_procs(dep_data) or list(enumerate(dep_sweep)))
                )
                dep_ok = bool(
                    (dep_data.get("dependency") or {}).get("require_ok", False) or after_ok_flag
                )
                prev_cid = _retry_single_run_in_place(
                    dep_dir,
                    dep_data,
                    dep_targets,
                    update_code=update_code,
                    submit_via_dag=True,
                    parent_run_dir=prev_dir,
                    parent_cid=prev_cid,
                    require_ok=dep_ok,
                )
                prev_dir = dep_dir
                advance_chain(f"Retried step {step_idx}/{total_chain} ({dep_dir.name})...")
    else:
        _retry_single_run_in_place(
            run_dir,
            data,
            targets,
            update_code=update_code,
            submit_via_dag=submit_via_dag,
            parent_run_dir=parent_run_dir,
            parent_cid=parent_cid,
            require_ok=require_ok,
        )


# ---------------------------------------------------------------------------
# Subcommand: `runs clean`
# ---------------------------------------------------------------------------
def cmd_clean(args: argparse.Namespace, runs_root: Path) -> None:
    dirs = list_run_dirs(runs_root)
    if not dirs:
        console.print("[dim]No runs to clean.[/dim]")
        return

    has_filters = bool(args.failed or args.aborted or args.dry or args.older_than)
    if not has_filters and not sys.stdin.isatty():
        console.print(
            "[yellow]Specify --failed, --aborted, --dry, or --older-than (e.g. 7d) in non-interactive mode.[/yellow]"
        )
        return

    older_delta: Optional[datetime.timedelta] = None
    if args.older_than:
        m = re.match(r"^(\d+)([dhm])$", args.older_than.strip().lower())
        if m:
            val, unit = int(m.group(1)), m.group(2)
            older_delta = (
                datetime.timedelta(days=val)
                if unit == "d"
                else (
                    datetime.timedelta(hours=val)
                    if unit == "h"
                    else datetime.timedelta(minutes=val)
                )
            )

    live_clusters_queried = False
    live_clusters: Optional[Dict[int, Dict[str, int]]] = None
    now = datetime.datetime.now()
    inactive_runs: List[Tuple[Path, str, str, str, str]] = []
    candidates: List[Tuple[Path, str, float]] = []
    skipped_pinned = 0

    for d in dirs:
        raw_data = load_run_json(d)
        if run_needs_live_reconcile(raw_data):
            if not live_clusters_queried:
                live_clusters = query_live_condor_clusters()
                live_clusters_queried = True
            data = reconcile_run(d, raw_data, live_clusters)
        else:
            data = raw_data

        st = str(data.get("_live_badge") or data.get("status", "UNKNOWN"))
        if st.startswith(("RUNNING", "QUEUED", "HELD", "WAITING")):
            continue
        if bool(data.get("pinned", False)):
            skipped_pinned += 1
            continue

        if not has_filters:
            plain_label, _, short_date = _extract_run_display_parts(d, data)
            arch = str(data.get("resources_requested", {}).get("architecture", "—"))
            _, dur_str, _ = summarize_metrics(data)
            inactive_runs.append((d, st, plain_label, arch, f"{dur_str} │ {short_date}"))
            continue

        match = False
        if args.failed and (
            st in ("FAILED", "SUBMIT_FAILED", "VANISHED", "CANCELLED")
            or st.startswith(("FAILED", "CANCELLED"))
        ):
            match = True
        if args.aborted and (
            st in ("ABORTED", "INTERRUPTED", "CANCELLED")
            or st.startswith(("INTERRUPTED", "CANCELLED", "ABORTED"))
        ):
            match = True
        if args.dry and st == "DRY_RUN":
            match = True
        if older_delta is not None:
            mtime = datetime.datetime.fromtimestamp(d.stat().st_mtime)
            if now - mtime > older_delta:
                match = True

        if match:
            candidates.append((d, st, dir_size_mb(d)))

    if not has_filters:
        if not inactive_runs:
            console.print("[green]No inactive runs eligible for deletion.[/green]")
            if skipped_pinned > 0:
                console.print(f"[dim]ℹ Skipped {skipped_pinned} pinned run(s).[/dim]")
            return
        import questionary

        name_w = max((len(x[2]) for x in inactive_runs), default=14)
        arch_w = max((len(x[3]) for x in inactive_runs), default=6)
        st_w = max((len(x[1]) for x in inactive_runs), default=8)
        choices = [
            questionary.Choice(
                f"{label:<{name_w}} │ {arch:<{arch_w}} │ {st:<{st_w}} │ {meta_tail}",
                value=(d, st),
            )
            for d, st, label, arch, meta_tail in inactive_runs
        ]
        selected = questionary.checkbox(
            "Select runs to permanently delete (Space to toggle, Enter to confirm):",
            choices=choices,
            style=get_prompt_style(),
        ).ask()
        if not selected:
            return
        candidates = [(d, st, dir_size_mb(d)) for d, st in selected]

    if not candidates:
        console.print("[green]No matching runs found to clean.[/green]")
        if skipped_pinned > 0:
            console.print(f"[dim]ℹ Skipped {skipped_pinned} pinned run(s).[/dim]")
        return

    total_mb = sum(x[2] for x in candidates)
    for d, st, mb in candidates:
        console.print(f"  • {_format_run_short(d)} ({style_status(st)} — {mb:.2f} MB)")

    if not args.yes:
        import questionary

        if not questionary.confirm(
            f"Delete {len(candidates)} run(s) and free {total_mb:.2f} MB?",
            default=False,
            style=get_prompt_style(),
        ).ask():
            return

    with step_progress(f"Deleting {len(candidates)} run(s)...", len(candidates)) as advance_clean:
        for d, _, _ in candidates:
            shutil.rmtree(d, ignore_errors=True)
            advance_clean(f"Deleting {d.name}...")
    update_latest_symlink(runs_root)
    console.print(
        f"[bold green]✔ Deleted {len(candidates)} run(s) ({total_mb:.2f} MB freed).[/bold green]"
    )
    if skipped_pinned > 0:
        console.print(f"[dim]ℹ Skipped {skipped_pinned} pinned run(s).[/dim]")


# ---------------------------------------------------------------------------
# Subcommand: `runs pin` & `runs unpin`
# ---------------------------------------------------------------------------


def cmd_pin(args: argparse.Namespace, runs_root: Path, unpin: bool = False) -> None:
    """Pin or unpin a run to protect it from `runs clean` and attach an optional note."""
    run_dir = resolve_run_dir(
        runs_root,
        getattr(args, "run", None),
        prompt_title="Select run to unpin:" if unpin else "Select run to pin:",
    )
    note_arg = getattr(args, "note", None)

    def _mutate(d: Dict[str, Any]) -> None:
        if unpin:
            d["pinned"] = False
            if note_arg == "":
                d.pop("note", None)
        else:
            if getattr(args, "run", None) is not None or not bool(d.get("pinned", False)) or note_arg is not None:
                d["pinned"] = True
            if note_arg is not None:
                d["note"] = note_arg.strip() or None

    updated_data = locked_mutate_run_json(run_dir / "run.json", run_dir / "repro", _mutate)
    short_run = _format_run_short(run_dir, updated_data)
    if unpin:
        console.print(f"[yellow]Unpinned[/yellow] {short_run} — eligible for 'runs clean' again")
    else:
        console.print(f"[bold green]📌 Pinned[/bold green] {short_run} — protected from 'runs clean'")
        if updated_data.get("note"):
            console.print(f'   [dim]Note: "{updated_data["note"]}"[/dim]')


# ---------------------------------------------------------------------------
# Subcommand: `runs diff`
# ---------------------------------------------------------------------------
def _format_env_summary(data: Dict[str, Any]) -> str:
    env_map = data.get("resources_requested", {}).get("env")
    if isinstance(env_map, dict) and env_map:
        return ", ".join(f"{k}={v}" for k, v in env_map.items())
    return "—"


def cmd_diff(args: argparse.Namespace, runs_root: Path) -> None:
    dir_a = resolve_run_dir(runs_root, args.run_a, prompt_title="Select first run (A):")
    dir_b = resolve_run_dir(runs_root, args.run_b, prompt_title="Select second run (B):")
    raw_a, raw_b = load_run_json(dir_a), load_run_json(dir_b)
    if run_needs_live_reconcile(raw_a) or run_needs_live_reconcile(raw_b):
        live_clusters = query_live_condor_clusters()
        da = reconcile_run(dir_a, raw_a, live_clusters) if run_needs_live_reconcile(raw_a) else raw_a
        db = reconcile_run(dir_b, raw_b, live_clusters) if run_needs_live_reconcile(raw_b) else raw_b
    else:
        da, db = raw_a, raw_b

    st_a = str(da.get("_live_badge") or da.get("status", "UNKNOWN"))
    st_b = str(db.get("_live_badge") or db.get("status", "UNKNOWN"))

    t = Table(
        title="⚖️  Run Provenance Comparison",
        title_style="bold cyan",
        box=box.ROUNDED,
        header_style="bold cyan",
    )
    plain_a, _, date_a = _extract_run_display_parts(dir_a, da)
    plain_b, _, date_b = _extract_run_display_parts(dir_b, db)
    t.add_column("Field", style="bold cyan")
    t.add_column(f"Run A: {plain_a} ({date_a})", style="white")
    t.add_column(f"Run B: {plain_b} ({date_b})", style="white")

    t.add_row("Status", style_status(st_a), style_status(st_b))
    fields = [
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
            "Env Exports",
            _format_env_summary(da),
            _format_env_summary(db),
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
    if da.get("pinned") or db.get("pinned") or da.get("note") or db.get("note"):
        def _format_pin_note(d: Dict[str, Any]) -> str:
            pin_str = "📌 Pinned" if d.get("pinned") else "No"
            note_val = d.get("note")
            return f"{pin_str} ({note_val})" if note_val else pin_str

        fields.append(("Pinned / Note", _format_pin_note(da), _format_pin_note(db)))
    for label, va, vb in fields:
        style_b = "bold yellow" if va != vb else "white"
        t.add_row(label, str(va), f"[{style_b}]{vb}[/{style_b}]")
    console.print(t)

    code_a, code_b = dir_a / "repro" / "code", dir_b / "repro" / "code"
    if code_a.is_dir() and code_b.is_dir():
        diff_text = _diff_directories(code_a, code_b).strip()
        if diff_text:
            from rich.syntax import Syntax

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
# Subcommand: `runs nodes` (`cluster`, `pools`)
# ---------------------------------------------------------------------------
def _style_node_state(state: str) -> str:
    styles = {
        "FREE": "bold green",
        "MIXED": "bold cyan",
        "BUSY": "bold yellow",
        "DRAINING": "bold magenta",
        "OFFLINE": "bold red",
    }
    col = styles.get(state, "white")
    return f"[{col}]{state}[/{col}]"


def _format_free_total(free: int, total: int, unit: str = "") -> str:
    if total <= 0:
        return "[dim]—[/dim]"
    sfx = f" {unit}" if unit else ""
    if free > 0:
        return f"[bold green]{free}[/bold green]/{total}{sfx}"
    return f"[dim]0/{total}{sfx}[/dim]"


def _format_users_cell(users_map: Dict[str, int], current_user: str) -> str:
    if not users_map:
        return "[dim]—[/dim]"
    parts: List[str] = []
    for u, cnt in sorted(users_map.items(), key=lambda kv: (-kv[1], kv[0])):
        label = f"{u}({cnt})" if cnt > 1 else u
        if current_user and u.lower() == current_user.lower():
            parts.append(f"[bold cyan]{label}[/bold cyan]")
        else:
            parts.append(label)
    return ", ".join(parts)


def _resolve_target_user_filter(args: argparse.Namespace, current_user: str) -> str:
    if getattr(args, "mine", False):
        return current_user
    raw_u = getattr(args, "user", None)
    if raw_u == "__ME__":
        return current_user
    return str(raw_u or "").strip()


def _node_matches_user(node: Dict[str, Any], target_user: str) -> bool:
    if not target_user:
        return True
    needle = target_user.lower()
    users_map = node.get("users", {}) or {}
    return any(needle in u.lower() for u in users_map)


def _format_jobs_cell(count: int) -> str:
    if count <= 0:
        return "[dim]0[/dim]"
    return f"[bold cyan]{count}[/bold cyan]"


def _build_local_cluster_map(runs_root: Optional[Path]) -> Dict[str, str]:
    if runs_root is None or not runs_root.is_dir():
        return {}
    mapping: Dict[str, str] = {}
    for rd in list_run_dirs(runs_root)[:60]:
        data = load_run_json(rd)
        if not data:
            continue
        sub = data.get("submission", {})
        tag_str = str(sub.get("tag") or "").strip()
        target_str = str(sub.get("target") or "").strip()
        if tag_str:
            label = tag_str
        elif target_str:
            label = Path(target_str).stem
        else:
            plain_label, _, _ = _extract_run_display_parts(rd, data)
            label = plain_label or rd.name
        for cid in get_all_cluster_ids(data):
            mapping[str(cid)] = label
    return mapping


def _build_nodes_view(
    args: argparse.Namespace,
    nodes: List[Dict[str, Any]],
    cluster_map: Optional[Dict[str, str]] = None,
) -> Tuple[Any, Tuple[Tuple[str, ...], ...]]:
    from rich.console import Group

    current_user = get_username()
    arch_filter = (getattr(args, "arch", None) or "").strip().upper()
    node_filter = (getattr(args, "node", None) or "").strip().lower()
    user_filter = _resolve_target_user_filter(args, current_user)
    free_only = bool(getattr(args, "free", False))
    busy_only = bool(getattr(args, "busy", False))
    detail_mode = bool(getattr(args, "detail", False))

    filtered = [
        n
        for n in nodes
        if (not arch_filter or arch_filter in str(n["arch"]).upper())
        and (
            not node_filter
            or node_filter in str(n["host"]).lower()
            or node_filter in str(n["machine"]).lower()
        )
        and (not free_only or n["state"] in ("FREE", "MIXED"))
        and (not busy_only or n["state"] in ("BUSY", "MIXED") or int(n.get("active_jobs", 0)) > 0)
        and _node_matches_user(n, user_filter)
    ]

    pool_source = (
        filtered
        if (node_filter or user_filter or busy_only)
        else [n for n in nodes if not arch_filter or arch_filter in str(n["arch"]).upper()]
    )
    pool_map = summarize_condor_pools(pool_source)

    pool_table = Table(
        title="🖥️  Hermes 2 Architecture Pools (Free / Total)",
        title_style="bold cyan",
        box=box.ROUNDED,
        header_style="bold cyan",
    )
    pool_table.add_column("Pool", style="bold magenta", no_wrap=True)
    pool_table.add_column("Status", no_wrap=True)
    pool_table.add_column("Nodes", justify="right", no_wrap=True)
    pool_table.add_column("GPUs", justify="right", no_wrap=True)
    pool_table.add_column("CPUs", justify="right", no_wrap=True)
    pool_table.add_column("RAM", justify="right", no_wrap=True)
    pool_table.add_column("Jobs", justify="right", no_wrap=True)
    pool_table.add_column("Users")

    sig_rows: List[Tuple[str, ...]] = []
    for arch, p in pool_map.items():
        if p["gpus_total"] > 0:
            avail_badge = (
                f"[bold green]● {p['gpus_free']} GPU free[/bold green]"
                if p["gpus_free"] > 0
                else "[bold yellow]○ All GPUs busy[/bold yellow]"
            )
        else:
            avail_badge = (
                f"[bold green]● {p['cpus_free']} CPU free[/bold green]"
                if p["cpus_free"] > 0
                else "[bold yellow]○ All CPUs busy[/bold yellow]"
            )
        p_row = (
            arch,
            avail_badge,
            _format_free_total(p["nodes_avail"], p["nodes_total"]),
            _format_free_total(p["gpus_free"], p["gpus_total"]),
            _format_free_total(p["cpus_free"], p["cpus_total"]),
            _format_free_total(
                round(p["mem_free_mb"] / 1024),
                max(1, round(p["mem_total_mb"] / 1024)),
                "G",
            )
            if p["mem_total_mb"] > 0
            else "[dim]—[/dim]",
            _format_jobs_cell(int(p["active_jobs"])),
            _format_users_cell(p.get("users", {}), current_user),
        )
        pool_table.add_row(*p_row)
        sig_rows.append(p_row)

    node_table = Table(
        title="🖧  Hermes 2 Worker Nodes (Free / Total)",
        title_style="bold cyan",
        box=box.ROUNDED,
        header_style="bold cyan",
    )
    node_table.add_column("Node", style="bold white", no_wrap=True)
    node_table.add_column("Pool", style="magenta", no_wrap=True)
    node_table.add_column("State", no_wrap=True)
    node_table.add_column("GPUs", justify="right", no_wrap=True)
    node_table.add_column("CPUs", justify="right", no_wrap=True)
    node_table.add_column("RAM", justify="right", no_wrap=True)
    node_table.add_column("Jobs", justify="right", no_wrap=True)
    node_table.add_column("Users", no_wrap=True)
    if detail_mode:
        node_table.add_column("Load", justify="right", no_wrap=True)
        node_table.add_column("GPU / Disk", style="dim")

    prev_arch: Optional[str] = None
    for n in filtered:
        cur_arch = str(n["arch"])
        if prev_arch is not None and cur_arch != prev_arch:
            node_table.add_section()
        prev_arch = cur_arch

        mem_tot_gb = round(int(n["mem_total_mb"]) / 1024)
        mem_free_gb = round(int(n["mem_free_mb"]) / 1024)
        base_cols: List[str] = [
            str(n["host"]),
            cur_arch,
            _style_node_state(str(n["state"])),
            _format_free_total(int(n["gpus_free"]), int(n["gpus_total"])),
            _format_free_total(int(n["cpus_free"]), int(n["cpus_total"])),
            _format_free_total(mem_free_gb, max(1, mem_tot_gb), "G")
            if int(n["mem_total_mb"]) > 0
            else "[dim]—[/dim]",
            _format_jobs_cell(int(n["active_jobs"])),
            _format_users_cell(n.get("users", {}), current_user),
        ]
        if detail_mode:
            load_val = n.get("load_avg")
            disk_val = n.get("disk_free_gb")
            gpu_model = str(n.get("gpu_model") or "")
            hw_parts: List[str] = []
            if gpu_model:
                hw_parts.append(gpu_model)
            if isinstance(disk_val, (int, float)) and disk_val > 0:
                hw_parts.append(f"{disk_val:.0f}G free")
            base_cols.extend(
                [
                    f"{load_val:.2f}" if isinstance(load_val, (int, float)) else "[dim]—[/dim]",
                    " · ".join(hw_parts) if hw_parts else "[dim]—[/dim]",
                ]
            )
        n_row = tuple(base_cols)
        node_table.add_row(*n_row)
        sig_rows.append(n_row)

    renderables: List[Any] = [pool_table, node_table]

    if detail_mode or user_filter:
        claims_list: List[Dict[str, Any]] = []
        for n in filtered:
            for c in n.get("claims", []):
                if not user_filter or user_filter.lower() in str(c.get("user", "")).lower():
                    claims_list.append(c)

        if claims_list or detail_mode:
            claims_table = Table(
                title="⚡ Active Slot Claims & Running Jobs",
                title_style="bold cyan",
                box=box.ROUNDED,
                header_style="bold cyan",
            )
            claims_table.add_column("Node", style="bold white", no_wrap=True)
            claims_table.add_column("Pool", style="magenta", no_wrap=True)
            claims_table.add_column("User", no_wrap=True)
            claims_table.add_column("Job", style="cyan")
            claims_table.add_column("GPUs", justify="right", no_wrap=True)
            claims_table.add_column("CPUs", justify="right", no_wrap=True)
            claims_table.add_column("RAM", justify="right", no_wrap=True)
            claims_table.add_column("Runtime", justify="right", style="green", no_wrap=True)

            cmap = cluster_map or {}
            for c in claims_list:
                u_str = str(c.get("user") or "—")
                if current_user and u_str.lower() == current_user.lower():
                    u_cell = f"[bold cyan]{u_str}[/bold cyan]"
                else:
                    u_cell = u_str
                rt_sec = c.get("runtime_seconds")
                rt_str = format_duration(rt_sec) if isinstance(rt_sec, (int, float)) else "[dim]—[/dim]"
                mem_gb = round(int(c.get("mem_mb", 0)) / 1024, 1)
                raw_jid = str(c.get("job_id") or "—")
                cid_key = raw_jid.split(".")[0]
                run_tag = cmap.get(cid_key)
                job_cell = f"{raw_jid} [dim]({run_tag})[/dim]" if run_tag else raw_jid
                c_row = (
                    str(c.get("host", "—")),
                    str(c.get("arch", "—")),
                    u_cell,
                    job_cell,
                    str(c.get("gpus", 0)),
                    str(c.get("cpus", 0)),
                    f"{mem_gb:g}G" if mem_gb > 0 else "[dim]—[/dim]",
                    rt_str,
                )
                claims_table.add_row(*c_row)
                sig_rows.append(c_row)

            renderables.append(claims_table)

    return Group(*renderables), tuple(sig_rows)


def cmd_nodes(args: argparse.Namespace, runs_root: Optional[Path] = None) -> None:
    constraint = getattr(args, "where", None)
    try:
        nodes = query_condor_nodes(constraint=constraint)
    except ValueError as exc:
        console.print(
            f"[bold red]✖ Invalid --where constraint or condor_status error:[/bold red]\n[dim]{exc}[/dim]"
        )
        sys.exit(1)
    if nodes is None:
        console.print(
            "[yellow]⚠ 'condor_status' is not available or returned an error (are you on hermes2?).[/yellow]"
        )
        sys.exit(1)
    if not nodes:
        console.print("[dim]No worker nodes matched the query in condor_status.[/dim]")
        return

    cluster_map = _build_local_cluster_map(runs_root)

    if getattr(args, "watch", False):
        from rich.live import Live

        interval = max(1, int(getattr(args, "interval", 2) or 2))
        initial_view, last_sig = _build_nodes_view(args, nodes, cluster_map=cluster_map)
        current_view = initial_view
        try:
            with Live(
                initial_view,
                console=console,
                screen=True,
                auto_refresh=False,
            ) as live:
                while True:
                    time.sleep(interval)
                    try:
                        fresh_nodes = query_condor_nodes(constraint=constraint)
                    except ValueError:
                        continue
                    if fresh_nodes is None:
                        continue
                    cluster_map = _build_local_cluster_map(runs_root)
                    new_view, new_sig = _build_nodes_view(args, fresh_nodes, cluster_map=cluster_map)
                    if new_sig != last_sig:
                        last_sig = new_sig
                        current_view = new_view
                        live.update(new_view, refresh=True)
        except KeyboardInterrupt:
            console.print(current_view)
            return
    else:
        view, _ = _build_nodes_view(args, nodes, cluster_map=cluster_map)
        console.print(view)


# ---------------------------------------------------------------------------
# Subcommand: `runs why` (`analyze`, `q`) — Queue Match Analysis
# ---------------------------------------------------------------------------
def _render_queue_analysis_panel(
    job_target: str,
    raw_mode: bool = False,
    quiet_if_empty: bool = False,
) -> bool:
    from rich.console import Group

    with status_spinner(f"Analyzing HTCondor slot requirements for Job {job_target}..."):
        analysis = query_condor_better_analyze(job_target)
    if analysis is None:
        if not quiet_if_empty:
            console.print(
                "[yellow]⚠ 'condor_q -better-analyze' is not available or returned no output (are you on hermes2?).[/yellow]"
            )
        return False

    if raw_mode:
        console.print(
            Panel(
                analysis["raw"],
                title=f"[bold cyan]⏳ Raw Queue Match Analysis — Job {analysis.get('job_id') or job_target}[/bold cyan]",
                border_style="cyan",
            )
        )
        return True

    steps = analysis.get("steps") or []
    machines_total = int(analysis.get("machines_total") or 0)
    if not steps and machines_total == 0 and quiet_if_empty:
        return False

    summary = analysis.get("summary") or {}
    verdict = str(analysis.get("verdict") or "")
    v_style = str(analysis.get("verdict_style") or "white")
    sched_note = str(analysis.get("scheduler_note") or "")
    jid_label = str(analysis.get("job_id") or job_target)

    info_tbl = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
    info_tbl.add_column("K", style="bold cyan", width=20, no_wrap=True)
    info_tbl.add_column("V", style="white")
    info_tbl.add_row("Verdict", f"[{v_style}]{verdict}[/{v_style}]")
    if sched_note:
        info_tbl.add_row("Scheduler Note", f"[dim]{sched_note}[/dim]")

    last_fail = str(analysis.get("last_failed_match") or "")
    if last_fail:
        info_tbl.add_row("Last Match Check", last_fail)

    if machines_total > 0:
        able = int(summary.get("able_to_run", 0))
        others = int(summary.get("serving_others", 0))
        yours = int(summary.get("running_your_jobs", 0))
        rej_job = int(summary.get("rejected_by_job", 0))
        rej_node = int(summary.get("rejected_by_node", 0))
        info_tbl.add_row(
            f"Machines ({machines_total} tot)",
            (
                f"[bold green]{able} eligible[/bold green] · "
                f"[cyan]{others} busy (others)[/cyan] · "
                f"[cyan]{yours} busy (yours)[/cyan] · "
                f"[yellow]{rej_job} job-rejected[/yellow] · "
                f"[dim]{rej_node} node-policy[/dim]"
            ),
        )

    renderables: List[Any] = [info_tbl]

    if steps:
        max_slots = max(int(s["slots"]) for s in steps)
        non_zero = [int(s["slots"]) for s in steps if int(s["slots"]) > 0]
        min_nonzero = min(non_zero) if non_zero else 0

        step_tbl = Table(
            title="Requirements Slot Funnel",
            title_style="bold cyan",
            box=box.SIMPLE_HEAVY,
            header_style="bold cyan",
        )
        step_tbl.add_column("Step", style="dim", no_wrap=True)
        step_tbl.add_column("Slots", justify="right", no_wrap=True)
        step_tbl.add_column("Condition", style="white")
        step_tbl.add_column("Requested", style="cyan", no_wrap=True)

        for s in steps:
            cnt = int(s["slots"])
            cond_str = str(s["condition"])
            if cnt == 0:
                cnt_cell = "[bold red]0[/bold red]"
                cond_str = f"[bold red]{cond_str}[/bold red] [red](blocking)[/red]"
            elif cnt == min_nonzero and min_nonzero < max_slots:
                cnt_cell = f"[bold yellow]{cnt}[/bold yellow]"
                cond_str = f"{cond_str} [yellow](tightest)[/yellow]"
            else:
                cnt_cell = f"[green]{cnt}[/green]"
            step_tbl.add_row(
                str(s["step"]),
                cnt_cell,
                cond_str,
                str(s.get("hint") or "—"),
            )
        renderables.append(step_tbl)
    elif analysis.get("raw"):
        renderables.append(f"[dim]{analysis['raw']}[/dim]")

    console.print(
        Panel(
            Group(*renderables),
            title=f"[bold cyan]⏳ Queue Match Analysis — Job {jid_label}[/bold cyan]",
            border_style="cyan",
            expand=False,
        )
    )
    return True


def cmd_why(args: argparse.Namespace, runs_root: Path) -> None:
    target = getattr(args, "run", None)
    proc_idx = int(getattr(args, "proc", 0) or 0)
    raw_mode = bool(getattr(args, "raw", False))

    # Allow passing a raw cluster.proc ID directly (e.g. `runs why 1074510.0`)
    if target and re.match(r"^\d+\.\d+$", str(target).strip()):
        _render_queue_analysis_panel(str(target).strip(), raw_mode=raw_mode)
        return

    # If no target specified, prefer the most recent QUEUED/HELD run, falling back to `latest`
    run_dir: Optional[Path] = None
    if not target:
        live_clusters = query_live_condor_clusters()
        for rd in list_run_dirs(runs_root)[:25]:
            d = load_run_json(rd)
            if not d:
                continue
            if run_needs_live_reconcile(d):
                d = reconcile_run(rd, d, live_clusters)
            st = str(d.get("_live_badge") or d.get("status", ""))
            if st.startswith(("QUEUED", "HELD")):
                run_dir = rd
                break
        if run_dir is None:
            run_dir = resolve_run_dir(runs_root, "latest")
    else:
        # Check if target is a raw cluster ID that doesn't match a local run
        run_dir = resolve_run_dir(runs_root, target, exit_on_fail=False)
        if run_dir is None and str(target).strip().isdigit():
            _render_queue_analysis_panel(f"{str(target).strip()}.{proc_idx}", raw_mode=raw_mode)
            return
        if run_dir is None:
            console.print(f"[red]✖ Could not resolve run '{target}'.[/red]")
            sys.exit(1)

    if not run_dir:
        sys.exit(1)

    data = load_run_json(run_dir)
    if not data:
        console.print(f"[red]✖ Could not read run.json in {run_dir.name}.[/red]")
        sys.exit(1)

    live_clusters = query_live_condor_clusters()
    data = reconcile_run(run_dir, data, live_clusters)
    cids = get_all_cluster_ids(data)
    if not cids:
        console.print(f"[yellow]⚠ Run '{run_dir.name}' has no HTCondor cluster ID.[/yellow]")
        sys.exit(1)

    job_target = f"{cids[-1]}.{proc_idx}"
    cur_status = str(data.get("_live_badge") or data.get("status", "UNKNOWN"))

    if cur_status.startswith("WAITING") or data.get("status") == "WAITING":
        dep = data.get("dependency", {})
        p_id = dep.get("parent_run_id") or "unknown"
        p_cid = dep.get("parent_cluster_id")
        p_dir = dep.get("parent_run_dir")
        r_ok = dep.get("require_ok", False)

        p_status = "UNKNOWN"
        p_data: Optional[Dict[str, Any]] = None
        if p_dir:
            try:
                p_data = load_run_json(Path(p_dir))
                p_data = reconcile_run(Path(p_dir), p_data, live_clusters)
                p_status = str(p_data.get("_live_badge") or p_data.get("status", "UNKNOWN"))
            except Exception:
                pass

        p_cid_str = f" (#{p_cid})" if p_cid else ""
        p_run_disp = _format_run_short(Path(p_dir), p_data) if p_dir else f"[bold]{p_id}[/bold]"
        info_tbl = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
        info_tbl.add_column("K", style="bold cyan", width=20, no_wrap=True)
        info_tbl.add_column("V", style="white")
        info_tbl.add_row("Verdict", "[bold blue]WAITING FOR PARENT RUN[/bold blue]")
        info_tbl.add_row("Parent Run", f"{p_run_disp}{p_cid_str}")
        info_tbl.add_row("Parent Status", style_status(p_status))
        info_tbl.add_row(
            "Condition",
            "[yellow]Parent must exit 0 (--after-ok)[/yellow]"
            if r_ok
            else "Runs when parent terminates (any exit code)",
        )
        info_tbl.add_row(
            "Action",
            "HTCondor DAGMan will automatically release and queue this job when the parent finishes.",
        )

        console.print(
            Panel(
                info_tbl,
                title=f"[bold cyan]⏳ Dependency Wait Analysis — Job {job_target}[/bold cyan]",
                border_style="cyan",
            )
        )
        return

    console.print(
        f"[dim]Analyzing run[/dim] {_format_run_short(run_dir, data)} "
        f"[dim](Job {job_target}, Status:[/dim] {style_status(cur_status)}[dim])[/dim]"
    )
    _render_queue_analysis_panel(job_target, raw_mode=raw_mode)


# ---------------------------------------------------------------------------
# Subcommands: `runs top` (`gpu`) & `runs ssh` (`shell`)
# ---------------------------------------------------------------------------
def _resolve_running_job_context(
    args: argparse.Namespace, runs_root: Path
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Resolves CLI target to `(job_target, run_name, arch)`.
    Supports raw `ClusterId.ProcId`, raw `ClusterId`, run `#index`, tag, or
    defaults to the most recent `RUNNING` run.
    """
    target = getattr(args, "run", None)
    proc_idx = int(getattr(args, "proc", 0) or 0)

    if target and re.match(r"^\d+\.\d+$", str(target).strip()):
        return str(target).strip(), None, None

    run_dir: Optional[Path] = None
    live_clusters = query_live_condor_clusters()
    if not target:
        for rd in list_run_dirs(runs_root)[:25]:
            d = load_run_json(rd)
            if not d:
                continue
            if run_needs_live_reconcile(d):
                d = reconcile_run(rd, d, live_clusters)
            st = str(d.get("_live_badge") or d.get("status", ""))
            if st.startswith("RUNNING"):
                run_dir = rd
                break
        if run_dir is None:
            run_dir = resolve_run_dir(runs_root, "latest")
    else:
        run_dir = resolve_run_dir(runs_root, target, exit_on_fail=False)
        if run_dir is None and str(target).strip().isdigit():
            return f"{str(target).strip()}.{proc_idx}", None, None
        if run_dir is None:
            console.print(f"[red]✖ Could not resolve run '{target}'.[/red]")
            sys.exit(1)

    if not run_dir:
        sys.exit(1)

    data = load_run_json(run_dir)
    if not data:
        console.print(f"[red]✖ Could not read run.json in {run_dir.name}.[/red]")
        sys.exit(1)

    data = reconcile_run(run_dir, data, live_clusters)
    status_str = str(data.get("_live_badge") or data.get("status", "UNKNOWN"))
    cids = get_all_cluster_ids(data)
    if not cids:
        console.print(f"[yellow]⚠ Run '{run_dir.name}' has no HTCondor cluster ID.[/yellow]")
        sys.exit(1)

    if status_str.startswith(("QUEUED", "HELD")):
        console.print(
            f"[yellow]⚠ Run '{run_dir.name}' is currently {status_str} (not executing on a worker node yet).[/yellow]\n"
            f"[dim]Tip: run [bold white]runs why {run_dir.name}[/bold white] to diagnose queue wait time.[/dim]"
        )
        sys.exit(1)
    if not status_str.startswith("RUNNING"):
        console.print(
            f"[yellow]⚠ Run '{run_dir.name}' is no longer running (status: {status_str}).[/yellow]\n"
            f"[dim]Tip: run [bold white]runs inspect {run_dir.name}[/bold white] to view recorded telemetry.[/dim]"
        )
        sys.exit(1)

    arch = str((data.get("resources_requested") or {}).get("architecture") or "—")
    plain_label, _, _ = _extract_run_display_parts(run_dir, data)
    return f"{cids[-1]}.{proc_idx}", plain_label, arch


def cmd_top(args: argparse.Namespace, runs_root: Path) -> None:
    job_target, run_name, arch = _resolve_running_job_context(args, runs_root)
    watch_mode = bool(getattr(args, "watch", False))
    interval = max(1, int(getattr(args, "interval", 2) or 2))

    if watch_mode:
        from rich.live import Live

        console.print(
            f"[dim]Connecting live telemetry stream to Job {job_target} via condor_ssh_to_job (Ctrl-C to exit)...[/dim]"
        )
        current_view: Optional[Panel] = None
        got_any = False
        try:
            with Live(console=console, screen=True, auto_refresh=False) as live:
                for frame in stream_job_live_telemetry(job_target, interval=interval):
                    got_any = True
                    panel = _build_live_telemetry_panel(job_target, frame, run_name=run_name, arch=arch)
                    current_view = panel
                    live.update(panel, refresh=True)
        except KeyboardInterrupt:
            if current_view is not None:
                console.print(current_view)
            return
        if current_view is not None:
            console.print(current_view)
        elif not got_any:
            console.print(
                f"[yellow]⚠ Could not establish telemetry stream to Job {job_target} (job may have finished).[/yellow]"
            )
    else:
        try:
            with status_spinner(f"Probing worker node GPU & processes (Job {job_target})..."):
                frame = query_job_live_telemetry(job_target)
        except RuntimeError as exc:
            console.print(
                f"[bold red]✖ condor_ssh_to_job failed for Job {job_target}:[/bold red]\n[dim]{exc}[/dim]"
            )
            sys.exit(1)
        if frame is None:
            console.print(
                "[yellow]⚠ 'condor_ssh_to_job' is not available or returned no telemetry (are you on hermes2?).[/yellow]"
            )
            sys.exit(1)
        console.print(_build_live_telemetry_panel(job_target, frame, run_name=run_name, arch=arch))


def cmd_ssh(args: argparse.Namespace, runs_root: Path) -> None:
    if not shutil.which("condor_ssh_to_job"):
        console.print("[yellow]⚠ 'condor_ssh_to_job' is not available in PATH.[/yellow]")
        sys.exit(1)
    job_target, run_name, _ = _resolve_running_job_context(args, runs_root)
    label = f"{run_name} ({job_target})" if run_name else job_target
    console.print(f"[cyan]Connecting interactive shell to {label} via condor_ssh_to_job...[/cyan]")
    res = subprocess.run(["condor_ssh_to_job", "-auto-retry", job_target])
    sys.exit(res.returncode)


def cmd_plot(args: argparse.Namespace, runs_root: Path) -> None:
    run_dir = resolve_run_dir(runs_root, getattr(args, "run", None) or "latest")
    proc_id = int(getattr(args, "proc", 0) or 0)
    width = int(getattr(args, "width", 60) or 60)
    watch = bool(getattr(args, "watch", False))
    interval = max(1, int(getattr(args, "interval", 5) or 5))
    save_arg = getattr(args, "save", None)

    raw_data = load_run_json(run_dir)
    if run_needs_live_reconcile(raw_data):
        data = reconcile_run(run_dir, raw_data, query_live_condor_clusters())
    else:
        data = raw_data

    telem_csv = run_dir / "repro" / f"telemetry_{proc_id}.csv"
    status_str = str(data.get("_live_badge") or data.get("status", "UNKNOWN"))
    is_active = status_str.startswith(("WAITING", "QUEUED", "RUNNING", "HELD"))

    if not telem_csv.is_file():
        if watch and is_active:
            console.print(
                f"[cyan]⏳ Waiting for first telemetry flush ([bold]{telem_csv.name}[/bold], flushes every 60s)...[/cyan]"
            )
            try:
                while not telem_csv.is_file():
                    time.sleep(interval)
                    raw_d = load_run_json(run_dir)
                    if run_needs_live_reconcile(raw_d):
                        data = reconcile_run(run_dir, raw_d, query_live_condor_clusters())
                    st_now = str(data.get("_live_badge") or data.get("status", "UNKNOWN"))
                    if not st_now.startswith(("WAITING", "QUEUED", "RUNNING", "HELD")) and not telem_csv.is_file():
                        console.print(f"[yellow]Run finished ({st_now}) with no telemetry CSV recorded.[/yellow]")
                        return
            except KeyboardInterrupt:
                return
        else:
            if is_active:
                console.print(
                    f"[yellow]⏳ Telemetry file [bold]{telem_csv.name}[/bold] not flushed yet (syncs every 60s).[/yellow]\n"
                    f"  Run with [bold cyan]--watch[/bold cyan] to wait and stream live charts."
                )
            else:
                console.print(f"[yellow]No telemetry CSV found at {telem_csv}.[/yellow]")
            return

    if watch:
        from rich.live import Live

        current_view: Optional[Panel] = None
        try:
            with Live(console=console, screen=True, auto_refresh=False) as live:
                while True:
                    raw_d = load_run_json(run_dir)
                    if run_needs_live_reconcile(raw_d):
                        data = reconcile_run(run_dir, raw_d, query_live_condor_clusters())
                    else:
                        data = raw_d
                    rows = read_telemetry_csv(telem_csv)
                    if rows:
                        current_view = _build_telemetry_plot_panel(
                            run_dir, data, proc_id, rows, width=width
                        )
                        live.update(current_view, refresh=True)
                    st_now = str(data.get("_live_badge") or data.get("status", "UNKNOWN"))
                    if not st_now.startswith(("WAITING", "QUEUED", "RUNNING", "HELD")):
                        break
                    time.sleep(interval)
        except KeyboardInterrupt:
            if current_view is not None:
                console.print(current_view)
            return
        if current_view is not None:
            console.print(current_view)
    else:
        rows = read_telemetry_csv(telem_csv)
        if not rows:
            console.print(f"[yellow]Telemetry file {telem_csv.name} has no samples yet.[/yellow]")
            return
        console.print(_build_telemetry_plot_panel(run_dir, data, proc_id, rows, width=width))

    if save_arg is not None:
        out_path = (
            run_dir / f"telemetry_{proc_id}.png"
            if save_arg == "__AUTO__"
            else Path(save_arg)
        )
        plain_label, _, short_date = _extract_run_display_parts(run_dir, data)
        arch = str((data.get("resources_requested") or {}).get("architecture") or "")
        rows_for_save = read_telemetry_csv(telem_csv)
        is_gpu = (
            any(float(r.get("vram_total_mb", 0.0) or 0.0) > 0 or float(r.get("gpu_util_pct", 0.0) or 0.0) > 0 for r in rows_for_save)
            or arch in ("GPU3090", "A100")
        )
        _save_telemetry_matplotlib(
            telem_csv,
            out_path,
            f"{plain_label} ({short_date}) — proc {proc_id}",
            is_gpu,
            data,
            runs_root,
        )


def cmd_path(args: argparse.Namespace, runs_root: Path) -> None:
    run_dir = resolve_run_dir(runs_root, getattr(args, "run", None) or "latest")
    if getattr(args, "code", False):
        target_path = run_dir / "repro" / "code"
    elif getattr(args, "repro", False):
        target_path = run_dir / "repro"
    elif getattr(args, "json", False):
        target_path = run_dir / "run.json"
    elif getattr(args, "telemetry", False):
        proc_idx = getattr(args, "proc", None)
        proc_id = 0 if proc_idx is None else int(proc_idx)
        target_path = run_dir / "repro" / f"telemetry_{proc_id}.csv"
    elif getattr(args, "out", False) or getattr(args, "err", False) or getattr(args, "condor", False):
        data = load_run_json(run_dir)
        target_path = _resolve_run_log_file(
            run_dir,
            data,
            err=bool(getattr(args, "err", False)),
            condor=bool(getattr(args, "condor", False)),
            proc=getattr(args, "proc", None),
            quiet=True,
        )
    else:
        target_path = run_dir
    sys.stdout.write(f"{target_path.resolve()}\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# CLI Argument Parser & Entrypoint
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = RichArgumentParser(
        prog="runs",
        description="Hermes 2 HTCondor Run Manager, Telemetry Inspector & Recovery CLI.",
        command_groups=[
            ("Monitor & Inspect", ["ls", "watch", "inspect", "logs", "diff", "path"]),
            ("Diagnostics & Telemetry", ["why", "top", "plot", "ssh", "nodes"]),
            ("Recovery & Lifecycle", ["retry", "sync", "pull", "stop", "pin", "unpin", "clean"]),
        ],
        examples=[
            ("runs", "List recent runs (defaults to `runs ls`)"),
            ("runs watch -s RUNNING", "Live-watch active runs in the queue"),
            ("runs logs latest -f", "Stream live stdout of the latest run"),
            ("runs why", "Diagnose why your latest queued job is waiting"),
            ("runs gpu -w", "Live GPU utilization & VRAM on worker node"),
            ("runs retry latest --in-place -u -y", "Update code snapshot and retry failed procs"),
            ("runs nodes -a GPU3090 -f", "Check free RTX 3090 nodes across Hermes 2"),
        ],
        footer_hint="Run 'runs <command> --help' for command-specific options.",
    )
    sub = p.add_subparsers(dest="cmd")

    p_ls = sub.add_parser("ls", help="List runs with live HTCondor status reconciliation")
    p_ls.add_argument("-n", "--limit", type=int, default=25, help="Max runs to display (default: 25)")
    p_ls.add_argument("-s", "--status", type=str, help="Filter by status (e.g. RUNNING, FAILED)")
    p_ls.add_argument("-t", "--tag", type=str, help="Filter by substring/tag in run ID")
    p_ls.add_argument("-p", "--pinned", action="store_true", help="Show only pinned runs")
    p_ls.add_argument("-w", "--watch", action="store_true", help="Live-refresh table every N seconds")
    p_ls.add_argument("-i", "--interval", type=int, default=2, help="Refresh interval in seconds for --watch")

    p_watch = sub.add_parser("watch", help="Live-watch runs and HTCondor queue progress")
    p_watch.add_argument("-n", "--limit", type=int, default=25, help="Max runs to display (default: 25)")
    p_watch.add_argument("-s", "--status", type=str, help="Filter by status (e.g. RUNNING, FAILED)")
    p_watch.add_argument("-t", "--tag", type=str, help="Filter by substring/tag in run ID")
    p_watch.add_argument("-p", "--pinned", action="store_true", help="Show only pinned runs")
    p_watch.add_argument("-i", "--interval", type=int, default=2, help="Refresh interval in seconds (default: 2)")

    p_ins = sub.add_parser(
        "inspect", aliases=["info", "show"], help="Inspect run telemetry, queue match analysis, and stderr tails"
    )
    p_ins.add_argument("run", nargs="?", help="Run ID, index (#), cluster ID, or 'latest'")

    p_why = sub.add_parser(
        "why",
        aliases=["analyze", "q"],
        help="Analyze why a queued job is waiting (condor_q -better-analyze)",
    )
    p_why.add_argument(
        "run",
        nargs="?",
        help="Run ID, index (#), cluster ID (e.g. 1074510.0), or defaults to latest queued run",
    )
    p_why.add_argument(
        "-p", "--proc", type=int, default=0, help="Process index for sweeps (default: 0)"
    )
    p_why.add_argument(
        "--raw", action="store_true", help="Show raw condor_q -better-analyze text output"
    )

    p_top = sub.add_parser(
        "top",
        aliases=["gpu"],
        help="Show live GPU VRAM, utilization, and process telemetry on the worker node (condor_ssh_to_job)",
    )
    p_top.add_argument(
        "run",
        nargs="?",
        help="Run ID, index (#), cluster ID, or defaults to latest RUNNING run",
    )
    p_top.add_argument(
        "-p", "--proc", type=int, default=0, help="Process index for sweeps (default: 0)"
    )
    p_top.add_argument(
        "-w", "--watch", action="store_true", help="Stream live worker telemetry continuously"
    )
    p_top.add_argument(
        "-i", "--interval", type=int, default=2, help="Refresh interval in seconds for --watch (default: 2)"
    )

    p_ssh = sub.add_parser(
        "ssh",
        aliases=["shell"],
        help="Open an interactive shell inside the running job's slot on the worker node (condor_ssh_to_job)",
    )
    p_ssh.add_argument(
        "run",
        nargs="?",
        help="Run ID, index (#), cluster ID, or defaults to latest RUNNING run",
    )
    p_ssh.add_argument(
        "-p", "--proc", type=int, default=0, help="Process index for sweeps (default: 0)"
    )

    p_path = sub.add_parser(
        "path",
        aliases=["dir", "where"],
        help="Print the resolved filesystem path of a run directory or its artifacts (for `cd $(runs path)`)",
    )
    p_path.add_argument(
        "run", nargs="?", default="latest", help="Run ID, index (#), cluster ID, or 'latest' (default)"
    )
    p_path.add_argument("-c", "--code", action="store_true", help="Print path to repro/code/ snapshot directory")
    p_path.add_argument("-r", "--repro", action="store_true", help="Print path to repro/ directory")
    p_path.add_argument("-o", "--out", action="store_true", help="Print path to stdout (.out) log file")
    p_path.add_argument("-e", "--err", action="store_true", help="Print path to stderr (.err) log file")
    p_path.add_argument("--condor", action="store_true", help="Print path to repro/condor.log")
    p_path.add_argument("-j", "--json", action="store_true", help="Print path to run.json")
    p_path.add_argument("-t", "--telemetry", action="store_true", help="Print path to repro/telemetry_<proc>.csv")
    p_path.add_argument("-p", "--proc", type=int, help="Process index for sweep log files")

    p_plot = sub.add_parser(
        "plot",
        aliases=["telemetry", "chart"],
        help="Visualize 5s hardware & I/O telemetry time-series with bottleneck diagnosis",
    )
    p_plot.add_argument(
        "run", nargs="?", default="latest", help="Run ID, index (#), cluster ID, or 'latest' (default)"
    )
    p_plot.add_argument("-p", "--proc", type=int, default=0, help="Process index for sweeps (default: 0)")
    p_plot.add_argument("-w", "--watch", action="store_true", help="Live-refresh terminal plot while active")
    p_plot.add_argument("-i", "--interval", type=int, default=5, help="Refresh interval for --watch (default: 5s)")
    p_plot.add_argument("--width", type=int, default=60, help="Width of terminal charts in characters (default: 60)")
    p_plot.add_argument(
        "-s",
        "--save",
        nargs="?",
        const="__AUTO__",
        default=None,
        help="Save high-res multi-panel figure via matplotlib (default: <run_dir>/telemetry_<proc>.png)",
    )

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
        "retry",
        aliases=["resume", "continue"],
        help="Retry failed runs or partial sweeps",
        examples=[
            ("runs retry latest --in-place -u -y", "Update code snapshot and retry failed procs in-place"),
            ("runs retry latest --new-run -y", "Relaunch as a brand-new run from frozen snapshot"),
            ("runs retry 3 --after 1 --after-ok -y", "Queue retry of run #3 after run #1 exits 0"),
        ],
    )
    p_ret.add_argument("run", nargs="?", help="Run ID, index (#), or 'latest'")
    g_ret_mode = p_ret.add_argument_group("Recovery Mode")
    g_ret_mode.add_argument("--in-place", action="store_true", help="Resume inside the same run directory")
    g_ret_mode.add_argument(
        "--new-run", action="store_true", help="Launch a brand-new run directory via `send`"
    )
    g_ret_mode.add_argument(
        "-u",
        "--update-code",
        action="store_true",
        help="Refresh repro/code/ snapshot from working tree before retrying",
    )
    g_ret_mode.add_argument(
        "--all",
        action="store_true",
        help="Re-run all sweep processes (even those with exit_code 0)",
    )
    g_ret_mode.add_argument(
        "--force",
        action="store_true",
        help="Allow in-place retry even if jobs are still active in condor_q",
    )
    g_ret_chain = p_ret.add_argument_group("Dependencies & Chaining")
    g_ret_chain.add_argument(
        "--after",
        type=str,
        default=None,
        metavar="run",
        help="Queue retry after specified run (ID, #index, 'latest', or cluster ID) completes",
    )
    g_ret_chain.add_argument(
        "--after-ok",
        action="store_true",
        help="When chained (via --after or downstream chain), only run if parent exits 0",
    )
    g_ret_chain.add_argument(
        "--chain",
        "--cascade",
        dest="chain",
        action="store_true",
        help="Automatically retry downstream chained runs that depended on this run",
    )
    g_ret_chain.add_argument(
        "--single",
        "--no-cascade",
        dest="single",
        action="store_true",
        help="Retry only the specified run without re-queueing downstream chained runs",
    )
    p_ret.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompts")

    p_sync = sub.add_parser(
        "sync", help="Hot-patch a run's repro/code/ snapshot with current workspace files"
    )
    p_sync.add_argument("run", nargs="?", help="Run ID, index (#), or 'latest'")

    p_pull = sub.add_parser(
        "pull",
        aliases=["fetch", "download"],
        help="Download or sync remote cluster runs to local machine for Jupyter/offline analysis",
        examples=[
            ("runs pull", "Incremental sync of recent runs (skips code & heavy weights)"),
            ("runs pull latest --weights", "Pull latest run including checkpoints (*.pt, *.ckpt)"),
            ("runs pull -s COMPLETED --pinned", "Pull only completed pinned runs"),
            ("runs pull --watch -i 15", "Continuously sync active runs every 15s"),
        ],
    )
    p_pull.add_argument(
        "run",
        nargs="*",
        help="Optional run ID(s), index (#), cluster ID, tag substring, or 'latest' (default: recent runs)",
    )
    g_pull_rem = p_pull.add_argument_group("Remote Target")
    g_pull_rem.add_argument(
        "-r",
        "--remote",
        type=str,
        metavar="spec",
        help="Remote '[user@]host[:/path/to/project]' or '/remote/path' (overrides config)",
    )
    g_pull_rem.add_argument(
        "-H",
        "--host",
        type=str,
        metavar="host",
        help="Remote SSH host (default: 'hermes' or remote_host in .hermes.toml)",
    )
    g_pull_rem.add_argument(
        "-d",
        "--remote-dir",
        type=str,
        metavar="path",
        help="Remote project or runs/ directory path (auto-discovered on remote if omitted)",
    )
    g_pull_flt = p_pull.add_argument_group("Selection & Filters")
    g_pull_flt.add_argument(
        "-n",
        "--limit",
        type=int,
        default=25,
        help="Max recent runs to check when no run query is specified",
    )
    g_pull_flt.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Pull all matching remote runs without the -n 25 limit",
    )
    g_pull_flt.add_argument(
        "-s",
        "--status",
        type=str,
        help="Filter remote runs by status (e.g. COMPLETED, RUNNING, FAILED)",
    )
    g_pull_flt.add_argument(
        "-t",
        "--tag",
        type=str,
        help="Filter remote runs by tag or run ID substring",
    )
    g_pull_flt.add_argument(
        "-p",
        "--pinned",
        action="store_true",
        help="Pull only pinned remote runs",
    )
    g_pull_cnt = p_pull.add_argument_group("Content & Filters")
    g_pull_cnt.add_argument(
        "-w",
        "--weights",
        action="store_true",
        help="Include model weights/checkpoints (*.pt, *.ckpt, *.safetensors, etc.)",
    )
    g_pull_cnt.add_argument(
        "-c",
        "--code",
        action="store_true",
        help="Include repro/code/ source snapshots (skipped by default)",
    )
    g_pull_cnt.add_argument(
        "--full",
        action="store_true",
        help="Include both model weights (--weights) and code snapshots (--code)",
    )
    g_pull_cnt.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="glob",
        help="Glob pattern(s) of files to always include (e.g. --include '*.pt')",
    )
    g_pull_cnt.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="glob",
        help="Glob pattern(s) of files to exclude (e.g. --exclude '*.npz')",
    )
    g_pull_exec = p_pull.add_argument_group("Sync Behavior")
    g_pull_exec.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Re-download all matching files even if already up-to-date locally",
    )
    g_pull_exec.add_argument(
        "--watch",
        action="store_true",
        help="Continuously sync active remote runs until they complete",
    )
    g_pull_exec.add_argument(
        "-i",
        "--interval",
        type=int,
        default=15,
        help="Refresh interval in seconds for --watch",
    )
    g_pull_exec.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview remote runs and files without transferring data",
    )

    p_stop = sub.add_parser("stop", aliases=["kill", "rm"], help="Abort active HTCondor job (condor_rm)")
    p_stop.add_argument("run", nargs="?", help="Run ID, index (#), cluster ID, or 'latest'")
    p_stop.add_argument("-a", "--all", action="store_true", help="Abort all currently active runs")
    p_stop.add_argument(
        "--single",
        "--no-cascade",
        dest="single",
        action="store_true",
        help="Stop only the specified run without stopping downstream chained runs",
    )
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

    p_pin = sub.add_parser(
        "pin",
        aliases=["star", "keep", "lock"],
        help="Pin a run to protect it from 'runs clean' and attach an optional note",
    )
    p_pin.add_argument("run", nargs="?", help="Run ID, index (#), cluster ID, or 'latest'")
    p_pin.add_argument(
        "-n", "--note", type=str, default=None, help="Optional note/description to attach to the run"
    )
    p_pin.add_argument("-u", "--unpin", action="store_true", help="Unpin the specified run")

    p_unpin = sub.add_parser(
        "unpin",
        aliases=["unstar", "unlock"],
        help="Unpin a run so it becomes eligible for 'runs clean'",
    )
    p_unpin.add_argument("run", nargs="?", help="Run ID, index (#), cluster ID, or 'latest'")

    p_diff = sub.add_parser("diff", help="Compare provenance and code snapshots between two runs")
    p_diff.add_argument("run_a", nargs="?", help="First run ID or index")
    p_diff.add_argument("run_b", nargs="?", help="Second run ID or index")

    p_nodes = sub.add_parser(
        "nodes",
        aliases=["cluster", "pools"],
        help="Show live Hermes 2 architecture pool, worker node, and slot claim status (condor_status)",
    )
    p_nodes.add_argument(
        "-a", "--arch", type=str, help="Filter by architecture pool (e.g. GPU3090, A100, AMD)"
    )
    p_nodes.add_argument(
        "-n", "--node", type=str, help="Filter by node hostname substring (e.g. 3090, her07)"
    )
    p_nodes.add_argument(
        "-u",
        "--user",
        nargs="?",
        const="__ME__",
        help="Filter by active RemoteUser (defaults to your username if given without value)",
    )
    p_nodes.add_argument(
        "-m", "--mine", action="store_true", help="Show only nodes and slot claims running your jobs"
    )
    p_nodes.add_argument(
        "-f", "--free", action="store_true", help="Show only nodes with free GPUs/CPUs"
    )
    p_nodes.add_argument(
        "-b", "--busy", action="store_true", help="Show only busy/claimed nodes"
    )
    p_nodes.add_argument(
        "-d",
        "--detail",
        "--slots",
        dest="detail",
        action="store_true",
        help="Show hardware load, GPU model, disk, and per-slot active job claims",
    )
    p_nodes.add_argument(
        "--where",
        type=str,
        help="Pass a custom HTCondor ClassAd -constraint expression to condor_status",
    )
    p_nodes.add_argument(
        "-w", "--watch", action="store_true", help="Live-refresh cluster availability"
    )
    p_nodes.add_argument(
        "-i", "--interval", type=int, default=2, help="Refresh interval in seconds for --watch"
    )

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    runs_root = get_runs_root(Path.cwd())

    cmd = args.cmd or "ls"
    if cmd == "ls":
        if not hasattr(args, "limit"):
            args.limit, args.status, args.tag, args.watch, args.interval = 25, None, None, False, 2
        if not hasattr(args, "pinned"):
            args.pinned = False
        cmd_ls(args, runs_root)
    elif cmd == "watch":
        args.watch = True
        cmd_ls(args, runs_root)
    elif cmd in ("inspect", "info", "show"):
        cmd_inspect(args, runs_root)
    elif cmd in ("why", "analyze", "q"):
        cmd_why(args, runs_root)
    elif cmd in ("top", "gpu"):
        cmd_top(args, runs_root)
    elif cmd in ("ssh", "shell"):
        cmd_ssh(args, runs_root)
    elif cmd in ("path", "dir", "where"):
        cmd_path(args, runs_root)
    elif cmd in ("plot", "telemetry", "chart"):
        cmd_plot(args, runs_root)
    elif cmd in ("logs", "tail"):
        cmd_logs(args, runs_root)
    elif cmd in ("retry", "resume", "continue"):
        cmd_retry(args, runs_root)
    elif cmd == "sync":
        cmd_sync(args, runs_root)
    elif cmd in ("pull", "fetch", "download"):
        cmd_pull(args, runs_root)
    elif cmd in ("stop", "kill", "rm"):
        cmd_stop(args, runs_root)
    elif cmd in ("clean", "prune"):
        cmd_clean(args, runs_root)
    elif cmd in ("pin", "star", "keep", "lock"):
        cmd_pin(args, runs_root, unpin=bool(getattr(args, "unpin", False)))
    elif cmd in ("unpin", "unstar", "unlock"):
        cmd_pin(args, runs_root, unpin=True)
    elif cmd == "diff":
        cmd_diff(args, runs_root)
    elif cmd in ("nodes", "cluster", "pools"):
        cmd_nodes(args, runs_root)


if __name__ == "__main__":
    main()
