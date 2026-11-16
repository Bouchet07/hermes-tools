"""
telemetry.py — Live worker SSH telemetry (`condor_ssh_to_job`) and 5s time-series CSV plotting (`runs top` & `runs plot`).
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich import box
from rich.panel import Panel
from rich.table import Table

from hermes_tools.core import (
    console,
    discover_project_venv,
    format_duration,
    format_run_short,
    resolve_submit_python,
    status_spinner,
    style_status,
)

# ---------------------------------------------------------------------------
# Live Worker Node GPU & Process Telemetry (`condor_ssh_to_job`)
# ---------------------------------------------------------------------------
_IGNORED_PROBE_COMMS = {
    "ps",
    "sh",
    "bash",
    "dash",
    "sshd",
    "condor_starter",
    "condor_exec.exe",
    "sleep",
    "nvidia-smi",
    "hostname",
    "cat",
    "id",
    "time",
}


def build_ssh_telemetry_probe(interval: Optional[int] = None) -> str:
    """
    Builds a POSIX-safe shell one-liner for `condor_ssh_to_job` that emits
    hostname, loadavg, GPU metrics (`nvidia-smi`), and user processes (`ps`).
    """
    body = (
        "LC_ALL=C; export LC_ALL; "
        "emit_frame() { "
        "echo '===HERMES_FRAME_BEGIN==='; "
        "echo '---HOST---'; "
        "hostname -f 2>/dev/null || hostname 2>/dev/null || echo unknown; "
        "cat /proc/loadavg 2>/dev/null || echo ''; "
        "echo '---GPU---'; "
        "if command -v nvidia-smi >/dev/null 2>&1; then "
        "nvidia-smi --query-gpu=index,name,gpu_bus_id,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit "
        "--format=csv,noheader,nounits 2>/dev/null || true; "
        "fi; "
        "echo '---GPU_APPS---'; "
        "if command -v nvidia-smi >/dev/null 2>&1; then "
        "nvidia-smi --query-compute-apps=pid,gpu_bus_id,used_memory --format=csv,noheader,nounits 2>/dev/null || true; "
        "fi; "
        "echo '---PS---'; "
        "ps -u \"$(id -un)\" -o pid=,ppid=,stat=,pcpu=,rss=,etime=,comm=,args= --sort=-pcpu 2>/dev/null || true; "
        "echo '---PROC_STAT---'; "
        "cat /proc/self/stat 2>/dev/null || true; "
        "for _p in $(ps -u \"$(id -un)\" -o pid= 2>/dev/null); do cat \"/proc/$_p/stat\" 2>/dev/null || true; done; "
        "echo '===HERMES_FRAME_END==='; "
        "}; "
    )
    if interval is not None and interval > 0:
        return body + f"while true; do emit_frame; sleep {int(interval)}; done"
    return body + "emit_frame"


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(str(val).strip())
    except (ValueError, TypeError):
        return default


def _parse_ps_etime_seconds(etime_str: str) -> int:
    s = (etime_str or "").strip()
    if not s:
        return 0
    days = 0
    if "-" in s:
        d_part, s = s.split("-", 1)
        if d_part.isdigit():
            days = int(d_part)
    parts = [int(p) if p.isdigit() else 0 for p in s.split(":")]
    if len(parts) == 3:
        return days * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return days * 86400 + parts[0] * 60 + parts[1]
    return days * 86400


def _format_etime_seconds(total_sec: int) -> str:
    sec = max(0, int(total_sec))
    days, rem = divmod(sec, 86400)
    hours, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    if days > 0:
        return f"{days}-{hours:02d}:{mins:02d}:{secs:02d}"
    if hours > 0:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def _parse_proc_stat_line(line: str) -> Optional[Tuple[int, int, int]]:
    """
    Parses a Linux `/proc/<pid>/stat` line into `(pid, utime_plus_stime_ticks, starttime_ticks)`.
    """
    s = (line or "").strip()
    if not s or ")" not in s:
        return None
    head, _, tail = s.rpartition(")")
    pid_str = head.partition("(")[0].strip()
    if not pid_str.isdigit():
        return None
    rest = tail.split()
    if len(rest) <= 19:
        return None
    try:
        pid = int(pid_str)
        cpu_ticks = int(rest[11]) + int(rest[12])
        start_ticks = int(rest[19])
        return pid, cpu_ticks, start_ticks
    except (ValueError, IndexError):
        return None


def parse_ssh_telemetry_frame(
    raw_frame: str,
    prev_proc_ticks: Optional[Dict[int, Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    """
    Parses a single telemetry frame emitted by `build_ssh_telemetry_probe` into
    host metadata, GPU telemetry records, and active user workload processes.
    """
    section = ""
    host_lines: List[str] = []
    gpu_lines: List[str] = []
    app_lines: List[str] = []
    ps_lines: List[str] = []
    stat_lines: List[str] = []

    for line in (raw_frame or "").splitlines():
        s = line.strip()
        if s in ("===HERMES_FRAME_BEGIN===", "===HERMES_FRAME_END==="):
            continue
        if s == "---HOST---":
            section = "HOST"
            continue
        if s == "---GPU---":
            section = "GPU"
            continue
        if s == "---GPU_APPS---":
            section = "GPU_APPS"
            continue
        if s == "---PS---":
            section = "PS"
            continue
        if s == "---PROC_STAT---":
            section = "PROC_STAT"
            continue

        if section == "HOST" and s:
            host_lines.append(s)
        elif section == "GPU" and s:
            gpu_lines.append(s)
        elif section == "GPU_APPS" and s:
            app_lines.append(s)
        elif section == "PS" and s:
            ps_lines.append(s)
        elif section == "PROC_STAT" and s:
            stat_lines.append(s)

    now_ticks: Optional[int] = None
    proc_kernel_stat: Dict[int, Tuple[int, int]] = {}
    for idx, st_line in enumerate(stat_lines):
        parsed_st = _parse_proc_stat_line(st_line)
        if parsed_st is None:
            continue
        st_pid, cpu_t, start_t = parsed_st
        if idx == 0:
            now_ticks = start_t
        else:
            proc_kernel_stat[st_pid] = (cpu_t, start_t)
            if now_ticks is not None and start_t > now_ticks:
                now_ticks = start_t

    hostname = host_lines[0] if host_lines else "unknown"
    load_avg: Optional[float] = None
    if len(host_lines) >= 2:
        toks = host_lines[1].split()
        if toks:
            try:
                load_avg = float(toks[0])
            except ValueError:
                load_avg = None

    gpus: List[Dict[str, Any]] = []
    for g_line in gpu_lines:
        parts = [p.strip() for p in g_line.split(",")]
        if len(parts) < 6:
            continue
        short_model = (
            parts[1]
            .replace("NVIDIA GeForce ", "")
            .replace("NVIDIA ", "")
            .replace("Tesla ", "")
            .strip()
        )
        gpus.append(
            {
                "index": parts[0],
                "model": short_model,
                "bus_id": parts[2],
                "util_pct": _safe_float(parts[3]),
                "mem_used_mb": _safe_float(parts[4]),
                "mem_total_mb": _safe_float(parts[5]),
                "temp_c": _safe_float(parts[6]) if len(parts) > 6 else None,
                "power_w": _safe_float(parts[7]) if len(parts) > 7 else None,
                "power_limit_w": _safe_float(parts[8]) if len(parts) > 8 else None,
            }
        )

    pid_vram_mb: Dict[int, float] = {}
    for a_line in app_lines:
        parts = [p.strip() for p in a_line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            pid = int(parts[0])
            pid_vram_mb[pid] = pid_vram_mb.get(pid, 0.0) + _safe_float(parts[2])

    raw_procs: List[Dict[str, Any]] = []
    for p_line in ps_lines:
        toks8 = p_line.split(None, 7)
        if len(toks8) < 5 or not toks8[0].isdigit():
            continue
        if len(toks8) >= 7 and toks8[1].isdigit() and not ("." in toks8[2]):
            pid = int(toks8[0])
            ppid = int(toks8[1])
            stat_str = toks8[2]
            cpu_pct = _safe_float(toks8[3])
            rss_kb = _safe_float(toks8[4])
            etime = toks8[5]
            comm = toks8[6]
            args_str = toks8[7] if len(toks8) > 7 else comm
        else:
            toks6 = p_line.split(None, 5)
            pid = int(toks6[0])
            ppid = 0
            stat_str = "R" if _safe_float(toks6[1]) > 0 else "S"
            cpu_pct = _safe_float(toks6[1])
            rss_kb = _safe_float(toks6[2])
            etime = toks6[3]
            comm = toks6[4]
            args_str = toks6[5] if len(toks6) > 5 else comm

        if "HERMES_FRAME_BEGIN" in args_str or "nvidia-smi --query" in args_str:
            continue
        if "condor_ssh_to_job" in args_str or "condor_ssh_to_j" in comm:
            continue
        if comm in _IGNORED_PROBE_COMMS and pid not in pid_vram_mb and cpu_pct < 1.0:
            continue
        if args_str.endswith("/job.sh") or "/repro/job.sh" in args_str:
            continue

        etime_sec = _parse_ps_etime_seconds(etime)
        if pid in proc_kernel_stat and now_ticks is not None:
            cpu_ticks, start_ticks = proc_kernel_stat[pid]
            kern_elapsed_ticks = max(0, now_ticks - start_ticks)
            kern_elapsed_sec = kern_elapsed_ticks // 100
            if etime_sec == 0 and kern_elapsed_sec > 0:
                etime_sec = kern_elapsed_sec
                etime = _format_etime_seconds(kern_elapsed_sec)
            if prev_proc_ticks and pid in prev_proc_ticks:
                prev_cpu_t, prev_now_t = prev_proc_ticks[pid]
                dt_ticks = now_ticks - prev_now_t
                if dt_ticks > 0:
                    cpu_pct = max(0.0, ((cpu_ticks - prev_cpu_t) / float(dt_ticks)) * 100.0)
            elif cpu_pct == 0.0 and kern_elapsed_ticks > 0:
                cpu_pct = max(0.0, (cpu_ticks / float(kern_elapsed_ticks)) * 100.0)

        state_char = stat_str[0].upper() if stat_str else "S"
        raw_procs.append(
            {
                "pid": pid,
                "ppid": ppid,
                "state_char": state_char,
                "cpu_pct": cpu_pct,
                "rss_mb": rss_kb / 1024.0,
                "vram_mb": pid_vram_mb.get(pid, 0.0),
                "elapsed": etime,
                "etime_sec": etime_sec,
                "comm": comm,
                "command": args_str,
            }
        )

    by_pid: Dict[int, Dict[str, Any]] = {p["pid"]: p for p in raw_procs}

    def _find_root_key(p: Dict[str, Any]) -> str:
        cmd = str(p["command"])
        if "multiprocessing." in cmd and p["ppid"] in by_pid:
            parent = by_pid[p["ppid"]]
            return str(parent["command"])
        return cmd

    grouped_map: Dict[str, List[Dict[str, Any]]] = {}
    for p in raw_procs:
        key = _find_root_key(p)
        grouped_map.setdefault(key, []).append(p)

    processes: List[Dict[str, Any]] = []
    for cmd_key, members in grouped_map.items():
        root = max(members, key=lambda m: (int(m["etime_sec"]), -int(m["pid"])))
        workers = len(members) - 1
        tot_cpu = sum(float(m["cpu_pct"]) for m in members)
        tot_vram = sum(float(m["vram_mb"]) for m in members)
        max_rss = max(float(m["rss_mb"]) for m in members)

        n_run = sum(1 for m in members if m["state_char"] == "R")
        n_io = sum(1 for m in members if m["state_char"] == "D")
        if n_io > 0 and n_run > 0:
            state_label = f"RUN({n_run})·I/O({n_io})"
        elif n_io > 0:
            state_label = f"DISK I/O ({n_io})" if len(members) > 1 else "DISK I/O"
        elif n_run > 0 or tot_cpu >= 5.0:
            state_label = f"RUN ({n_run})" if (len(members) > 1 and n_run > 0) else "RUN"
        else:
            state_label = "IDLE"

        processes.append(
            {
                "pid": root["pid"],
                "workers": workers,
                "state": state_label,
                "cpu_pct": tot_cpu,
                "rss_mb": max_rss,
                "vram_mb": tot_vram,
                "elapsed": root["elapsed"],
                "comm": root["comm"],
                "command": cmd_key,
            }
        )

    processes.sort(key=lambda p: (-float(p["vram_mb"]), -float(p["cpu_pct"]), -float(p["rss_mb"])))

    return {
        "hostname": hostname,
        "host": hostname.split(".")[0] if hostname else "unknown",
        "load_avg": load_avg,
        "gpus": gpus,
        "processes": processes,
        "_proc_ticks": (
            {pid: (cpu_t, now_ticks) for pid, (cpu_t, _) in proc_kernel_stat.items()}
            if now_ticks is not None
            else {}
        ),
    }


def query_job_live_telemetry(job_target: str) -> Optional[Dict[str, Any]]:
    """Runs a single-shot `condor_ssh_to_job` probe and returns parsed worker telemetry."""
    if not shutil.which("condor_ssh_to_job"):
        return None
    probe_cmd = build_ssh_telemetry_probe(interval=None)
    try:
        res = subprocess.run(
            ["condor_ssh_to_job", "-auto-retry", str(job_target), probe_cmd],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        out = (res.stdout or "").strip()
        if "===HERMES_FRAME_BEGIN===" not in out:
            err_msg = (res.stderr or out or "").strip()
            if err_msg:
                raise RuntimeError(err_msg)
            return None
        return parse_ssh_telemetry_frame(out)
    except RuntimeError:
        raise
    except Exception:
        return None


def stream_job_live_telemetry(job_target: str, interval: int = 2):
    """
    Opens a single persistent `condor_ssh_to_job` session and yields parsed
    telemetry frames every `interval` seconds with zero SSH reconnect lag.
    """
    if not shutil.which("condor_ssh_to_job"):
        return
    probe_cmd = build_ssh_telemetry_probe(interval=max(1, int(interval)))
    proc = subprocess.Popen(
        ["condor_ssh_to_job", "-auto-retry", str(job_target), probe_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    try:
        buf: List[str] = []
        in_frame = False
        prev_ticks: Optional[Dict[int, Tuple[int, int]]] = None
        assert proc.stdout is not None
        for line in proc.stdout:
            s = line.strip()
            if s == "===HERMES_FRAME_BEGIN===":
                buf = []
                in_frame = True
                continue
            if s == "===HERMES_FRAME_END===":
                if in_frame:
                    parsed = parse_ssh_telemetry_frame("\n".join(buf), prev_proc_ticks=prev_ticks)
                    if parsed.get("_proc_ticks"):
                        prev_ticks = parsed["_proc_ticks"]
                    yield parsed
                in_frame = False
                buf = []
                continue
            if in_frame:
                buf.append(line)
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def _build_live_telemetry_panel(
    job_target: str,
    frame: Dict[str, Any],
    run_name: Optional[str] = None,
    arch: Optional[str] = None,
) -> Panel:
    from rich.console import Group

    host_short = str(frame.get("host") or "unknown")
    host_fqdn = str(frame.get("hostname") or host_short)
    load_avg = frame.get("load_avg")

    info_tbl = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
    info_tbl.add_column("K", style="bold cyan", width=20, no_wrap=True)
    info_tbl.add_column("V", style="white")
    if run_name:
        info_tbl.add_row("Run", f"[bold white]{run_name}[/bold white] ({style_status('RUNNING')})")
    clean_arch = str(arch or "").strip()
    if clean_arch.upper() in ("", "—", "UNDEFINED", "NONE", "ANY"):
        pool_str = ""
    else:
        pool_str = f" · Pool: [bold magenta]{clean_arch}[/bold magenta]"
    load_str = f" · Load: [cyan]{load_avg:.2f}[/cyan]" if isinstance(load_avg, (int, float)) else ""
    info_tbl.add_row("Worker Node", f"[bold white]{host_fqdn}[/bold white]{pool_str}{load_str}")

    renderables: List[Any] = [info_tbl]

    gpus = frame.get("gpus") or []
    if gpus:
        gpu_tbl = Table(
            title="Allocated GPUs",
            title_style="bold cyan",
            box=box.SIMPLE_HEAVY,
            header_style="bold cyan",
        )
        gpu_tbl.add_column("GPU", style="dim", no_wrap=True)
        gpu_tbl.add_column("Model", style="bold white", no_wrap=True)
        gpu_tbl.add_column("Util", justify="right", no_wrap=True)
        gpu_tbl.add_column("VRAM Used / Total", justify="right", no_wrap=True)
        gpu_tbl.add_column("Temp", justify="right", no_wrap=True)
        gpu_tbl.add_column("Power", justify="right", no_wrap=True)

        for g in gpus:
            util = float(g.get("util_pct", 0.0))
            util_col = "bold green" if util >= 50 else ("yellow" if util >= 10 else "dim")
            used_gb = float(g.get("mem_used_mb", 0.0)) / 1024.0
            tot_gb = max(0.1, float(g.get("mem_total_mb", 0.0)) / 1024.0)
            vram_pct = min(100, round((used_gb / tot_gb) * 100))
            temp = g.get("temp_c")
            temp_str = f"{temp:.0f}°C" if isinstance(temp, (int, float)) and temp > 0 else "—"
            pw = g.get("power_w")
            plim = g.get("power_limit_w")
            if isinstance(pw, (int, float)) and pw > 0:
                pw_str = (
                    f"{pw:.0f} / {plim:.0f} W"
                    if isinstance(plim, (int, float)) and plim > 0
                    else f"{pw:.0f} W"
                )
            else:
                pw_str = "—"

            gpu_tbl.add_row(
                str(g.get("index", "0")),
                str(g.get("model", "GPU")),
                f"[{util_col}]{util:.0f} %[/{util_col}]",
                f"[cyan]{used_gb:.1f}[/cyan] / {tot_gb:.1f} GB ({vram_pct}%)",
                temp_str,
                pw_str,
            )
        renderables.append(gpu_tbl)

    procs = frame.get("processes") or []
    if procs:
        proc_tbl = Table(
            title="Active Job Processes",
            title_style="bold cyan",
            box=box.SIMPLE_HEAVY,
            header_style="bold cyan",
        )
        proc_tbl.add_column("PID", style="dim", justify="right", no_wrap=True)
        proc_tbl.add_column("State", no_wrap=True)
        proc_tbl.add_column("CPU %", justify="right", style="green", no_wrap=True)
        proc_tbl.add_column("Host RAM", justify="right", no_wrap=True)
        proc_tbl.add_column("GPU VRAM", justify="right", style="cyan", no_wrap=True)
        proc_tbl.add_column("Elapsed", justify="right", no_wrap=True)
        proc_tbl.add_column("Command", style="white")

        for p in procs[:12]:
            st_raw = str(p.get("state") or "RUN")
            if "I/O" in st_raw:
                st_cell = f"[bold yellow]{st_raw}[/bold yellow]"
            elif st_raw.startswith("RUN"):
                st_cell = f"[bold green]{st_raw}[/bold green]"
            else:
                st_cell = f"[dim]{st_raw}[/dim]"

            rss_mb = float(p.get("rss_mb", 0.0))
            ram_str = f"{rss_mb / 1024.0:.1f} GB" if rss_mb >= 1024.0 else f"{rss_mb:.0f} MB"
            vram_mb = float(p.get("vram_mb", 0.0))
            if vram_mb >= 1024.0:
                vram_str = f"{vram_mb / 1024.0:.1f} GB"
            elif vram_mb > 0:
                vram_str = f"{vram_mb:.0f} MB"
            else:
                vram_str = "[dim]—[/dim]"
            cmd_text = str(p.get("command") or p.get("comm") or "—")
            if len(cmd_text) > 52:
                cmd_text = cmd_text[:49] + "..."
            workers = int(p.get("workers") or 0)
            if workers > 0:
                cmd_text = f"[bold cyan][+{workers} workers][/bold cyan] {cmd_text}"

            proc_tbl.add_row(
                str(p.get("pid", "—")),
                st_cell,
                f"{float(p.get('cpu_pct', 0.0)):.1f}%",
                ram_str,
                vram_str,
                str(p.get("elapsed", "—")),
                cmd_text,
            )
        renderables.append(proc_tbl)

    return Panel(
        Group(*renderables),
        title=f"[bold cyan]⚡ Live Worker Telemetry — {host_short} (Job {job_target})[/bold cyan]",
        border_style="cyan",
        expand=False,
    )


# ---------------------------------------------------------------------------
# 5-Second Time-Series CSV Telemetry & Sparkline / Matplotlib Plotting
# ---------------------------------------------------------------------------
def read_telemetry_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """Reads a 25-column `repro/telemetry_<proc>.csv` into a list of typed sample dicts."""
    if not csv_path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as fh:
            for raw in csv.DictReader(fh):
                parsed: Dict[str, Any] = {"timestamp_iso": str(raw.get("timestamp_iso", "") or "")}
                for k, v in raw.items():
                    if k == "timestamp_iso" or k is None:
                        continue
                    try:
                        parsed[k] = float(v or 0.0)
                    except (TypeError, ValueError):
                        parsed[k] = 0.0
                rows.append(parsed)
    except OSError:
        return []
    return rows


def compute_telemetry_summary(
    csv_path: Path, rows: Optional[List[Dict[str, Any]]] = None
) -> Optional[Dict[str, Any]]:
    """Computes compact summary statistics from a telemetry CSV file or pre-parsed rows."""
    t_rows = rows if rows is not None else read_telemetry_csv(csv_path)
    if not t_rows:
        return None
    n_samples = len(t_rows)
    elapsed_vals = [float(r.get("elapsed_s", 0.0) or 0.0) for r in t_rows]
    dts = [
        (elapsed_vals[1] - el if len(elapsed_vals) > 1 else 5.0)
        if i == 0
        else max(0.0, el - elapsed_vals[i - 1])
        for i, el in enumerate(elapsed_vals)
    ]

    def _col(name: str) -> List[float]:
        return [float(r.get(name, 0.0) or 0.0) for r in t_rows]

    cpu_tot_v = _col("cpu_total_pct")
    cpu_sys_v = _col("cpu_sys_pct")
    rss_v = _col("rss_mb")
    pss_v = _col("pss_mb")
    iow_v = _col("io_wait_procs")
    ior_v = _col("io_read_mb_s")
    iowr_v = _col("io_write_mb_s")
    dr_v = _col("disk_read_mb_s")
    dw_v = _col("disk_write_mb_s")
    gu_v = _col("gpu_util_pct")
    gbw_v = _col("gpu_mem_bw_pct")
    vu_v = _col("vram_used_mb")
    vt_v = _col("vram_total_mb")
    gt_v = _col("gpu_temp_c")
    gp_v = _col("gpu_power_w")
    gc_v = _col("gpu_sm_clock_mhz")
    sorted_gu = sorted(gu_v)
    return {
        "samples": n_samples,
        "cpu_pct_avg": round(sum(cpu_tot_v) / n_samples, 1),
        "cpu_pct_max": round(max(cpu_tot_v), 1),
        "cpu_sys_pct_avg": round(sum(cpu_sys_v) / n_samples, 1),
        "rss_mb_avg": round(sum(rss_v) / n_samples, 2),
        "rss_mb_max": round(max(rss_v), 2),
        "pss_mb_max": round(max(pss_v), 2),
        "io_wait_pct": round(sum(1 for x in iow_v if x > 0) / n_samples * 100.0, 1),
        "io_read_mb_total": round(sum(v * dt for v, dt in zip(ior_v, dts)), 2),
        "io_write_mb_total": round(sum(v * dt for v, dt in zip(iowr_v, dts)), 2),
        "disk_read_mb_total": round(sum(v * dt for v, dt in zip(dr_v, dts)), 2),
        "disk_write_mb_total": round(sum(v * dt for v, dt in zip(dw_v, dts)), 2),
        "gpu_util_avg": round(sum(gu_v) / n_samples, 1),
        "gpu_util_p50": round(sorted_gu[n_samples // 2], 1),
        "gpu_util_max": round(max(gu_v), 1),
        "gpu_mem_bw_avg": round(sum(gbw_v) / n_samples, 1),
        "vram_mb_avg": round(sum(vu_v) / n_samples, 1),
        "vram_mb_max": round(max(vu_v), 1),
        "vram_total_mb": round(max(vt_v), 1),
        "gpu_temp_max_c": round(max(gt_v), 1),
        "gpu_power_avg_w": round(sum(gp_v) / n_samples, 1),
        "gpu_power_max_w": round(max(gp_v), 1),
        "gpu_clock_min_mhz": round(min(gc_v), 1),
        "gpu_clock_max_mhz": round(max(gc_v), 1),
    }


_SPARK_BLOCKS = " ▁▂▃▄▅▆▇█"


def _bin_series(values: List[float], width: int, agg: str = "mean") -> List[float]:
    if not values:
        return []
    n = len(values)
    w = max(1, min(width, n))
    if w == n:
        return list(values)
    out: List[float] = []
    for i in range(w):
        lo = int(i * n / w)
        hi = max(lo + 1, int((i + 1) * n / w))
        chunk = values[lo:hi]
        out.append(max(chunk) if agg == "max" else sum(chunk) / len(chunk))
    return out


def _render_sparkline_rows(
    values: List[float],
    width: int,
    *,
    height: int = 2,
    max_val: Optional[float] = None,
    color: str = "green",
) -> List[str]:
    binned = _bin_series(values, width)
    if not binned:
        return [""] * height
    hi = max_val if (max_val is not None and max_val > 0) else max(binned)
    if hi <= 0:
        hi = 1.0
    total_levels = height * 8
    rows_out: List[str] = []
    for row_idx in range(height - 1, -1, -1):
        row_chars: List[str] = []
        base_lvl = row_idx * 8
        for v in binned:
            norm = max(0.0, min(1.0, v / hi))
            lvl = int(round(norm * total_levels))
            if v > 0 and lvl == 0:
                lvl = 1
            cell_lvl = max(0, min(8, lvl - base_lvl))
            row_chars.append(_SPARK_BLOCKS[cell_lvl])
        rows_out.append(f"[{color}]{''.join(row_chars)}[/{color}]")
    return rows_out


def _series_stats(values: List[float]) -> Tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    s = sorted(values)
    p95 = s[min(len(s) - 1, int(0.95 * (len(s) - 1)))]
    return s[0], sum(values) / len(values), p95, s[-1]


def _diagnose_telemetry(summary: Dict[str, Any], is_gpu: bool) -> List[str]:
    diags: List[str] = []
    gpu_avg = float(summary.get("gpu_util_avg", 0.0) or 0.0)
    gpu_bw = float(summary.get("gpu_mem_bw_avg", 0.0) or 0.0)
    vram_max = float(summary.get("vram_mb_max", 0.0) or 0.0)
    vram_tot = float(summary.get("vram_total_mb", 0.0) or 0.0)
    g_temp = float(summary.get("gpu_temp_max_c", 0.0) or 0.0)
    cpu_avg = float(summary.get("cpu_pct_avg", 0.0) or 0.0)
    cpu_sys = float(summary.get("cpu_sys_pct_avg", 0.0) or 0.0)
    iow_pct = float(summary.get("io_wait_pct", 0.0) or 0.0)

    if is_gpu and gpu_avg < 50.0 and (iow_pct > 15.0 or cpu_avg > 90.0):
        diags.append(
            f"[bold yellow]⚠ GPU Starvation Detected:[/bold yellow] GPU averaged {gpu_avg:.1f}% utilization "
            f"while workers spent {iow_pct:.1f}% of samples in Disk/CephFS I/O wait (CPU avg {cpu_avg:.1f}%). "
            f"Consider increasing DataLoader num_workers, caching dataset in RAM/scratch, or pre-packing files."
        )
    if is_gpu and gpu_avg < 60.0 and gpu_bw > 70.0:
        diags.append(
            f"[bold cyan]ℹ Memory-Bandwidth Bound:[/bold cyan] High GPU memory controller load ({gpu_bw:.1f}%) "
            f"with {gpu_avg:.1f}% compute utilization. Consider mixed precision (bf16/fp16) or torch.compile operator fusion."
        )
    if is_gpu and vram_tot > 0:
        vram_pct = vram_max / vram_tot
        if vram_pct > 0.92:
            diags.append(
                f"[bold red]⚠ VRAM OOM Proximity:[/bold red] Peak VRAM reached {vram_max / 1024:.2f}/{vram_tot / 1024:.1f} GB "
                f"({vram_pct * 100:.0f}%). Reduce batch size or enable gradient checkpointing to avoid OOM."
            )
        elif vram_pct < 0.35 and gpu_avg > 20.0:
            diags.append(
                f"[cyan]ℹ VRAM Headroom Available:[/cyan] Peak VRAM only used {vram_pct * 100:.0f}% "
                f"({vram_max / 1024:.2f}/{vram_tot / 1024:.1f} GB). Batch size could potentially be increased."
            )
    if is_gpu and g_temp >= 83.0:
        diags.append(
            f"[bold red]⚠ Thermal / Clock Throttling Risk:[/bold red] GPU reached {g_temp:.0f}°C."
        )
    if cpu_avg > 50.0 and cpu_sys > 0.35 * cpu_avg:
        diags.append(
            f"[bold yellow]⚠ Kernel / Syscall Contention:[/bold yellow] System CPU averaged {cpu_sys:.1f}% "
            f"(>35% of {cpu_avg:.1f}% total CPU), often caused by thread oversubscription or small-file CephFS metadata storms."
        )
    if not diags:
        diags.append("[bold green]✔ Healthy resource utilization profile.[/bold green]")
    return diags


def _build_telemetry_plot_panel(
    run_dir: Path,
    data: Dict[str, Any],
    proc_id: int,
    rows: List[Dict[str, Any]],
    width: int = 60,
) -> Panel:
    from rich.console import Group

    summary = compute_telemetry_summary(run_dir / "repro" / f"telemetry_{proc_id}.csv", rows=rows) or {}
    arch = str((data.get("resources_requested") or {}).get("architecture") or "")
    vtot = float(summary.get("vram_total_mb", 0.0) or 0.0)
    gmax = float(summary.get("gpu_util_max", 0.0) or 0.0)
    is_gpu = vtot > 0 or gmax > 0 or arch in ("GPU3090", "A100")

    t_start = str(rows[0].get("timestamp_iso", "") or "")[:19].replace("T", " ")
    t_end = str(rows[-1].get("timestamp_iso", "") or "")[:19].replace("T", " ")
    dur_s = max(0.0, float(rows[-1].get("elapsed_s", 0.0) or 0.0) - float(rows[0].get("elapsed_s", 0.0) or 0.0))
    n_samples = len(rows)
    cadence = round(dur_s / max(1, n_samples - 1), 1) if n_samples > 1 else 5.0
    header_line = (
        f"[bold white]{t_start}[/bold white] [dim]→[/dim] [bold white]{t_end}[/bold white]  "
        f"[dim]│[/dim]  Duration: [cyan]{format_duration(int(dur_s))}[/cyan]  "
        f"[dim]│[/dim]  [bold]{n_samples}[/bold] samples @ ~{cadence:g}s"
    )

    chart_tbl = Table(box=None, show_header=True, header_style="bold dim", pad_edge=False)
    chart_tbl.add_column("Metric", style="bold cyan", width=18, no_wrap=True)
    chart_tbl.add_column("Time Series", no_wrap=True)
    chart_tbl.add_column("Min │ Avg │ P95 │ Max", justify="right", style="white", no_wrap=True)

    def _add_chart(
        label: str,
        vals: List[float],
        color: str,
        *,
        max_val: Optional[float] = None,
        unit: str = "",
        extra_note: str = "",
    ) -> None:
        mn, av, p95, mx = _series_stats(vals)
        sp_rows = _render_sparkline_rows(vals, width, height=2, max_val=max_val, color=color)
        stats_str = f"{mn:.1f} │ [bold]{av:.1f}[/bold] │ {p95:.1f} │ {mx:.1f}{unit}"
        if extra_note:
            stats_str += f"\n[dim]{extra_note}[/dim]"
        chart_tbl.add_row(label, "\n".join(sp_rows), stats_str)

    if is_gpu:
        _add_chart("GPU Compute (%)", [r.get("gpu_util_pct", 0.0) for r in rows], "green", max_val=100.0, unit="%")
        _add_chart("GPU Mem BW (%)", [r.get("gpu_mem_bw_pct", 0.0) for r in rows], "cyan", max_val=100.0, unit="%")
        vram_gb = [r.get("vram_used_mb", 0.0) / 1024.0 for r in rows]
        vtot_gb = vtot / 1024.0 if vtot > 0 else None
        peak_pct_note = f"Peak {max(vram_gb) / vtot_gb * 100:.0f}% of {vtot_gb:.1f} GB" if vtot_gb else ""
        _add_chart("GPU VRAM (GB)", vram_gb, "magenta", max_val=vtot_gb, unit="G", extra_note=peak_pct_note)
        temps = [r.get("gpu_temp_c", 0.0) for r in rows]
        pwrs = [r.get("gpu_power_w", 0.0) for r in rows]
        clks = [r.get("gpu_sm_clock_mhz", 0.0) for r in rows]
        _add_chart(
            "GPU Power (W)",
            pwrs,
            "yellow",
            unit="W",
            extra_note=f"Temp max {max(temps):.0f}°C · SM {min(clks):.0f}-{max(clks):.0f} MHz",
        )

    cpu_tot = [r.get("cpu_total_pct", 0.0) for r in rows]
    cpu_usr_avg = sum(r.get("cpu_user_pct", 0.0) for r in rows) / n_samples
    cpu_sys_avg = sum(r.get("cpu_sys_pct", 0.0) for r in rows) / n_samples
    _add_chart(
        "CPU Usage (%)",
        cpu_tot,
        "blue",
        unit="%",
        extra_note=f"usr {cpu_usr_avg:.1f}% / sys {cpu_sys_avg:.1f}%",
    )

    pss_gb = [r.get("pss_mb", 0.0) / 1024.0 for r in rows]
    rss_max_gb = max(r.get("rss_mb", 0.0) for r in rows) / 1024.0
    avail_gb = rows[-1].get("node_ram_avail_mb", 0.0) / 1024.0
    _add_chart(
        "Host RAM PSS (GB)",
        pss_gb,
        "bright_cyan",
        unit="G",
        extra_note=f"RSS max {rss_max_gb:.2f}G · Node free {avail_gb:.1f}G",
    )

    disk_tot = [r.get("disk_read_mb_s", 0.0) + r.get("disk_write_mb_s", 0.0) for r in rows]
    iow_pct = float(summary.get("io_wait_pct", 0.0) or 0.0)
    _add_chart(
        "Disk I/O (MB/s)",
        disk_tot,
        "yellow",
        unit="M/s",
        extra_note=f"D-state wait {iow_pct:.1f}% · R {summary.get('disk_read_mb_total', 0.0):.0f}M / W {summary.get('disk_write_mb_total', 0.0):.0f}M",
    )

    diags = _diagnose_telemetry(summary, is_gpu)
    diag_panel = Panel(
        "\n".join(diags),
        title="[bold cyan]🩺 Automated Bottleneck Diagnosis[/bold cyan]",
        border_style="dim",
        expand=True,
    )

    status_str = str(data.get("_live_badge") or data.get("status", "UNKNOWN"))
    return Panel(
        Group(header_line, "", chart_tbl, "", diag_panel),
        title=f"[bold cyan]📈 Telemetry — {format_run_short(run_dir, data)} [proc {proc_id}] ({style_status(status_str)})[/bold cyan]",
        border_style="cyan",
        expand=False,
    )


def _find_matplotlib_python(data: Dict[str, Any], runs_root: Path) -> Optional[str]:
    candidates: List[str] = [sys.executable]
    sub_venv = (data.get("submission") or {}).get("venv")
    if sub_venv:
        vp = Path(str(sub_venv))
        for py in (vp / "bin" / "python", vp / "Scripts" / "python.exe"):
            if py.is_file():
                candidates.append(str(py))
    env_venv = os.environ.get("VIRTUAL_ENV")
    if env_venv:
        vp = Path(env_venv)
        for py in (vp / "bin" / "python", vp / "Scripts" / "python.exe"):
            if py.is_file():
                candidates.append(str(py))
    proj_venv = discover_project_venv(runs_root.parent)
    if proj_venv:
        for py in (proj_venv / "bin" / "python", proj_venv / "Scripts" / "python.exe"):
            if py.is_file():
                candidates.append(str(py))
    sub_py = resolve_submit_python()
    if sub_py:
        candidates.append(str(sub_py))

    seen = set()
    for py in candidates:
        if not py or py in seen:
            continue
        seen.add(py)
        try:
            chk = subprocess.run(
                [py, "-c", "import matplotlib"],
                capture_output=True,
                timeout=5,
            )
            if chk.returncode == 0:
                return py
        except Exception:
            continue
    return None


def _save_telemetry_matplotlib(
    csv_path: Path,
    out_path: Path,
    title: str,
    is_gpu: bool,
    data: Dict[str, Any],
    runs_root: Path,
) -> bool:
    py_exe = _find_matplotlib_python(data, runs_root)
    if not py_exe:
        console.print(
            "[bold red]✘ Could not find `matplotlib` in the current or project Python environment.[/bold red]\n"
            "  Install it (`uv pip install matplotlib`) to enable `--save`."
        )
        return False

    plot_script = """
import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv_path, out_path, title, is_gpu_str = sys.argv[1:5]
is_gpu = is_gpu_str == "1"
with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as fh:
    rows = list(csv.DictReader(fh))
if not rows:
    sys.exit(2)

def col(k):
    return [float(r.get(k, 0.0) or 0.0) for r in rows]

t_min = [x / 60.0 for x in col("elapsed_s")]
n_panels = 5 if is_gpu else 3
fig, axes = plt.subplots(n_panels, 1, figsize=(12, 10), dpi=200, sharex=True)
fig.suptitle(title, fontsize=13, fontweight="bold")
ax_i = 0

if is_gpu:
    ax = axes[ax_i]; ax_i += 1
    ax.plot(t_min, col("gpu_util_pct"), label="GPU Compute %", color="#2ca02c", lw=1.6)
    ax.plot(t_min, col("gpu_mem_bw_pct"), label="GPU Mem BW %", color="#17becf", lw=1.2, alpha=0.85)
    ax.set_ylim(0, 105)
    ax.set_ylabel("GPU %")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)

ax = axes[ax_i]; ax_i += 1
if is_gpu:
    vram_gb = [x / 1024.0 for x in col("vram_used_mb")]
    vtot_gb = max(col("vram_total_mb")) / 1024.0
    ax.plot(t_min, vram_gb, label="VRAM Used (GB)", color="#9467bd", lw=1.6)
    if vtot_gb > 0:
        ax.axhline(vtot_gb, color="#9467bd", ls="--", lw=1.0, alpha=0.6, label=f"VRAM Limit ({vtot_gb:.1f} GB)")
ax.plot(t_min, [x / 1024.0 for x in col("pss_mb")], label="Host PSS (GB)", color="#1f77b4", lw=1.4)
ax.plot(t_min, [x / 1024.0 for x in col("rss_mb")], label="Host RSS (GB)", color="#aec7e8", lw=1.0, ls=":")
ax.set_ylabel("Memory (GB)")
ax.grid(True, alpha=0.25)
ax.legend(loc="upper right", fontsize=8)

ax = axes[ax_i]; ax_i += 1
ax.plot(t_min, col("cpu_total_pct"), label="CPU Total %", color="#1f77b4", lw=1.5)
ax.plot(t_min, col("cpu_sys_pct"), label="CPU Sys %", color="#ff7f0e", lw=1.2, alpha=0.85)
ax2 = ax.twinx()
ax2.plot(t_min, col("node_load_1m"), label="Load 1m", color="#7f7f7f", lw=1.0, ls="--", alpha=0.7)
ax.set_ylabel("CPU %")
ax2.set_ylabel("Load 1m")
ax.grid(True, alpha=0.25)
ax.legend(loc="upper left", fontsize=8)
ax2.legend(loc="upper right", fontsize=8)

ax = axes[ax_i]; ax_i += 1
ax.plot(t_min, col("disk_read_mb_s"), label="Disk Read (MB/s)", color="#2ca02c", lw=1.4)
ax.plot(t_min, col("disk_write_mb_s"), label="Disk Write (MB/s)", color="#d62728", lw=1.2)
iow = col("io_wait_procs")
for idx_i, w in enumerate(iow):
    if w > 0 and idx_i + 1 < len(t_min):
        ax.axvspan(t_min[idx_i], t_min[idx_i + 1], color="#ff9896", alpha=0.25)
ax.set_ylabel("Disk I/O (MB/s)")
ax.grid(True, alpha=0.25)
ax.legend(loc="upper right", fontsize=8)

if is_gpu:
    ax = axes[ax_i]; ax_i += 1
    ax.plot(t_min, col("gpu_power_w"), label="Power (W)", color="#ff7f0e", lw=1.4)
    ax.plot(t_min, col("gpu_temp_c"), label="Temp (°C)", color="#d62728", lw=1.2)
    ax2 = ax.twinx()
    ax2.plot(t_min, col("gpu_sm_clock_mhz"), label="SM Clock (MHz)", color="#8c564b", lw=1.0, alpha=0.8)
    ax.set_ylabel("Power / Temp")
    ax2.set_ylabel("Clock (MHz)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

axes[-1].set_xlabel("Elapsed Time (minutes)")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(out_path, dpi=200)
"""
    with status_spinner(f"Rendering high-res telemetry figure ({out_path.name})..."):
        res = subprocess.run(
            [py_exe, "-c", plot_script, str(csv_path), str(out_path), title, "1" if is_gpu else "0"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
    if res.returncode != 0:
        console.print(f"[bold red]✘ Matplotlib export failed:[/bold red] {(res.stderr or res.stdout).strip()}")
        return False
    console.print(f"[bold green]✔ Saved telemetry figure:[/bold green] {out_path.resolve()}")
    return True
