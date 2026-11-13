"""
send_cli.py — HTCondor Job Submission & Provenance CLI entrypoint (`send`).
"""

from __future__ import annotations

import argparse
import datetime
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich import box
from rich.panel import Panel
from rich.table import Table

from hermes_tools.condor import (
    ARCH_SPECS,
    build_send_command_tokens,
    format_condor_arg_line,
    generate_bash_wrapper,
    generate_dag_file,
    generate_dagman_config,
    generate_sub_file,
    generate_wait_parent_script,
    get_all_cluster_ids,
    query_condor_nodes,
    query_live_condor_clusters,
    reconcile_run,
    run_needs_live_reconcile,
    sanitize_dagman_condor_sub,
    summarize_condor_pools,
)
from hermes_tools.core import (
    atomic_write_json,
    console,
    discover_project_venv,
    expand_arg_sweeps,
    get_hostname,
    get_prompt_style,
    get_runs_root,
    get_username,
    is_valid_memory_input,
    load_project_config,
    load_run_json,
    locked_mutate_run_json,
    parse_memory_to_mb,
    resolve_submit_python,
    sanitize_tag,
    update_latest_symlink,
    write_text_lf,
)
from hermes_tools.snapshot import (
    clone_code_snapshot,
    create_code_snapshot,
    freeze_python_env,
    get_git_info,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="send",
        description="Definitive HTCondor job launcher & provenance manager for Hermes 2 (I3A).",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  send train.py                                      # Launch interactive TUI wizard\n"
            "  send train.py -a GPU3090 -- --epochs 100 --lr 1e-4 # Direct RTX 3090 submission\n"
            "  send train.py -a A100 -g 2 -m 120G                 # Dual A100 with 120 GB RAM\n"
            "  send sim.py -a AMD -c 16 -m 32G -t baseline        # 16-core AMD CPU job with tag\n"
            "  send train.py -a GPU3090 -s '--fold {0..4}'        # 5-job parallel array sweep"
        ),
    )
    parser.add_argument(
        "target", nargs="?", help="Path to Python script (.py), Bash script (.sh), or binary"
    )
    parser.add_argument(
        "-a", "--arch", choices=list(ARCH_SPECS.keys()), help="Target Hermes 2 +Architecture pool"
    )
    parser.add_argument(
        "-c", "--cpus", type=int, help="Number of CPU cores (ignored on GPU3090/A100)"
    )
    parser.add_argument(
        "-m",
        "--mem",
        type=str,
        help="RAM request, e.g., '8G', '32GB', '4096M' (ignored on GPU3090)",
    )
    parser.add_argument(
        "-g", "--gpus", type=int, choices=[1, 2, 3, 4], help="Number of GPUs (for A100: 1-4)"
    )
    parser.add_argument("--args", type=str, default="", help="Arguments to pass to your script")
    parser.add_argument(
        "-s",
        "--sweep",
        action="append",
        default=[],
        help="Queue parallel parameter sweep (supports '{0..4}', '{00..09}', '{a,b}', ';' or repeatable -s)",
    )
    parser.add_argument(
        "--sweep-file", type=str, help="File containing one argument string per line for sweep"
    )
    parser.add_argument(
        "-j",
        "--max-parallel",
        type=int,
        default=None,
        help="Limit concurrent running sweep jobs (e.g. -j 1 or -j 2; emits max_idle = K)",
    )
    parser.add_argument(
        "--chain",
        "--seq",
        action="store_true",
        help="Split sweep into separate sequential runs chained via --after (one run per sweep item)",
    )
    parser.add_argument(
        "-t",
        "--tag",
        action="append",
        default=[],
        help="Optional tag(s) for run directory (repeatable or '{a,b,c}' / 'a,b,c' for --chain)",
    )
    parser.add_argument(
        "--no-tag",
        action="store_true",
        help="Do not auto-derive tags from sweep arguments when using --chain",
    )
    parser.add_argument(
        "--venv", type=str, help="Explicit path to virtual environment (auto-detected by default)"
    )
    parser.add_argument(
        "--no-venv", action="store_true", help="Run without activating a Python virtual environment"
    )
    parser.add_argument(
        "--node", action="append", default=[], help="Target specific node(s), e.g., nv3090-01"
    )
    parser.add_argument(
        "--exclude", action="append", default=[], help="Exclude specific node(s), e.g., her07-100"
    )
    parser.add_argument(
        "--project", type=str, help="I3A +Proyecto accounting code (e.g., '100-100')"
    )
    parser.add_argument(
        "--scratch",
        nargs="+",
        default=[],
        help="Enable RAMDisk scratch mode and transfer specified input files",
    )
    parser.add_argument(
        "-e",
        "--env",
        action="append",
        default=[],
        help="Export custom environment variable(s) inside job.sh, e.g. -e WANDB_MODE=offline",
    )
    parser.add_argument(
        "--after",
        type=str,
        help="Queue run as a dependent job after specified run (by ID, index, or cluster ID) completes",
    )
    parser.add_argument(
        "--after-ok",
        action="store_true",
        help="When used with --after, only run if parent succeeds (exit 0); cancel if parent fails",
    )
    parser.add_argument(
        "--from-run",
        type=str,
        help="Clone frozen repro/code/ snapshot and provenance from an existing run",
    )
    parser.add_argument(
        "-f",
        "--follow",
        action="store_true",
        help="Wait for job to start and follow live stdout until completion",
    )
    parser.add_argument(
        "-y", "--yes", action="store_true", help="Skip confirmation prompt and submit immediately"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate run folder and scripts without submitting",
    )
    return parser


def _as_str_list(val: Any) -> List[str]:
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, str) and val.strip():
        return [val.strip()]
    return []


def _resolve_target_and_venv(
    args: argparse.Namespace,
    proj_cfg: Dict[str, Any],
    cwd: Path,
    source_run_dir: Optional[Path] = None,
    source_data: Optional[Dict[str, Any]] = None,
) -> tuple[Path, str, Optional[Path], bool, bool, bool]:
    """Resolves the target executable and virtual environment, returning (target_path, exec_type, venv_path, explicit_venv_used, no_venv_flag, interactive_mode)."""
    interactive_mode = False
    src_sub = (source_data.get("submission") or {}) if source_data else {}
    target_str = args.target or (src_sub.get("target") if source_run_dir else None) or proj_cfg.get("target")
    if not target_str:
        import questionary

        interactive_mode = True
        candidates = sorted(
            [
                p.name
                for p in cwd.iterdir()
                if p.is_file()
                and (p.suffix in (".py", ".sh") or os.access(p, os.X_OK))
                and not p.name.startswith(".")
            ]
        )
        if candidates:
            choices = candidates + ["[Enter custom path...]"]
            selected = questionary.select(
                "Select script or executable to submit:",
                choices=choices,
                style=get_prompt_style(),
            ).ask()
            if selected is None:
                sys.exit(0)
            target_str = (
                questionary.path("Path to executable:", style=get_prompt_style()).ask()
                if selected == "[Enter custom path...]"
                else selected
            )
        else:
            target_str = questionary.path(
                "Path to script or executable:", style=get_prompt_style()
            ).ask()

        if not target_str:
            sys.exit(0)

    if source_run_dir is not None:
        src_code = source_run_dir / "repro" / "code"
        snap_rel = src_sub.get("snapshot_executable")
        cands = [src_code / str(target_str)]
        if snap_rel:
            cands.append(source_run_dir / str(snap_rel))
        cands.append(src_code / Path(str(target_str)).name)
        found = next((c for c in cands if c.is_file()), None)
        if not found:
            console.print(
                f"[bold red]✘ Error:[/bold red] Target '{target_str}' was not found inside frozen snapshot {src_code}."
            )
            sys.exit(1)
        target_path = found
    else:
        target_path = Path(str(target_str)).expanduser().resolve()
        if not target_path.is_file():
            console.print(f"[bold red]✘ Error:[/bold red] Target file '{target_str}' does not exist.")
            sys.exit(1)

    exec_type = (
        "python"
        if target_path.suffix == ".py"
        else ("bash" if target_path.suffix == ".sh" else "binary")
    )

    src_venv = src_sub.get("venv") if source_run_dir else None
    if args.venv is not None:
        venv_arg = args.venv
    elif source_run_dir and not args.no_venv and src_venv and Path(str(src_venv)).is_dir():
        venv_arg = str(src_venv)
    else:
        venv_arg = proj_cfg.get("venv")
    no_venv_flag = bool(
        args.no_venv
        or (src_sub.get("no_venv", False) if (source_run_dir and args.venv is None) else False)
        or proj_cfg.get("no_venv", False)
    )
    venv_path: Optional[Path] = None
    explicit_venv_used = bool(venv_arg)
    if not no_venv_flag:
        venv_path = discover_project_venv(cwd, str(venv_arg) if venv_arg else None)
        if exec_type == "python" and not venv_path:
            console.print(
                "[yellow]⚠ No local Python virtual environment (.venv/venv) detected.[/yellow]"
            )
            if interactive_mode and sys.stdin.isatty():
                import questionary

                custom_venv = questionary.path(
                    "Enter path to venv (or leave empty to use system Python):",
                    style=get_prompt_style(),
                ).ask()
                if custom_venv is None:
                    sys.exit(0)
                if custom_venv.strip():
                    venv_path = discover_project_venv(cwd, custom_venv.strip())
                    explicit_venv_used = True

    return target_path, exec_type, venv_path, explicit_venv_used, no_venv_flag, interactive_mode


def _resolve_hardware_resources(
    args: argparse.Namespace,
    proj_cfg: Dict[str, Any],
    interactive_mode: bool,
) -> tuple[str, Optional[int], Optional[int], Optional[int], bool]:
    """Resolves architecture, CPUs, RAM (MB), and GPUs, returning (arch, cpus, mem_mb, gpus, interactive_mode)."""
    cfg_arch = str(proj_cfg.get("arch", "")).strip().upper() or None
    if cfg_arch and cfg_arch not in ARCH_SPECS:
        console.print(
            f"[bold red]✘ Error:[/bold red] Invalid arch '{cfg_arch}' in project config (must be one of {list(ARCH_SPECS.keys())})."
        )
        sys.exit(1)

    cpus_arg = args.cpus if args.cpus is not None else proj_cfg.get("cpus")
    if cpus_arg is not None and int(cpus_arg) <= 0:
        console.print("[bold red]✘ Error:[/bold red] --cpus must be a positive integer.")
        sys.exit(1)

    mem_arg = args.mem if args.mem is not None else proj_cfg.get("mem")
    gpus_arg = args.gpus if args.gpus is not None else proj_cfg.get("gpus")

    arch = args.arch or (cfg_arch if not interactive_mode else None)
    if not arch:
        import questionary

        interactive_mode = True
        nodes = query_condor_nodes()
        pool_map = summarize_condor_pools(nodes) if nodes else {}
        arch_choices = []
        for k, spec in ARCH_SPECS.items():
            base_title = str(spec["title"])
            p_info = pool_map.get(k)
            if p_info:
                if k in ("GPU3090", "A100"):
                    badge = f" [{p_info['gpus_free']}/{p_info['gpus_total']} GPUs free]"
                else:
                    badge = f" [{p_info['cpus_free']}/{p_info['cpus_total']} CPUs free]"
                title = f"{base_title}{badge}"
            else:
                title = base_title
            arch_choices.append(questionary.Choice(title=title, value=k))

        arch = questionary.select(
            "Select Hermes 2 Target Architecture (+Architecture):",
            choices=arch_choices,
            default=cfg_arch if cfg_arch in ARCH_SPECS else None,
            style=get_prompt_style(),
        ).ask()
        if not arch:
            sys.exit(0)

    cpus: Optional[int] = int(cpus_arg) if cpus_arg is not None else None
    mem_mb: Optional[int] = None
    if mem_arg:
        try:
            mem_mb = parse_memory_to_mb(str(mem_arg))
        except ValueError as exc:
            console.print(f"[bold red]✘ Invalid --mem value '{mem_arg}':[/bold red] {exc}")
            sys.exit(1)

    gpus: Optional[int] = int(gpus_arg) if gpus_arg is not None else None

    if arch == "GPU3090":
        if args.cpus is not None or args.mem is not None or args.gpus is not None:
            console.print(
                "[dim cyan]ℹ Note: GPU3090 locks the entire node automatically; omitting request_cpus/request_memory/request_gpus.[/dim cyan]"
            )
        cpus, mem_mb, gpus = None, None, None
    elif arch == "A100":
        if args.cpus is not None:
            console.print(
                "[dim cyan]ℹ Note: CPU count is managed automatically on A100 nodes; omitting request_cpus.[/dim cyan]"
            )
        cpus = None
        if gpus is None:
            if interactive_mode:
                import questionary

                g_ans = questionary.select(
                    "Number of NVIDIA A100 (40GB) GPUs to request:",
                    choices=["1", "2", "3", "4"],
                    default="1",
                    style=get_prompt_style(),
                ).ask()
                if not g_ans:
                    sys.exit(0)
                gpus = int(g_ans)
            else:
                gpus = 1

        if mem_mb is None and interactive_mode:
            import questionary

            m_ans = questionary.text(
                "System RAM request (Leave blank for default <=64G, or e.g. '120G'):",
                default="",
                validate=lambda v: is_valid_memory_input(v, allow_empty=True)
                or "Enter valid size (e.g. '120G', '96GB', '98304M') or leave blank",
                style=get_prompt_style(),
            ).ask()
            if m_ans is None:
                sys.exit(0)
            if m_ans.strip():
                mem_mb = parse_memory_to_mb(m_ans.strip())

        if mem_mb is not None and mem_mb <= 65536:
            if args.mem is not None:
                console.print(
                    "[dim cyan]ℹ Note: A100 nodes allocate up to 64 GB RAM by default; omitting request_memory (<= 64 GB).[/dim cyan]"
                )
            mem_mb = None
    else:
        if args.gpus is not None:
            console.print(
                f"[dim cyan]ℹ Note: Ignoring --gpus on CPU architecture '{arch}'.[/dim cyan]"
            )
        gpus = None

        if cpus is None:
            if interactive_mode:
                import questionary

                c_ans = questionary.text(
                    "Number of CPU cores (request_cpus):",
                    default="4",
                    validate=lambda v: (v.isdigit() and int(v) > 0) or "Enter a positive integer",
                    style=get_prompt_style(),
                ).ask()
                if not c_ans:
                    sys.exit(0)
                cpus = int(c_ans)
            else:
                cpus = 4

        if mem_mb is None:
            if interactive_mode:
                import questionary

                m_ans = questionary.text(
                    "System RAM (e.g., '8G', '16G', '4096M'):",
                    default="8G",
                    validate=lambda v: is_valid_memory_input(v, allow_empty=False)
                    or "Enter valid size (e.g. '8G', '32GB', '4096M')",
                    style=get_prompt_style(),
                ).ask()
                if not m_ans:
                    sys.exit(0)
                mem_mb = parse_memory_to_mb(m_ans)
            else:
                mem_mb = 8192

    return arch, cpus, mem_mb, gpus, interactive_mode


def _derive_sweep_item_tag(base_tag: str, raw_sweep_item: str, idx: int) -> str:
    """Derives a short, readable tag for a single chained sweep item."""
    raw = (raw_sweep_item or "").strip()
    try:
        tokens = shlex.split(raw) if raw else []
    except ValueError:
        tokens = raw.split() if raw else []

    pairs: List[Tuple[Optional[str], str]] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("-") and len(tok) > 1:
            if "=" in tok:
                k, v = tok.split("=", 1)
                pairs.append((k.lstrip("-"), v))
                i += 1
            elif i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                pairs.append((tok.lstrip("-"), tokens[i + 1]))
                i += 2
            else:
                pairs.append((None, tok.lstrip("-")))
                i += 1
        else:
            pairs.append((None, tok))
            i += 1

    item_tag = ""
    if len(pairs) == 1:
        k, v = pairs[0]
        if k and re.fullmatch(r"-?\d+(?:\.\d+)?", v):
            item_tag = sanitize_tag(f"{k}_{v}")
        else:
            item_tag = sanitize_tag(v)
    elif pairs:
        parts = [f"{k}_{v}" if k else v for k, v in pairs]
        item_tag = sanitize_tag("_".join(parts))[:24].strip("_-")

    if not item_tag:
        item_tag = f"s{idx}"

    return f"{base_tag}_{item_tag}" if base_tag else item_tag


def _resolve_sweeps_and_tag(
    args: argparse.Namespace,
    proj_cfg: Dict[str, Any],
    passthrough_args: str,
    interactive_mode: bool,
) -> tuple[str, List[str], List[str], List[str], str, List[str]]:
    """Resolves single/sweep arguments and run tag(s), returning (combined_single_args, sweep_list, raw_sweep_items, cli_sweep_specs, base_tag, tag_list)."""
    combined_single_args = " ".join(
        x for x in [args.args.strip(), passthrough_args.strip()] if x
    ).strip()

    raw_sweep_inputs: List[str] = list(args.sweep)
    cli_sweep_specs: List[str] = list(args.sweep)
    if args.sweep_file:
        sf_path = Path(args.sweep_file).expanduser().resolve()
        if not sf_path.is_file():
            console.print(f"[bold red]✘ Sweep file not found:[/bold red] {sf_path}")
            sys.exit(1)
        raw_sweep_inputs.extend(
            [
                line.strip()
                for line in sf_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
        )

    expanded_sweeps: List[str] = []
    for item in raw_sweep_inputs:
        expanded_sweeps.extend(expand_arg_sweeps(item))

    if expanded_sweeps:
        raw_sweep_items = list(expanded_sweeps)
        sweep_list = [
            f"{combined_single_args} {s}".strip() if combined_single_args else s
            for s in expanded_sweeps
        ]
    elif interactive_mode and not combined_single_args:
        import questionary

        ans_args = questionary.text(
            "Arguments (Supports sweeps via '{0..4}', '{00..09}', '{a,b}', or ';' separator):",
            default="",
            style=get_prompt_style(),
        ).ask()
        if ans_args is None:
            sys.exit(0)
        ans_clean = ans_args.strip()
        sweep_list = expand_arg_sweeps(ans_clean)
        raw_sweep_items = list(sweep_list)
        if len(sweep_list) > 1 and ans_clean:
            cli_sweep_specs = [ans_clean]
    else:
        sweep_list = [combined_single_args]
        raw_sweep_items = [combined_single_args]

    for s_arg in sweep_list:
        if s_arg:
            format_condor_arg_line(s_arg)

    no_tag_requested = bool(getattr(args, "no_tag", False))
    raw_tag_inputs: List[str] = list(args.tag) if isinstance(args.tag, list) else ([args.tag] if args.tag else [])
    if len(raw_tag_inputs) == 1 and raw_tag_inputs[0].strip().lower() in ("none", ""):
        no_tag_requested = True

    if not raw_tag_inputs and not no_tag_requested:
        cfg_tag = str(proj_cfg.get("tag", "")).strip()
        if cfg_tag:
            raw_tag_inputs = [cfg_tag]

    if interactive_mode and not raw_tag_inputs and not no_tag_requested:
        import questionary

        ans_tag = questionary.text(
            "Optional run tag for folder name (e.g., 'exp1_lr1e4', or Enter to skip):",
            default="",
            style=get_prompt_style(),
        ).ask()
        if ans_tag is None:
            sys.exit(0)
        if ans_tag.strip():
            raw_tag_inputs = [ans_tag.strip()]

    tag_list: List[str] = []
    if not no_tag_requested:
        for t_raw in raw_tag_inputs:
            t_str = str(t_raw).strip()
            if not t_str or t_str.lower() == "none":
                continue
            if "{" in t_str or ";" in t_str:
                expanded_t = expand_arg_sweeps(t_str)
            elif "," in t_str:
                expanded_t = [p.strip() for p in t_str.split(",")]
            else:
                expanded_t = [t_str]
            for tok in expanded_t:
                cleaned = sanitize_tag(tok)
                if cleaned:
                    tag_list.append(cleaned)

    if no_tag_requested:
        tag_list = []
        base_tag = ""
    else:
        base_tag = (
            tag_list[0]
            if len(tag_list) == 1
            else ("_".join(tag_list) if tag_list and not getattr(args, "chain", False) else "")
        )

    return combined_single_args, sweep_list, raw_sweep_items, cli_sweep_specs, base_tag, tag_list


def _render_submission_summary(
    *,
    rel_target: str,
    exec_type: str,
    cwd: Path,
    run_id: str,
    tag: Optional[str] = None,
    venv_path: Optional[Path],
    arch: str,
    cpus: Optional[int],
    mem_mb: Optional[int],
    gpus: Optional[int],
    sweep_list: List[str],
    git_info: Dict[str, object],
    env_vars: Dict[str, str],
    equivalent_cli: str,
    dependency_info: Optional[str] = None,
    cloned_from_label: Optional[str] = None,
    max_parallel: Optional[int] = None,
    chain: bool = False,
    chain_tags: Optional[List[str]] = None,
) -> None:
    summary_table = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
    summary_table.add_column("Property", style="bold cyan", width=20)
    summary_table.add_column("Value", style="white")

    summary_table.add_row("Target Executable", f"[bold]{rel_target}[/bold] ({exec_type})")
    run_tag_disp = f"[bold cyan]{tag}[/bold cyan] [dim]({run_id})[/dim]" if tag else f"[dim]{run_id}[/dim]"
    summary_table.add_row("Run Tag / ID", run_tag_disp)
    if venv_path:
        try:
            venv_disp = venv_path.relative_to(cwd).as_posix()
        except ValueError:
            venv_disp = venv_path.name
    else:
        venv_disp = "[dim]None (System)[/dim]"
    summary_table.add_row("Python Venv", venv_disp)
    summary_table.add_row("Architecture", f"[bold magenta]{arch}[/bold magenta]")

    if arch == "GPU3090":
        hw_str = "1x RTX 3090 (24GB VRAM) + Full Node CPUs & RAM (Node-Locked)"
    elif arch == "A100":
        ram_str = f"{mem_mb} MB ({mem_mb / 1024:.1f} GB)" if mem_mb else "Default (<= 64 GB)"
        hw_str = f"{gpus}x NVIDIA A100 (40GB VRAM) │ RAM: {ram_str}"
    else:
        assert cpus is not None and mem_mb is not None
        hw_str = f"{cpus} CPU Cores │ {mem_mb} MB RAM ({mem_mb / 1024:.1f} GB)"
    summary_table.add_row("Resources", hw_str)

    if len(sweep_list) > 1:
        if chain:
            preview_lines = []
            for i, s in enumerate(sweep_list[:5]):
                t_badge = f" [cyan][{chain_tags[i]}][/cyan]" if chain_tags and i < len(chain_tags) else ""
                preview_lines.append(f"  [dim]Run {i + 1}:[/dim]{t_badge} {s}")
            preview_sweeps = "\n".join(preview_lines)
            if len(sweep_list) > 5:
                preview_sweeps += f"\n  [dim]... (+{len(sweep_list) - 5} more chained runs)[/dim]"
            summary_table.add_row(
                "Job Mode",
                f"[bold cyan]Sequential Chained Runs ({len(sweep_list)} separate runs: Run 1 -> Run 2 -> ...)[/bold cyan]\n{preview_sweeps}",
            )
        else:
            preview_sweeps = "\n".join(
                [f"  [dim]Process {i}:[/dim] {s}" for i, s in enumerate(sweep_list[:5])]
            )
            if len(sweep_list) > 5:
                preview_sweeps += f"\n  [dim]... (+{len(sweep_list) - 5} more parallel jobs)[/dim]"
            mode_label = (
                f"Parallel Array Sweep ({len(sweep_list)} jobs, max {max_parallel} parallel)"
                if max_parallel
                else f"Parallel Array Sweep ({len(sweep_list)} jobs)"
            )
            summary_table.add_row(
                "Job Mode",
                f"[bold yellow]{mode_label}[/bold yellow]\n{preview_sweeps}",
            )
    else:
        summary_table.add_row(
            "Arguments", sweep_list[0] if sweep_list[0] else "[dim](none)[/dim]"
        )

    if dependency_info:
        summary_table.add_row("Dependency", dependency_info)

    if cloned_from_label:
        summary_table.add_row(
            "Code Snapshot", f"[cyan]Frozen snapshot from[/cyan] {cloned_from_label}"
        )
    elif git_info["is_git"]:
        dirty_badge = (
            "[yellow](+ repro/uncommitted.patch)[/yellow]"
            if git_info["dirty"]
            else "[green](clean)[/green]"
        )
        commit_short = str(git_info["commit"])[:8]
        summary_table.add_row(
            "Git Provenance",
            f"Branch [bold]{git_info['branch']}[/bold] @ {commit_short} {dirty_badge} + [cyan]repro/code/[/cyan]",
        )
    else:
        summary_table.add_row("Code Snapshot", "Isolated copy in [cyan]repro/code/[/cyan]")

    if env_vars:
        env_preview = ", ".join(f"[cyan]{k}[/cyan]={v}" for k, v in env_vars.items())
        summary_table.add_row("Env Exports", env_preview)

    summary_table.add_row("Fast CLI Command", f"[bold green]{equivalent_cli}[/bold green]")

    console.print()
    console.print(
        Panel(
            summary_table,
            title="[bold cyan]🚀 Hermes 2 Job Submission Summary[/bold cyan]",
            border_style="cyan",
            expand=False,
        )
    )


def _submit_dag_run(
    cwd: Path, dag_path: Path, run_dir: Path, repro_dir: Path
) -> Tuple[Optional[int], str]:
    """Submits a DAGMan workflow via condor_submit_dag -no_submit -> sanitize -> condor_submit."""
    run_json_path = run_dir / "run.json"
    try:
        try:
            dag_submit_arg = str(dag_path.relative_to(cwd))
        except ValueError:
            dag_submit_arg = str(dag_path)

        subprocess.run(
            ["condor_submit_dag", "-f", "-no_submit", dag_submit_arg],
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )

        condor_sub_path = Path(f"{dag_path}.condor.sub")
        if condor_sub_path.is_file():
            sub_txt = condor_sub_path.read_text(encoding="utf-8", errors="replace")
            write_text_lf(condor_sub_path, sanitize_dagman_condor_sub(sub_txt))

        try:
            sub_submit_arg = str(condor_sub_path.relative_to(cwd))
        except ValueError:
            sub_submit_arg = str(condor_sub_path)

        sub_res = subprocess.run(
            ["condor_submit", sub_submit_arg],
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        condor_out = sub_res.stdout.strip()
        cluster_match = re.search(r"submitted to cluster\s+(\d+)", condor_out)
        dagman_cluster_id = int(cluster_match.group(1)) if cluster_match else None

        def _record_dag_info(data: Dict[str, Any]) -> None:
            sub_block = data.setdefault("submission", {})
            sub_block["condor_submit_output"] = condor_out
            sub_block["cluster_id"] = dagman_cluster_id
            dep_block = data.setdefault("dependency", {})
            dep_block["dagman_cluster_id"] = dagman_cluster_id

        locked_mutate_run_json(run_json_path, repro_dir, _record_dag_info)
        return dagman_cluster_id, condor_out
    except FileNotFoundError:
        locked_mutate_run_json(
            run_json_path,
            repro_dir,
            lambda d: d.update({"status": "SUBMIT_FAILED"}),
        )
        console.print(
            f"[bold red]✘ Error:[/bold red] 'condor_submit_dag' not found in PATH (are you on hermes2?).\n"
            f"  Run directory and repro capsule were still generated at: [cyan]runs/{run_dir.name}/[/cyan]"
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        err_msg = (e.stderr or e.stdout or "").strip()

        def _record_dag_err(data: Dict[str, Any]) -> None:
            data["status"] = "SUBMIT_FAILED"
            data.setdefault("submission", {})["condor_submit_error"] = err_msg

        locked_mutate_run_json(run_json_path, repro_dir, _record_dag_err)
        console.print(f"[bold red]✘ condor_submit_dag failed:[/bold red]\n{err_msg}")
        sys.exit(1)


def _submit_direct_run(
    cwd: Path, sub_path: Path, run_dir: Path, repro_dir: Path
) -> Tuple[Optional[int], str]:
    """Submits a standard job.sub via condor_submit and records cluster_id in run.json."""
    run_json_path = run_dir / "run.json"
    try:
        res = subprocess.run(
            ["condor_submit", str(sub_path)],
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        condor_out = res.stdout.strip()
        cluster_match = re.search(r"submitted to cluster\s+(\d+)", condor_out)
        cluster_id = int(cluster_match.group(1)) if cluster_match else None

        def _record_submit_info(data: Dict[str, Any]) -> None:
            sub_block = data.setdefault("submission", {})
            sub_block["condor_submit_output"] = condor_out
            sub_block["cluster_id"] = cluster_id

        locked_mutate_run_json(run_json_path, repro_dir, _record_submit_info)
        return cluster_id, condor_out
    except FileNotFoundError:
        locked_mutate_run_json(
            run_json_path,
            repro_dir,
            lambda d: d.update({"status": "SUBMIT_FAILED"}),
        )
        console.print(
            f"[bold red]✘ Error:[/bold red] 'condor_submit' not found in PATH (are you on hermes2?).\n"
            f"  Run directory and repro capsule were still generated at: [cyan]runs/{run_dir.name}/[/cyan]"
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        err_msg = (e.stderr or e.stdout or "").strip()

        def _record_submit_err(data: Dict[str, Any]) -> None:
            data["status"] = "SUBMIT_FAILED"
            data.setdefault("submission", {})["condor_submit_error"] = err_msg

        locked_mutate_run_json(run_json_path, repro_dir, _record_submit_err)
        console.print(f"[bold red]✘ condor_submit failed:[/bold red]\n{err_msg}")
        sys.exit(1)


def main() -> None:
    argv = sys.argv[1:]
    passthrough_args = ""
    if "--" in argv:
        sep_idx = argv.index("--")
        passthrough_list = argv[sep_idx + 1 :]
        argv = argv[:sep_idx]
        passthrough_args = " ".join(shlex.quote(x) for x in passthrough_list)

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.after_ok and not args.after and not args.chain:
        console.print("[bold red]✘ Error:[/bold red] --after-ok requires --after <run> or --chain.")
        sys.exit(1)

    cwd = Path.cwd()
    runs_root = get_runs_root(cwd).resolve()
    proj_cfg = load_project_config(cwd)

    raw_mp = args.max_parallel if args.max_parallel is not None else proj_cfg.get("max_parallel")
    max_parallel = int(raw_mp) if raw_mp is not None else None
    if max_parallel is not None and max_parallel < 1:
        console.print("[bold red]✘ Error:[/bold red] --max-parallel (-j) must be >= 1.")
        sys.exit(1)
    if args.chain and max_parallel is not None:
        console.print(
            "[bold red]✘ Error:[/bold red] Cannot combine --chain (separate runs) with -j / --max-parallel (single-run sweep throttle)."
        )
        sys.exit(1)

    source_run_dir: Optional[Path] = None
    source_data: Optional[Dict[str, Any]] = None
    cloned_from_label: Optional[str] = None
    if getattr(args, "from_run", None):
        from hermes_tools.runs_cli import _format_run_short, resolve_run_dir

        source_run_dir = resolve_run_dir(runs_root, args.from_run)
        source_data = load_run_json(source_run_dir)
        cloned_from_label = _format_run_short(source_run_dir, source_data)
        src_arch = (source_data.get("resources_requested") or {}).get("architecture")
        if src_arch and args.arch is None:
            proj_cfg = {**proj_cfg, "arch": src_arch}

    parent_run_dir: Optional[Path] = None
    parent_data: Optional[Dict[str, Any]] = None
    parent_cid: Optional[int] = None
    submit_via_dag = False

    if args.after:
        from hermes_tools.runs_cli import resolve_run_dir

        parent_run_dir = resolve_run_dir(runs_root, args.after, exit_on_fail=False)
        if parent_run_dir and parent_run_dir.is_dir():
            parent_data = load_run_json(parent_run_dir)
            if not parent_data:
                console.print(
                    f"[bold red]✘ Error:[/bold red] Could not load run.json from parent run '{parent_run_dir.name}'."
                )
                sys.exit(1)

            live_clusters = query_live_condor_clusters()
            if run_needs_live_reconcile(parent_data):
                parent_data = reconcile_run(parent_run_dir, parent_data, live_clusters)
            p_cids = get_all_cluster_ids(parent_data)
            parent_cid = p_cids[-1] if p_cids else None
            parent_status = str(parent_data.get("_live_badge") or parent_data.get("status", "UNKNOWN"))
            is_parent_active = parent_status.startswith(("QUEUED", "RUNNING", "HELD", "WAITING")) or (
                live_clusters is not None and parent_cid is not None and parent_cid in live_clusters
            )

            if not is_parent_active:
                if parent_status == "COMPLETED":
                    console.print(
                        f"[dim]Parent run [bold]{parent_run_dir.name}[/bold] has already completed successfully. "
                        f"Submitting directly without DAGMan delay...[/dim]"
                    )
                    submit_via_dag = False
                elif args.after_ok:
                    console.print(
                        f"[bold red]✘ Cannot submit:[/bold red] Parent run '{parent_run_dir.name}' has already finished "
                        f"with status '{parent_status}'. Aborting because --after-ok was specified."
                    )
                    sys.exit(1)
                else:
                    console.print(
                        f"[dim]Parent run [bold]{parent_run_dir.name}[/bold] has already finished ({parent_status}). "
                        f"Proceeding with direct submission...[/dim]"
                    )
                    submit_via_dag = False
            else:
                submit_via_dag = True
        elif args.after.strip().isdigit():
            raw_cid = int(args.after.strip())
            live_clusters = query_live_condor_clusters()
            if args.dry_run or (live_clusters is not None and raw_cid in live_clusters):
                parent_cid = raw_cid
                parent_run_dir = None
                parent_data = None
                submit_via_dag = True
                if not args.dry_run:
                    console.print(
                        f"[dim]Parent job [bold]#{parent_cid}[/bold] detected as active raw HTCondor cluster in condor_q.[/dim]"
                    )
            else:
                console.print(
                    f"[bold red]✘ Error:[/bold red] Cluster ID #{raw_cid} is not active in HTCondor (condor_q) and does not match any local run in {runs_root}."
                )
                sys.exit(1)
        else:
            console.print(f"[bold red]✘ Error:[/bold red] Parent run '{args.after}' could not be resolved.")
            sys.exit(1)

    project_code = args.project if args.project is not None else proj_cfg.get("project")
    nodes_include = list(args.node) or _as_str_list(proj_cfg.get("node"))
    nodes_exclude = list(args.exclude) or _as_str_list(proj_cfg.get("exclude"))
    scratch_files = list(args.scratch) or _as_str_list(proj_cfg.get("scratch"))

    env_vars: Dict[str, str] = {}
    if source_data is not None:
        src_env = (source_data.get("resources_requested") or {}).get("env")
        if isinstance(src_env, dict):
            for k, v in src_env.items():
                if k and v is not None:
                    env_vars[str(k).strip()] = str(v)
    cfg_env = proj_cfg.get("env")
    if isinstance(cfg_env, dict):
        for k, v in cfg_env.items():
            if k and v is not None:
                env_vars[str(k).strip()] = str(v)
    for item in args.env:
        if "=" not in item:
            console.print(
                f"[bold red]✘ Error:[/bold red] Invalid --env '{item}' (expected KEY=VALUE)."
            )
            sys.exit(1)
        k, v = item.split("=", 1)
        env_vars[k.strip()] = v

    (
        target_path,
        exec_type,
        venv_path,
        explicit_venv_used,
        no_venv_flag,
        interactive_mode,
    ) = _resolve_target_and_venv(args, proj_cfg, cwd, source_run_dir=source_run_dir, source_data=source_data)

    arch, cpus, mem_mb, gpus, interactive_mode = _resolve_hardware_resources(
        args, proj_cfg, interactive_mode
    )

    combined_single_args, sweep_list, raw_sweep_items, cli_sweep_specs, tag, tag_list = _resolve_sweeps_and_tag(
        args, proj_cfg, passthrough_args, interactive_mode
    )
    no_tag_flag = bool(
        getattr(args, "no_tag", False)
        or (len(args.tag) == 1 and str(args.tag[0]).strip().lower() in ("none", ""))
    )

    if source_run_dir is not None and source_data is not None:
        rel_target = str(args.target or (source_data.get("submission") or {}).get("target") or target_path.name)
    else:
        rel_target = (
            target_path.relative_to(cwd).as_posix()
            if target_path.is_relative_to(cwd)
            else target_path.as_posix()
        )
    resolved_after_id: Optional[str] = None
    if parent_run_dir is not None:
        resolved_after_id = parent_run_dir.name
    elif parent_cid is not None:
        resolved_after_id = str(parent_cid)
    elif args.after:
        resolved_after_id = str(args.after)

    is_chained_sweep = bool(args.chain and len(sweep_list) > 1)
    base_tokens = build_send_command_tokens(
        send_bin="send",
        target=rel_target,
        arch=arch,
        cpus=cpus,
        mem_mb=mem_mb,
        gpus=gpus,
        tag=(tag_list if (is_chained_sweep and len(tag_list) > 1) else (tag or None)),
        no_tag=no_tag_flag,
        no_venv=no_venv_flag,
        venv=venv_path.as_posix() if (explicit_venv_used and venv_path) else None,
        nodes_include=nodes_include,
        nodes_exclude=nodes_exclude,
        project_code=str(project_code) if project_code else None,
        scratch_files=scratch_files,
        env_items=list(args.env),
        sweep_file=str(args.sweep_file) if args.sweep_file else None,
        sweep_specs=cli_sweep_specs,
        after=resolved_after_id,
        after_ok=bool(args.after_ok),
        from_run=source_run_dir.name if source_run_dir else None,
        max_parallel=max_parallel,
        chain=is_chained_sweep,
    )
    cli_parts = [base_tokens[0]] + [shlex.quote(t) for t in base_tokens[1:]]
    if len(sweep_list) > 1:
        if combined_single_args:
            cli_parts.extend(["--", combined_single_args])
    elif sweep_list and sweep_list[0] and not cli_sweep_specs and not args.sweep_file:
        cli_parts.extend(["--", sweep_list[0]])

    equivalent_cli = " ".join(cli_parts)

    now = datetime.datetime.now()
    submit_epoch = int(now.timestamp())
    ts_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    safe_tag = f"_{tag}" if tag else ""
    run_id = f"{ts_str}_{target_path.stem}_{arch}{safe_tag}"
    run_dir = runs_root / run_id
    repro_dir = run_dir / "repro"

    if source_run_dir is not None and source_data is not None:
        git_info = dict(source_data.get("git") or {})
        git_info["cloned_from_run"] = source_run_dir.name
    else:
        git_info = get_git_info(cwd)

    dep_info: Optional[str] = None
    if parent_run_dir is not None:
        cid_part = f" (#{parent_cid})" if parent_cid else ""
        ok_part = " [bold yellow][exit 0 required][/bold yellow]" if args.after_ok else ""
        dep_info = f"Starts after [bold]{parent_run_dir.name}[/bold]{cid_part}{ok_part}"
        if not submit_via_dag:
            dep_info += " [dim](parent completed; direct submit)[/dim]"
    elif parent_cid is not None:
        ok_part = " [bold yellow][exit 0 required][/bold yellow]" if args.after_ok else ""
        dep_info = f"Starts after external cluster [bold]#{parent_cid}[/bold]{ok_part}"

    step_tags: List[str] = []
    step_run_ids: List[str] = []
    if is_chained_sweep:
        if not no_tag_flag and len(tag_list) > 1 and len(tag_list) != len(sweep_list):
            console.print(
                f"[bold red]✘ Error:[/bold red] Provided {len(tag_list)} tags {tag_list}, "
                f"but sweep has {len(sweep_list)} items."
            )
            sys.exit(1)

        used_ids = set()
        for idx in range(len(sweep_list)):
            raw_item = raw_sweep_items[idx] if idx < len(raw_sweep_items) else sweep_list[idx]
            if no_tag_flag:
                st_tag = ""
            elif len(tag_list) > 1:
                st_tag = tag_list[idx]
            else:
                st_tag = _derive_sweep_item_tag(tag, raw_item, idx)

            if st_tag:
                cand_id = f"{ts_str}_{target_path.stem}_{arch}_{st_tag}"
                if cand_id in used_ids or (runs_root / cand_id).exists():
                    cand_id = f"{cand_id}_{idx}"
            else:
                offset_sec = idx
                while True:
                    step_ts = (now + datetime.timedelta(seconds=offset_sec)).strftime(
                        "%Y-%m-%d_%H-%M-%S"
                    )
                    cand_id = f"{step_ts}_{target_path.stem}_{arch}"
                    if cand_id not in used_ids and not (runs_root / cand_id).exists():
                        break
                    offset_sec += 1
            used_ids.add(cand_id)
            step_tags.append(st_tag)
            step_run_ids.append(cand_id)

    _render_submission_summary(
        rel_target=rel_target,
        exec_type=exec_type,
        cwd=cwd,
        run_id=f"{step_run_ids[0]} .. (+{len(step_run_ids) - 1})" if is_chained_sweep else run_id,
        tag=tag or None,
        venv_path=venv_path,
        arch=arch,
        cpus=cpus,
        mem_mb=mem_mb,
        gpus=gpus,
        sweep_list=sweep_list,
        git_info=git_info,
        env_vars=env_vars,
        equivalent_cli=equivalent_cli,
        dependency_info=dep_info,
        cloned_from_label=cloned_from_label,
        max_parallel=max_parallel,
        chain=is_chained_sweep,
        chain_tags=step_tags if is_chained_sweep else None,
    )

    if not args.yes and not args.dry_run:
        import questionary

        prompt_msg = (
            f"Submit {len(sweep_list)} chained runs to HTCondor queue now?"
            if is_chained_sweep
            else "Submit job to HTCondor queue now?"
        )
        confirm = questionary.confirm(
            prompt_msg, default=True, style=get_prompt_style()
        ).ask()
        if not confirm:
            console.print("[yellow]Submission cancelled.[/yellow]")
            sys.exit(0)

    if is_chained_sweep:
        first_run_dir: Optional[Path] = None
        prev_run_dir: Optional[Path] = None
        prev_cid: Optional[int] = None
        created_runs: List[Tuple[Path, Optional[int], str, str]] = []
        patch_rel_path: Optional[str] = None

        for idx, single_arg in enumerate(sweep_list):
            step_tag = step_tags[idx]
            step_run_id = step_run_ids[idx]
            step_run_dir = runs_root / step_run_id
            step_repro_dir = step_run_dir / "repro"
            step_repro_dir.mkdir(parents=True, exist_ok=True)

            if idx == 0:
                freeze_python_env(venv_path, step_repro_dir, detach=True)
                if source_run_dir is not None:
                    snap_exec_path, snap_py_root = clone_code_snapshot(
                        source_run_dir, rel_target, step_repro_dir
                    )
                    if (step_repro_dir / "uncommitted.patch").is_file():
                        patch_rel_path = "repro/uncommitted.patch"
                else:
                    snap_exec_path, snap_py_root = create_code_snapshot(
                        target_path, cwd, step_repro_dir, git_info
                    )
                    if git_info.get("is_git") and git_info.get("diff"):
                        write_text_lf(step_repro_dir / "uncommitted.patch", str(git_info["diff"]))
                        patch_rel_path = "repro/uncommitted.patch"
                first_run_dir = step_run_dir
            else:
                assert first_run_dir is not None
                first_env = first_run_dir / "repro" / "env.txt"
                if first_env.is_file():
                    shutil.copy2(first_env, step_repro_dir / "env.txt")
                snap_exec_path, snap_py_root = clone_code_snapshot(
                    first_run_dir, rel_target, step_repro_dir
                )
                patch_rel_path = (
                    "repro/uncommitted.patch"
                    if (step_repro_dir / "uncommitted.patch").is_file()
                    else None
                )

            if idx == 0:
                step_parent_dir = parent_run_dir
                step_parent_cid = parent_cid
                step_submit_via_dag = submit_via_dag
                step_resolved_after = resolved_after_id
            else:
                step_parent_dir = prev_run_dir
                step_parent_cid = prev_cid
                step_submit_via_dag = True
                step_resolved_after = (
                    prev_run_dir.name
                    if prev_run_dir is not None
                    else (str(prev_cid) if prev_cid is not None else None)
                )

            step_tokens = build_send_command_tokens(
                send_bin="send",
                target=rel_target,
                arch=arch,
                cpus=cpus,
                mem_mb=mem_mb,
                gpus=gpus,
                tag=step_tag or None,
                no_tag=no_tag_flag if not step_tag else False,
                no_venv=no_venv_flag,
                venv=venv_path.as_posix() if (explicit_venv_used and venv_path) else None,
                nodes_include=nodes_include,
                nodes_exclude=nodes_exclude,
                project_code=str(project_code) if project_code else None,
                scratch_files=scratch_files,
                env_items=list(args.env),
                args_flag_val=single_arg or None,
                after=step_resolved_after,
                after_ok=bool(args.after_ok) if step_resolved_after else False,
                from_run=source_run_dir.name if source_run_dir else None,
            )
            step_equivalent_cli = " ".join(
                [step_tokens[0]] + [shlex.quote(t) for t in step_tokens[1:]]
            )

            sh_path = step_repro_dir / "job.sh"
            sh_content = generate_bash_wrapper(
                exec_type=exec_type,
                target_exec_path=snap_exec_path,
                venv_path=venv_path,
                pythonpath_dir=snap_py_root,
                project_cwd=cwd,
                run_dir=step_run_dir,
                repro_dir=step_repro_dir,
                submit_epoch=submit_epoch,
                sweep_count=1,
                requested_cpus=cpus,
                env_vars=env_vars,
            )
            write_text_lf(sh_path, sh_content)
            try:
                sh_path.chmod(0o755)
            except OSError:
                pass

            sub_path = step_repro_dir / "job.sub"
            sub_content = generate_sub_file(
                sh_path=sh_path,
                project_cwd=cwd,
                run_dir=step_run_dir,
                repro_dir=step_repro_dir,
                script_stem=target_path.stem,
                arch=arch,
                cpus=cpus,
                mem_mb=mem_mb,
                gpus=gpus,
                sweep_args=[single_arg],
                nodes_include=nodes_include,
                nodes_exclude=nodes_exclude,
                project_code=str(project_code) if project_code else None,
                scratch_files=scratch_files,
            )
            write_text_lf(sub_path, sub_content)

            step_has_dep = step_submit_via_dag or (args.dry_run and (idx > 0 or bool(args.after)))
            dag_path: Optional[Path] = None
            if step_has_dep:
                wait_script_path = step_repro_dir / "wait_parent.py"
                write_text_lf(wait_script_path, generate_wait_parent_script())
                try:
                    wait_script_path.chmod(0o755)
                except OSError:
                    pass
                dag_path = step_repro_dir / "job.dag"
                submit_py = resolve_submit_python()
                dagman_cfg_path: Optional[Path] = None
                dagman_cfg = generate_dagman_config(shutil.which("condor_submit"))
                if dagman_cfg:
                    dagman_cfg_path = step_repro_dir / "dagman.config"
                    write_text_lf(dagman_cfg_path, dagman_cfg)
                dag_content = generate_dag_file(
                    sub_path=sub_path,
                    wait_script_path=wait_script_path,
                    python_bin=submit_py,
                    child_dir=step_run_dir,
                    parent_dir=step_parent_dir,
                    parent_cluster_id=step_parent_cid,
                    require_ok=bool(args.after_ok),
                    config_path=dagman_cfg_path,
                )
                write_text_lf(dag_path, dag_content)

            step_status = (
                "DRY_RUN"
                if args.dry_run
                else ("WAITING" if step_submit_via_dag else "QUEUED")
            )
            step_run_json: Dict[str, Any] = {
                "run_id": step_run_id,
                "status": step_status,
                "submission": {
                    "submitted_at": now.isoformat(timespec="seconds"),
                    "user": get_username(),
                    "submit_host": get_hostname(),
                    "target": rel_target,
                    "tag": step_tag or None,
                    "snapshot_executable": snap_exec_path.relative_to(step_run_dir).as_posix(),
                    "working_directory": str(cwd),
                    "run_directory": str(step_run_dir),
                    "venv": str(venv_path) if venv_path else None,
                    "no_venv": no_venv_flag,
                    "equivalent_cli": step_equivalent_cli,
                    "cluster_id": None,
                    "chain_index": idx,
                    "chain_total": len(sweep_list),
                    **({"cloned_from_run": source_run_dir.name} if source_run_dir else {}),
                },
                "resources_requested": {
                    "architecture": arch,
                    "cpus": cpus,
                    "memory_mb": mem_mb,
                    "gpus": 1 if arch == "GPU3090" else gpus,
                    "sweep_count": 1,
                    "sweep_arguments": [single_arg],
                    "node_include": nodes_include,
                    "node_exclude": nodes_exclude,
                    "project_code": str(project_code) if project_code else None,
                    "scratch_files": scratch_files,
                    "env": env_vars,
                },
                "git": {
                    "is_git": git_info.get("is_git", False),
                    "branch": git_info.get("branch"),
                    "commit": git_info.get("commit"),
                    "dirty": git_info.get("dirty", False),
                    "patch_file": patch_rel_path,
                    **({"cloned_from_run": source_run_dir.name} if source_run_dir else {}),
                },
                "executions": [],
            }
            if step_has_dep:
                step_run_json["dependency"] = {
                    "parent_run_id": step_resolved_after
                    or (
                        step_parent_dir.name
                        if step_parent_dir
                        else (str(step_parent_cid) if step_parent_cid else str(args.after))
                    ),
                    "parent_run_dir": str(step_parent_dir) if step_parent_dir else None,
                    "parent_cluster_id": step_parent_cid,
                    "require_ok": bool(args.after_ok),
                    "dagman_cluster_id": None,
                    "job_cluster_id": None,
                }
            atomic_write_json(step_run_dir / "run.json", step_run_json, step_repro_dir)

            submitted_cid: Optional[int] = None
            if not args.dry_run:
                if step_submit_via_dag and dag_path is not None:
                    submitted_cid, _ = _submit_dag_run(cwd, dag_path, step_run_dir, step_repro_dir)
                else:
                    submitted_cid, _ = _submit_direct_run(cwd, sub_path, step_run_dir, step_repro_dir)

            prev_run_dir = step_run_dir
            prev_cid = submitted_cid
            created_runs.append((step_run_dir, submitted_cid, step_status, step_tag))

        update_latest_symlink(runs_root, created_runs[-1][0])
        chain_lines = []
        for idx, (r_dir, cid, st, s_tag) in enumerate(created_runs, start=1):
            cid_disp = f"#{cid}" if cid else "dry"
            st_color = "cyan" if st == "QUEUED" else ("blue" if st == "WAITING" else "yellow")
            tag_part = f" ({s_tag})" if s_tag else ""
            chain_lines.append(
                f"  {idx}. [bold]{r_dir.name}[/bold]{tag_part} — [{st_color}]{st}[/{st_color}] [{cid_disp}]"
            )
        console.print(
            Panel(
                "\n".join(chain_lines),
                title=f"[bold green]🔗 Chained {len(created_runs)} Sequential Runs[/bold green]",
                border_style="green",
                expand=False,
            )
        )
        if args.follow and not args.dry_run:
            from hermes_tools.runs_cli import follow_run_log

            follow_run_log(created_runs[0][0], created_runs[0][0] / f"{target_path.stem}.out", initial_lines=60)
        return

    # 7. Materialize Run Directory & Repro Capsule (Cached/detached env freeze)
    repro_dir.mkdir(parents=True, exist_ok=True)
    freeze_python_env(venv_path, repro_dir, detach=True)

    patch_rel_path: Optional[str] = None
    if source_run_dir is not None:
        snap_exec_path, snap_py_root = clone_code_snapshot(source_run_dir, rel_target, repro_dir)
        if (repro_dir / "uncommitted.patch").is_file():
            patch_rel_path = "repro/uncommitted.patch"
    else:
        snap_exec_path, snap_py_root = create_code_snapshot(target_path, cwd, repro_dir, git_info)
        if git_info.get("is_git") and git_info.get("diff"):
            write_text_lf(repro_dir / "uncommitted.patch", str(git_info["diff"]))
            patch_rel_path = "repro/uncommitted.patch"

    sh_path = repro_dir / "job.sh"
    sh_content = generate_bash_wrapper(
        exec_type=exec_type,
        target_exec_path=snap_exec_path,
        venv_path=venv_path,
        pythonpath_dir=snap_py_root,
        project_cwd=cwd,
        run_dir=run_dir,
        repro_dir=repro_dir,
        submit_epoch=submit_epoch,
        sweep_count=len(sweep_list),
        requested_cpus=cpus,
        env_vars=env_vars,
    )
    write_text_lf(sh_path, sh_content)
    try:
        sh_path.chmod(0o755)
    except OSError:
        pass

    sub_path = repro_dir / "job.sub"
    sub_content = generate_sub_file(
        sh_path=sh_path,
        project_cwd=cwd,
        run_dir=run_dir,
        repro_dir=repro_dir,
        script_stem=target_path.stem,
        arch=arch,
        cpus=cpus,
        mem_mb=mem_mb,
        gpus=gpus,
        sweep_args=sweep_list,
        nodes_include=nodes_include,
        nodes_exclude=nodes_exclude,
        project_code=str(project_code) if project_code else None,
        scratch_files=scratch_files,
        max_parallel=max_parallel,
    )
    write_text_lf(sub_path, sub_content)

    is_sweep = len(sweep_list) > 1

    dag_path: Optional[Path] = None
    dag_content: Optional[str] = None
    if submit_via_dag or (args.dry_run and args.after):
        wait_script_path = repro_dir / "wait_parent.py"
        write_text_lf(wait_script_path, generate_wait_parent_script())
        try:
            wait_script_path.chmod(0o755)
        except OSError:
            pass

        dag_path = repro_dir / "job.dag"
        submit_py = resolve_submit_python()
        dagman_cfg_path: Optional[Path] = None
        dagman_cfg = generate_dagman_config(shutil.which("condor_submit"))
        if dagman_cfg:
            dagman_cfg_path = repro_dir / "dagman.config"
            write_text_lf(dagman_cfg_path, dagman_cfg)
        dag_content = generate_dag_file(
            sub_path=sub_path,
            wait_script_path=wait_script_path,
            python_bin=submit_py,
            child_dir=run_dir,
            parent_dir=parent_run_dir,
            parent_cluster_id=parent_cid,
            require_ok=bool(args.after_ok),
            config_path=dagman_cfg_path,
        )
        write_text_lf(dag_path, dag_content)

    initial_status = "DRY_RUN" if args.dry_run else ("WAITING" if submit_via_dag else "QUEUED")
    run_json_data: Dict[str, Any] = {
        "run_id": run_id,
        "status": initial_status,
        "submission": {
            "submitted_at": now.isoformat(timespec="seconds"),
            "user": get_username(),
            "submit_host": get_hostname(),
            "target": rel_target,
            "tag": tag or None,
            "snapshot_executable": snap_exec_path.relative_to(run_dir).as_posix(),
            "working_directory": str(cwd),
            "run_directory": str(run_dir),
            "venv": str(venv_path) if venv_path else None,
            "no_venv": no_venv_flag,
            "equivalent_cli": equivalent_cli,
            "cluster_id": None,
            **({"cloned_from_run": source_run_dir.name} if source_run_dir else {}),
        },
        "resources_requested": {
            "architecture": arch,
            "cpus": cpus,
            "memory_mb": mem_mb,
            "gpus": 1 if arch == "GPU3090" else gpus,
            "sweep_count": len(sweep_list),
            "max_parallel": max_parallel,
            "sweep_arguments": sweep_list,
            "node_include": nodes_include,
            "node_exclude": nodes_exclude,
            "project_code": str(project_code) if project_code else None,
            "scratch_files": scratch_files,
            "env": env_vars,
        },
        "git": {
            "is_git": git_info.get("is_git", False),
            "branch": git_info.get("branch"),
            "commit": git_info.get("commit"),
            "dirty": git_info.get("dirty", False),
            "patch_file": patch_rel_path,
            **({"cloned_from_run": source_run_dir.name} if source_run_dir else {}),
        },
        "executions": [],
    }
    if submit_via_dag or (args.dry_run and args.after):
        run_json_data["dependency"] = {
            "parent_run_id": resolved_after_id or (parent_run_dir.name if parent_run_dir else (str(parent_cid) if parent_cid else str(args.after))),
            "parent_run_dir": str(parent_run_dir) if parent_run_dir else None,
            "parent_cluster_id": parent_cid,
            "require_ok": bool(args.after_ok),
            "dagman_cluster_id": None,
            "job_cluster_id": None,
        }

    run_json_path = run_dir / "run.json"
    atomic_write_json(run_json_path, run_json_data, repro_dir)

    if args.dry_run:
        from rich.syntax import Syntax

        update_latest_symlink(runs_root, run_dir)
        console.print(
            f"\n[bold yellow]✔ Dry run complete![/bold yellow] Generated capsule in: [cyan]runs/{run_id}/[/cyan]"
        )
        if dag_content:
            console.print("[bold cyan]HTCondor DAGMan Workflow Definition (job.dag):[/bold cyan]")
            console.print(Syntax(dag_content, "ini", theme="monokai", line_numbers=True, word_wrap=True))
            console.print("\n[bold cyan]HTCondor Job Description (job.sub):[/bold cyan]")
        console.print(Syntax(sub_content, "ini", theme="monokai", line_numbers=True, word_wrap=True))
        sys.exit(0)

    # 8. Submit to HTCondor
    if submit_via_dag and dag_path is not None:
        dagman_cluster_id, condor_out = _submit_dag_run(cwd, dag_path, run_dir, repro_dir)
        update_latest_symlink(runs_root, run_dir)

        parent_label = (
            f"#{parent_cid}"
            if parent_cid
            else (parent_run_dir.name if parent_run_dir else "parent")
        )
        condor_q_cmd = f"condor_q {dagman_cluster_id}" if dagman_cluster_id else "condor_q"

        console.print(
            Panel(
                f"[bold green]✔ {condor_out}[/bold green]\n\n"
                f"  • [bold]Status        :[/bold] [bold blue]WAITING[/bold blue] (starts automatically when {parent_label} finishes)\n"
                f"  • [bold]Manage Run    :[/bold] [cyan]runs inspect latest[/cyan]\n"
                f"  • [bold]Live Table    :[/bold] [cyan]runs watch[/cyan]  [dim](or {condor_q_cmd})[/dim]\n"
                f"  • [bold]Diagnose Wait :[/bold] [cyan]runs why latest[/cyan]\n"
                f"  • [bold]Abort Waiting :[/bold] [cyan]runs stop latest[/cyan]",
                title="[bold green]🔗 Dependent Job Chained via DAGMan[/bold green]",
                border_style="green",
                expand=False,
            )
        )
    else:
        cluster_id, condor_out = _submit_direct_run(cwd, sub_path, run_dir, repro_dir)
        update_latest_symlink(runs_root, run_dir)

        condor_q_cmd = f"condor_q {cluster_id}" if cluster_id else "condor_q"

        console.print(
            Panel(
                f"[bold green]✔ {condor_out}[/bold green]\n\n"
                f"  • [bold]Manage Run    :[/bold] [cyan]runs inspect latest[/cyan]\n"
                f"  • [bold]Live Table    :[/bold] [cyan]runs watch[/cyan]  [dim](or {condor_q_cmd})[/dim]\n"
                f"  • [bold]Follow Stdout :[/bold] [cyan]runs logs -f[/cyan]\n"
                f"  • [bold]Follow Stderr :[/bold] [cyan]runs logs -e -f[/cyan]",
                title="[bold green]🎉 Job Submitted to Hermes 2[/bold green]",
                border_style="green",
                expand=False,
            )
        )

    if args.follow:
        from hermes_tools.runs_cli import follow_run_log

        first_out_file = (
            run_dir / f"{target_path.stem}_0.out"
            if is_sweep
            else run_dir / f"{target_path.stem}.out"
        )
        follow_run_log(run_dir, first_out_file, initial_lines=60)


if __name__ == "__main__":
    main()
