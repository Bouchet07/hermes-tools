"""
send_cli.py — HTCondor Job Submission & Provenance CLI entrypoint (`send`).
"""

from __future__ import annotations

import argparse
import datetime
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import questionary
from rich import box
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from hermes_tools.condor import (
    ARCH_SPECS,
    format_condor_arg_line,
    generate_bash_wrapper,
    generate_sub_file,
)
from hermes_tools.core import (
    PROMPT_STYLE,
    atomic_write_json,
    console,
    discover_project_venv,
    expand_arg_sweeps,
    is_valid_memory_input,
    locked_mutate_run_json,
    parse_memory_to_mb,
    sanitize_tag,
    update_latest_symlink,
)
from hermes_tools.snapshot import create_code_snapshot, freeze_python_env, get_git_info


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
    parser.add_argument("target", nargs="?", help="Path to Python script (.py), Bash script (.sh), or binary")
    parser.add_argument("-a", "--arch", choices=list(ARCH_SPECS.keys()), help="Target Hermes 2 +Architecture pool")
    parser.add_argument("-c", "--cpus", type=int, help="Number of CPU cores (ignored on GPU3090/A100)")
    parser.add_argument("-m", "--mem", type=str, help="RAM request, e.g., '8G', '32GB', '4096M' (ignored on GPU3090)")
    parser.add_argument("-g", "--gpus", type=int, choices=[1, 2, 3, 4], help="Number of GPUs (for A100: 1-4)")
    parser.add_argument("--args", type=str, default="", help="Arguments to pass to your script")
    parser.add_argument(
        "-s",
        "--sweep",
        action="append",
        default=[],
        help="Queue parallel parameter sweep (supports '{0..4}', '{a,b}', ';' or repeatable -s)",
    )
    parser.add_argument("--sweep-file", type=str, help="File containing one argument string per line for sweep")
    parser.add_argument("-t", "--tag", type=str, default="", help="Optional tag appended to the run directory name")
    parser.add_argument("--venv", type=str, help="Explicit path to virtual environment (auto-detected by default)")
    parser.add_argument("--no-venv", action="store_true", help="Run without activating a Python virtual environment")
    parser.add_argument("--node", action="append", default=[], help="Target specific node(s), e.g., nv3090-01")
    parser.add_argument("--exclude", action="append", default=[], help="Exclude specific node(s), e.g., her07-100")
    parser.add_argument("--project", type=str, help="I3A +Proyecto accounting code (e.g., '100-100')")
    parser.add_argument(
        "--scratch",
        nargs="+",
        default=[],
        help="Enable RAMDisk scratch mode and transfer specified input files",
    )
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt and submit immediately")
    parser.add_argument("--dry-run", action="store_true", help="Generate run folder and scripts without submitting")
    return parser


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

    if args.cpus is not None and args.cpus <= 0:
        console.print("[bold red]✘ Error:[/bold red] --cpus must be a positive integer.")
        sys.exit(1)

    cwd = Path.cwd()
    interactive_mode = False

    # 1. Resolve Target File
    target_str = args.target
    if not target_str:
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
                style=PROMPT_STYLE,
            ).ask()
            if selected is None:
                sys.exit(0)
            target_str = (
                questionary.path("Path to executable:", style=PROMPT_STYLE).ask()
                if selected == "[Enter custom path...]"
                else selected
            )
        else:
            target_str = questionary.path("Path to script or executable:", style=PROMPT_STYLE).ask()

        if not target_str:
            sys.exit(0)

    target_path = Path(target_str).expanduser().resolve()
    if not target_path.is_file():
        console.print(f"[bold red]✘ Error:[/bold red] Target file '{target_str}' does not exist.")
        sys.exit(1)

    exec_type = (
        "python"
        if target_path.suffix == ".py"
        else ("bash" if target_path.suffix == ".sh" else "binary")
    )

    # 2. Discover Virtual Environment
    venv_path: Optional[Path] = None
    explicit_venv_used = bool(args.venv)
    if not args.no_venv:
        venv_path = discover_project_venv(cwd, args.venv)
        if exec_type == "python" and not venv_path:
            console.print("[yellow]⚠ No local Python virtual environment (.venv/venv) detected.[/yellow]")
            if interactive_mode or sys.stdin.isatty():
                custom_venv = questionary.path(
                    "Enter path to venv (or leave empty to use system Python):",
                    style=PROMPT_STYLE,
                ).ask()
                if custom_venv is None:
                    sys.exit(0)
                if custom_venv.strip():
                    venv_path = discover_project_venv(cwd, custom_venv.strip())
                    explicit_venv_used = True

    # 3. Resolve Architecture & Hardware Resources
    arch = args.arch
    if not arch:
        interactive_mode = True
        arch_choices = [
            questionary.Choice(title=str(spec["title"]), value=k)
            for k, spec in ARCH_SPECS.items()
        ]
        arch = questionary.select(
            "Select Hermes 2 Target Architecture (+Architecture):",
            choices=arch_choices,
            style=PROMPT_STYLE,
        ).ask()
        if not arch:
            sys.exit(0)

    cpus: Optional[int] = args.cpus
    mem_mb: Optional[int] = None
    if args.mem:
        try:
            mem_mb = parse_memory_to_mb(args.mem)
        except ValueError as exc:
            console.print(f"[bold red]✘ Invalid --mem value '{args.mem}':[/bold red] {exc}")
            sys.exit(1)

    gpus: Optional[int] = args.gpus

    if arch == "GPU3090":
        if cpus is not None or mem_mb is not None or gpus is not None:
            console.print(
                "[dim cyan]ℹ Note: GPU3090 locks the entire node automatically; omitting request_cpus/request_memory/request_gpus.[/dim cyan]"
            )
        cpus, mem_mb, gpus = None, None, None
    elif arch == "A100":
        if cpus is not None:
            console.print(
                "[dim cyan]ℹ Note: CPU count is managed automatically on A100 nodes; omitting request_cpus.[/dim cyan]"
            )
        cpus = None
        if gpus is None:
            if interactive_mode:
                g_ans = questionary.select(
                    "Number of NVIDIA A100 (40GB) GPUs to request:",
                    choices=["1", "2", "3", "4"],
                    default="1",
                    style=PROMPT_STYLE,
                ).ask()
                if not g_ans:
                    sys.exit(0)
                gpus = int(g_ans)
            else:
                gpus = 1

        if mem_mb is None and interactive_mode:
            m_ans = questionary.text(
                "System RAM request (Leave blank for default <=64G, or e.g. '120G'):",
                default="",
                validate=lambda v: is_valid_memory_input(v, allow_empty=True)
                or "Enter valid size (e.g. '120G', '96GB', '98304M') or leave blank",
                style=PROMPT_STYLE,
            ).ask()
            if m_ans is None:
                sys.exit(0)
            if m_ans.strip():
                mem_mb = parse_memory_to_mb(m_ans.strip())

        if mem_mb is not None and mem_mb <= 65536:
            console.print(
                "[dim cyan]ℹ Note: A100 nodes allocate up to 64 GB RAM by default; omitting request_memory (<= 64 GB).[/dim cyan]"
            )
            mem_mb = None
    else:
        if gpus is not None:
            console.print(f"[dim cyan]ℹ Note: Ignoring --gpus on CPU architecture '{arch}'.[/dim cyan]")
            gpus = None

        if cpus is None:
            if interactive_mode:
                c_ans = questionary.text(
                    "Number of CPU cores (request_cpus):",
                    default="4",
                    validate=lambda v: (v.isdigit() and int(v) > 0) or "Enter a positive integer",
                    style=PROMPT_STYLE,
                ).ask()
                if not c_ans:
                    sys.exit(0)
                cpus = int(c_ans)
            else:
                cpus = 4

        if mem_mb is None:
            if interactive_mode:
                m_ans = questionary.text(
                    "System RAM (e.g., '8G', '16G', '4096M'):",
                    default="8G",
                    validate=lambda v: is_valid_memory_input(v, allow_empty=False)
                    or "Enter valid size (e.g. '8G', '32GB', '4096M')",
                    style=PROMPT_STYLE,
                ).ask()
                if not m_ans:
                    sys.exit(0)
                mem_mb = parse_memory_to_mb(m_ans)
            else:
                mem_mb = 8192

    # 4. Resolve Arguments, Sweeps & Tag
    combined_single_args = " ".join(
        x for x in [args.args.strip(), passthrough_args.strip()] if x
    ).strip()

    raw_sweep_inputs: List[str] = list(args.sweep)
    if args.sweep_file:
        sf_path = Path(args.sweep_file).expanduser().resolve()
        if not sf_path.is_file():
            console.print(f"[bold red]✘ Sweep file not found:[/bold red] {sf_path}")
            sys.exit(1)
        raw_sweep_inputs.extend(
            [
                line.strip()
                for line in sf_path.read_text().splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
        )

    expanded_sweeps: List[str] = []
    for item in raw_sweep_inputs:
        expanded_sweeps.extend(expand_arg_sweeps(item))

    if expanded_sweeps:
        base_variants = expand_arg_sweeps(combined_single_args) if combined_single_args else [""]
        sweep_list = [f"{b} {s}".strip() for b in base_variants for s in expanded_sweeps]
    else:
        if interactive_mode and not combined_single_args:
            ans_args = questionary.text(
                "Arguments (Supports sweeps via '{0..4}', '{a,b}', or ';' separator):",
                default="",
                style=PROMPT_STYLE,
            ).ask()
            if ans_args is None:
                sys.exit(0)
            combined_single_args = ans_args.strip()
        sweep_list = expand_arg_sweeps(combined_single_args)

    for s_arg in sweep_list:
        if s_arg:
            format_condor_arg_line(s_arg)

    raw_tag = args.tag.strip()
    if interactive_mode and not raw_tag:
        ans_tag = questionary.text(
            "Optional run tag for folder name (e.g., 'exp1_lr1e4', or Enter to skip):",
            default="",
            style=PROMPT_STYLE,
        ).ask()
        if ans_tag is None:
            sys.exit(0)
        raw_tag = ans_tag.strip()

    tag = sanitize_tag(raw_tag)

    # 5. Build Reproducible CLI String
    rel_target = target_path.relative_to(cwd) if target_path.is_relative_to(cwd) else target_path
    cli_parts = ["send", shlex.quote(str(rel_target)), "-a", arch]
    if cpus:
        cli_parts.extend(["-c", str(cpus)])
    if mem_mb:
        mem_flag = f"{mem_mb // 1024}G" if mem_mb % 1024 == 0 else f"{mem_mb}M"
        cli_parts.extend(["-m", mem_flag])
    if gpus:
        cli_parts.extend(["-g", str(gpus)])
    if tag:
        cli_parts.extend(["-t", shlex.quote(tag)])
    if args.no_venv:
        cli_parts.append("--no-venv")
    elif explicit_venv_used and venv_path:
        cli_parts.extend(["--venv", shlex.quote(str(venv_path))])
    for n in args.node:
        cli_parts.extend(["--node", shlex.quote(n)])
    for ex in args.exclude:
        cli_parts.extend(["--exclude", shlex.quote(ex)])
    if args.project:
        cli_parts.extend(["--project", shlex.quote(args.project)])
    if args.scratch:
        cli_parts.append("--scratch")
        cli_parts.extend(shlex.quote(sf) for sf in args.scratch)
    if len(sweep_list) > 1:
        for s in sweep_list:
            cli_parts.extend(["-s", shlex.quote(s)])
    elif sweep_list and sweep_list[0]:
        cli_parts.extend(["--", sweep_list[0]])

    equivalent_cli = " ".join(cli_parts)

    # 6. Prepare Paths & Display Summary Panel
    now = datetime.datetime.now()
    submit_epoch = int(now.timestamp())
    ts_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    safe_tag = f"_{tag}" if tag else ""
    run_id = f"{ts_str}_{target_path.stem}_{arch}{safe_tag}"
    runs_root = (cwd / "runs").resolve()
    run_dir = runs_root / run_id
    repro_dir = run_dir / "repro"

    git_info = get_git_info(cwd)

    summary_table = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
    summary_table.add_column("Property", style="bold cyan", width=20)
    summary_table.add_column("Value", style="white")

    summary_table.add_row("Target Executable", f"[bold]{rel_target}[/bold] ({exec_type})")
    summary_table.add_row("Working Directory", str(cwd))
    summary_table.add_row(
        "Run Output Folder",
        f"[green]runs/{run_id}/[/green] (exported as [bold]$HERMES_RUN_DIR[/bold])",
    )
    summary_table.add_row("Python Venv", str(venv_path) if venv_path else "[dim]None (System)[/dim]")
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
        preview_sweeps = "\n".join(
            [f"  [dim]Process {i}:[/dim] {s}" for i, s in enumerate(sweep_list[:5])]
        )
        if len(sweep_list) > 5:
            preview_sweeps += f"\n  [dim]... (+{len(sweep_list) - 5} more parallel jobs)[/dim]"
        summary_table.add_row(
            "Job Mode",
            f"[bold yellow]Parallel Array Sweep ({len(sweep_list)} jobs)[/bold yellow]\n{preview_sweeps}",
        )
    else:
        summary_table.add_row("Arguments", sweep_list[0] if sweep_list[0] else "[dim](none)[/dim]")

    if git_info["is_git"]:
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

    if not args.yes and not args.dry_run:
        confirm = questionary.confirm(
            "Submit job to HTCondor queue now?", default=True, style=PROMPT_STYLE
        ).ask()
        if not confirm:
            console.print("[yellow]Submission cancelled.[/yellow]")
            sys.exit(0)

    # 7. Materialize Run Directory & Repro Capsule
    repro_dir.mkdir(parents=True, exist_ok=True)
    snap_exec_path, snap_py_root = create_code_snapshot(target_path, cwd, repro_dir, git_info)

    patch_rel_path: Optional[str] = None
    if git_info.get("is_git") and git_info.get("diff"):
        (repro_dir / "uncommitted.patch").write_text(str(git_info["diff"]))
        patch_rel_path = "repro/uncommitted.patch"

    freeze_python_env(venv_path, repro_dir)

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
    )
    sh_path.write_text(sh_content)
    sh_path.chmod(0o755)

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
        nodes_include=args.node,
        nodes_exclude=args.exclude,
        project_code=args.project,
        scratch_files=args.scratch,
    )
    sub_path.write_text(sub_content)

    is_sweep = len(sweep_list) > 1
    run_json_data: Dict[str, Any] = {
        "run_id": run_id,
        "status": "DRY_RUN" if args.dry_run else "QUEUED",
        "submission": {
            "submitted_at": now.isoformat(timespec="seconds"),
            "user": os.environ.get("USER", "unknown"),
            "submit_host": os.uname().nodename,
            "target": str(rel_target),
            "snapshot_executable": str(snap_exec_path.relative_to(run_dir)),
            "working_directory": str(cwd),
            "run_directory": str(run_dir),
            "venv": str(venv_path) if venv_path else None,
            "equivalent_cli": equivalent_cli,
            "cluster_id": None,
        },
        "resources_requested": {
            "architecture": arch,
            "cpus": cpus,
            "memory_mb": mem_mb,
            "gpus": 1 if arch == "GPU3090" else gpus,
            "sweep_count": len(sweep_list),
            "sweep_arguments": sweep_list,
            "node_include": args.node,
            "node_exclude": args.exclude,
            "project_code": args.project,
            "scratch_files": args.scratch,
        },
        "git": {
            "is_git": git_info["is_git"],
            "branch": git_info["branch"],
            "commit": git_info["commit"],
            "dirty": git_info["dirty"],
            "patch_file": patch_rel_path,
        },
    }
    if is_sweep:
        run_json_data["executions"] = []
    else:
        run_json_data["execution"] = None

    run_json_path = run_dir / "run.json"
    atomic_write_json(run_json_path, run_json_data, repro_dir)

    if args.dry_run:
        console.print(
            f"\n[bold yellow]✔ Dry run complete![/bold yellow] Generated capsule in: [cyan]runs/{run_id}/[/cyan]"
        )
        console.print(Syntax(sub_content, "ini", theme="monokai", line_numbers=True))
        sys.exit(0)

    # 8. Submit to HTCondor
    try:
        res = subprocess.run(
            ["condor_submit", str(sub_path)],
            cwd=cwd,
            capture_output=True,
            text=True,
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
        update_latest_symlink(runs_root, run_dir)

        first_out = (
            f"runs/latest/{target_path.stem}_0.out"
            if is_sweep
            else f"runs/latest/{target_path.stem}.out"
        )
        first_err = (
            f"runs/latest/{target_path.stem}_0.err"
            if is_sweep
            else f"runs/latest/{target_path.stem}.err"
        )
        condor_q_cmd = f"condor_q {cluster_id}" if cluster_id else "condor_q"

        console.print(
            Panel(
                f"[bold green]✔ {condor_out}[/bold green]\n\n"
                f"  • [bold]Manage Run    :[/bold] [cyan]runs inspect latest[/cyan]\n"
                f"  • [bold]Monitor Queue :[/bold] [cyan]{condor_q_cmd}[/cyan]\n"
                f"  • [bold]Live Stdout   :[/bold] [cyan]tail -f {first_out}[/cyan]\n"
                f"  • [bold]Live Stderr   :[/bold] [cyan]tail -f {first_err}[/cyan]",
                title="[bold green]🎉 Job Submitted to Hermes 2[/bold green]",
                border_style="green",
                expand=False,
            )
        )

    except FileNotFoundError:
        locked_mutate_run_json(
            run_json_path,
            repro_dir,
            lambda d: d.update({"status": "SUBMIT_FAILED"}),
        )
        console.print(
            f"[bold red]✘ Error:[/bold red] 'condor_submit' not found in PATH (are you on hermes2?).\n"
            f"  Run directory and repro capsule were still generated at: [cyan]runs/{run_id}/[/cyan]"
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


if __name__ == "__main__":
    main()