"""
condor.py — Hermes 2 architecture specs, job.sh/job.sub generators, and live condor_q reconciliation.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_tools.core import console, get_username, locked_mutate_run_json

ARCH_SPECS: Dict[str, Dict[str, object]] = {
    "GPU3090": {
        "title": "GPU3090   │ NVIDIA RTX 3090 (24GB) — Exclusive full-node lock (Auto CPUs/RAM)",
        "needs_cpus": False,
        "needs_mem": False,
        "needs_gpus": False,
    },
    "A100": {
        "title": "A100      │ NVIDIA A100 (40GB)     — Request 1-4 GPUs (Custom RAM if >64GB)",
        "needs_cpus": False,
        "needs_mem": "optional_gt_64g",
        "needs_gpus": True,
    },
    "AMD": {
        "title": "AMD       │ AMD EPYC Pool          — General high-throughput CPU compute",
        "needs_cpus": True,
        "needs_mem": True,
        "needs_gpus": False,
    },
    "EPYC7662": {
        "title": "EPYC7662  │ AMD EPYC 7662          — Primary multi-core CPU architecture",
        "needs_cpus": True,
        "needs_mem": True,
        "needs_gpus": False,
    },
    "INTEL": {
        "title": "INTEL     │ Intel Xeon 6226R       — Intel-specific instruction sets / MKL",
        "needs_cpus": True,
        "needs_mem": True,
        "needs_gpus": False,
    },
    "UNDEFINED": {
        "title": "UNDEFINED │ Any Available Node     — Fastest queue dispatch (No arch filter)",
        "needs_cpus": True,
        "needs_mem": True,
        "needs_gpus": False,
    },
}

_JOB_SH_BODY = r"""
CONDOR_PROC_ID="${1:-0}"
shift || true

export HERMES_PROC_ID="${CONDOR_PROC_ID}"
RUN_JSON="${HERMES_RUN_DIR}/run.json"

START_EPOCH=$(date +%s)
START_ISO=$(date --iso-8601=seconds)
QUEUE_WAIT=$((START_EPOCH - SUBMIT_EPOCH))
if [ "${QUEUE_WAIT}" -lt 0 ]; then QUEUE_WAIT=0; fi

TIME_TMP=$(mktemp "/tmp/hermes_time_${USER:-user}_${CONDOR_PROC_ID}_XXXXXX.txt" 2>/dev/null || echo "${REPRO_DIR}/.time_${CONDOR_PROC_ID}.tmp")

# --- Deferred Signal Trap for Preemption / Eviction Auditing ---
CAUGHT_SIG=""
TRAP_FIRED=0
CHILD_PID=""
USED_USR_TIME=0

trap_handler() {
    CAUGHT_SIG="$1"
    TRAP_FIRED=1
    SHORT_SIG="${CAUGHT_SIG#SIG}"
    echo "" >&2
    echo "⚠️  [SIGNAL RECEIVED] Job caught ${CAUGHT_SIG} at $(date --iso-8601=seconds) on $(hostname -f 2>/dev/null || hostname)" >&2
    if [ -n "${CHILD_PID}" ] && kill -0 "${CHILD_PID}" 2>/dev/null; then
        if [ "${USED_USR_TIME}" -eq 1 ]; then
            # Forward signal to the workload child of /usr/bin/time so time still records peak RSS
            pkill -"${SHORT_SIG}" -P "${CHILD_PID}" 2>/dev/null || kill -s "${CAUGHT_SIG}" "${CHILD_PID}" 2>/dev/null || true
        else
            kill -s "${CAUGHT_SIG}" "${CHILD_PID}" 2>/dev/null || true
        fi
    fi
}
trap 'trap_handler SIGTERM' SIGTERM
trap 'trap_handler SIGINT' SIGINT
trap 'trap_handler SIGHUP' SIGHUP

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║ 🚀 HERMES 2 JOB EXECUTION TELEMETRY                                        ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo "  • Process ID       : ${CONDOR_PROC_ID} (of ${SWEEP_COUNT})"
echo "  • Started At       : ${START_ISO} (Queue wait: ${QUEUE_WAIT}s)"
echo "  • Execution Host   : $(hostname -f 2>/dev/null || hostname) ($(hostname -I 2>/dev/null | LC_ALL=C awk '{print $1}'))"
echo "  • OS / Kernel      : $(lsb_release -ds 2>/dev/null || grep PRETTY_NAME /etc/os-release 2>/dev/null | cut -d= -f2 | tr -d '"') | $(uname -sr)"
echo "  • Working Dir      : $(pwd)"
echo "  • Run Output Dir   : ${HERMES_RUN_DIR}"
echo "  • Snapshot Exec    : ${TARGET_EXEC}"
echo "  • Job Arguments    : $*"
echo ""

echo "─── [1/4] Node Hardware & Memory Profile ────────────────────────────────────"
CPU_MODEL=$(LC_ALL=C lscpu 2>/dev/null | LC_ALL=C awk -F: '/Model name/ {gsub(/^[ \t]+/, "", $2); print $2; exit}')
CPU_SOCKETS=$(LC_ALL=C lscpu 2>/dev/null | LC_ALL=C awk -F: '/Socket\(s\)/ {gsub(/^[ \t]+/, "", $2); print $2; exit}')
CPU_ALLOC=$(nproc 2>/dev/null || echo "0")
CPU_CORES=$(nproc --all 2>/dev/null || LC_ALL=C lscpu 2>/dev/null | LC_ALL=C awk -F: '/^CPU\(s\):/ {gsub(/^[ \t]+/, "", $2); print $2; exit}')
MEM_TOTAL=$(LC_ALL=C free -h 2>/dev/null | LC_ALL=C awk '/^Mem:/ {print $2}')
MEM_AVAIL=$(LC_ALL=C free -h 2>/dev/null | LC_ALL=C awk '/^Mem:/ {print $7}')

echo "  • CPU Architecture : ${CPU_MODEL:-Unknown} (${CPU_ALLOC} allocated / ${CPU_CORES:-?} logical CPUs across ${CPU_SOCKETS:-1} socket(s))"
echo "  • Node System RAM  : ${MEM_AVAIL:-?} available / ${MEM_TOTAL:-?} total"
echo "  • Scratch / Disk   : $(LC_ALL=C df -h . 2>/dev/null | LC_ALL=C awk 'NR==2 {print $4 " free on " $1 " (" $5 " used)"}')"

GPU_SUMMARY="None"
if command -v nvidia-smi &>/dev/null && ( [ -n "${CUDA_VISIBLE_DEVICES:-}" ] || nvidia-smi -L &>/dev/null ); then
    echo ""
    echo "─── [2/4] GPU Acceleration Telemetry ────────────────────────────────────────"
    echo "  • CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-All/Unset}"
    GPU_SUMMARY=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -n 1 || echo "NVIDIA GPU")
    nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.free,driver_version,temperature.gpu,pstate --format=csv,noheader 2>/dev/null | \
        LC_ALL=C awk -F', ' '{printf "  • GPU [%s] %s | VRAM: %s free / %s total | Temp: %sC | P-State: %s | Driver: %s\n    UUID: %s\n", $1, $2, $5, $4, $7, $8, $6, $3}'
    echo ""
    nvidia-smi
else
    echo ""
    echo "─── [2/4] GPU Acceleration Telemetry ────────────────────────────────────────"
    echo "  • No GPU allocated or nvidia-smi unavailable on this worker."
fi

echo ""
echo "─── [3/4] Environment & Toolchain Setup ─────────────────────────────────────"
if [ -n "${SNAP_PY_ROOT}" ]; then
    export PYTHONPATH="${SNAP_PY_ROOT}:${PROJECT_CWD}${PYTHONPATH:+:${PYTHONPATH}}"
else
    export PYTHONPATH="${PROJECT_CWD}${PYTHONPATH:+:${PYTHONPATH}}"
fi

if [ -n "${REQUESTED_CPUS}" ]; then
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${REQUESTED_CPUS}}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${REQUESTED_CPUS}}"
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${REQUESTED_CPUS}}"
    export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${REQUESTED_CPUS}}"
fi

echo "  • PYTHONPATH       : ${PYTHONPATH}"
echo "  • HERMES_RUN_DIR   : ${HERMES_RUN_DIR}"
if [ -n "${OMP_NUM_THREADS:-}" ]; then
    echo "  • Thread Pool Cap  : OMP=${OMP_NUM_THREADS} | MKL=${MKL_NUM_THREADS:-unset} | OPENBLAS=${OPENBLAS_NUM_THREADS:-unset}"
fi

SETUP_OK=1
EXIT_CODE=0
if [ -n "${VENV_PATH}" ]; then
    if [ -f "${VENV_PATH}/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "${VENV_PATH}/bin/activate"
        echo "  • Virtual Env      : ${VENV_PATH}"
        echo "  • Python Binary    : $(command -v python) ($(python --version 2>&1))"
    else
        echo "❌ ERROR: Virtual environment activation script missing at ${VENV_PATH}/bin/activate" >&2
        SETUP_OK=0
        EXIT_CODE=1
    fi
else
    echo "  • Virtual Env      : None (Native/System execution)"
fi

if [ "${SETUP_OK}" -eq 1 ]; then
    echo ""
    echo "─── [4/4] Executing Workload ────────────────────────────────────────────────"
    case "${EXEC_TYPE}" in
        python) CMD=(python -u "${TARGET_EXEC}" "$@") ;;
        bash)   CMD=(bash "${TARGET_EXEC}" "$@") ;;
        *)      CMD=("${TARGET_EXEC}" "$@") ;;
    esac
    echo "▶ Command: ${CMD[*]}"
    echo "─────────────────────────────────────────────────────────────────────────────"

    if [ -x /usr/bin/time ]; then
        USED_USR_TIME=1
        LC_MESSAGES=C LC_NUMERIC=C /usr/bin/time -v -o "${TIME_TMP}" "${CMD[@]}" &
    else
        USED_USR_TIME=0
        "${CMD[@]}" &
    fi
    CHILD_PID=$!

    while true; do
        TRAP_FIRED=0
        wait "${CHILD_PID}"
        EXIT_CODE=$?
        if [ "${TRAP_FIRED}" -eq 0 ]; then
            break
        fi
    done

    if [ -n "${CAUGHT_SIG}" ] && [ "${EXIT_CODE}" -eq 0 ]; then
        EXIT_CODE=143
    fi
else
    rm -f "${TIME_TMP}"
fi

END_EPOCH=$(date +%s)
END_ISO=$(date --iso-8601=seconds)
DURATION=$((END_EPOCH - START_EPOCH))
HOURS=$((DURATION / 3600))
MINS=$(((DURATION % 3600) / 60))
SECS=$((DURATION % 60))

MAX_RSS_KB="0"
MAX_RSS_MB="0.0"
CPU_PCT="N/A"
MAJ_FAULTS="0"
VOL_CTX="0"
INVOL_CTX="0"

if [ -f "${TIME_TMP}" ]; then
    MAX_RSS_KB=$(LC_ALL=C awk -F: '/Maximum resident set size/ {gsub(/^[ \t]+/, "", $2); print $2}' "${TIME_TMP}")
    CPU_PCT=$(LC_ALL=C awk -F: '/Percent of CPU this job got/ {gsub(/^[ \t]+/, "", $2); print $2}' "${TIME_TMP}")
    MAJ_FAULTS=$(LC_ALL=C awk -F: '/Major \(requiring I\/O\) page faults/ {gsub(/^[ \t]+/, "", $2); print $2}' "${TIME_TMP}")
    VOL_CTX=$(LC_ALL=C awk -F: '/Voluntary context switches/ {gsub(/^[ \t]+/, "", $2); print $2}' "${TIME_TMP}")
    INVOL_CTX=$(LC_ALL=C awk -F: '/Involuntary context switches/ {gsub(/^[ \t]+/, "", $2); print $2}' "${TIME_TMP}")
    rm -f "${TIME_TMP}"
fi

echo "─────────────────────────────────────────────────────────────────────────────"
echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║ 📊 POST-MORTEM RESOURCE HIGH-WATERMARKS & SUMMARY                          ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo "  • Exit Code        : ${EXIT_CODE}${CAUGHT_SIG:+ (Signal: ${CAUGHT_SIG})}"
echo "  • Finished At      : ${END_ISO}"
LC_ALL=C printf "  • Wall-Clock Time  : %02dh %02dm %02ds (%d seconds | Queue wait: %ds)\n" "${HOURS}" "${MINS}" "${SECS}" "${DURATION}" "${QUEUE_WAIT}"

if [ -n "${MAX_RSS_KB}" ] && [ "${MAX_RSS_KB}" -gt 0 ] 2>/dev/null; then
    MAX_RSS_MB=$(LC_ALL=C awk -v kb="${MAX_RSS_KB}" 'BEGIN {printf "%.2f", kb / 1024}')
    MAX_RSS_GB=$(LC_ALL=C awk -v kb="${MAX_RSS_KB}" 'BEGIN {printf "%.2f", kb / 1048576}')
    echo "  • Peak RAM (MaxRSS): ${MAX_RSS_MB} MB (${MAX_RSS_GB} GB)"
fi
echo "  • Average CPU Load : ${CPU_PCT}"
echo "  • I/O Page Faults  : ${MAJ_FAULTS} major faults"
echo "  • Context Switches : ${VOL_CTX} voluntary / ${INVOL_CTX} involuntary"

# --- Atomically Update Unified run.json (Env-isolated, NFS-safe flock) ---
export UPD_RUN_JSON="${RUN_JSON}"
export UPD_REPRO_DIR="${REPRO_DIR}"
export UPD_SWEEP_COUNT="${SWEEP_COUNT}"
export UPD_PROC_ID="${CONDOR_PROC_ID}"
export UPD_ARGS="$*"
export UPD_HOST="$(hostname -f 2>/dev/null || hostname)"
export UPD_CPU_MODEL="${CPU_MODEL}"
export UPD_CPU_ALLOC="${CPU_ALLOC:-0}"
export UPD_CPU_NODE="${CPU_CORES:-0}"
export UPD_GPU="${GPU_SUMMARY}"
export UPD_START="${START_ISO}"
export UPD_END="${END_ISO}"
export UPD_WAIT="${QUEUE_WAIT}"
export UPD_DUR="${DURATION}"
export UPD_EXIT="${EXIT_CODE}"
export UPD_SIG="${CAUGHT_SIG}"
export UPD_RSS_MB="${MAX_RSS_MB:-0.0}"
export UPD_CPU_PCT="${CPU_PCT}"
export UPD_FAULTS="${MAJ_FAULTS:-0}"

if [ ! -x "${UPDATER_PYTHON}" ]; then
    UPDATER_PYTHON="$(command -v python3 || command -v python || echo python3)"
fi

"${UPDATER_PYTHON}" - <<'PYEOF'
import fcntl
import json
import os
from pathlib import Path

run_json_path = Path(os.environ["UPD_RUN_JSON"])
repro_dir = Path(os.environ["UPD_REPRO_DIR"])
sweep_count = int(os.environ["UPD_SWEEP_COUNT"])
proc_id = int(os.environ["UPD_PROC_ID"])
exit_code = int(os.environ["UPD_EXIT"])
caught_sig = os.environ.get("UPD_SIG", "")


def safe_int(val: str) -> int:
    try:
        return int(val.strip())
    except (ValueError, AttributeError):
        return 0


def safe_float(val: str) -> float:
    try:
        return float(val.strip().replace(",", "."))
    except (ValueError, AttributeError):
        return 0.0


exec_record = {
    "process_id": proc_id,
    "arguments": os.environ.get("UPD_ARGS", ""),
    "hostname": os.environ.get("UPD_HOST", ""),
    "cpu_model": os.environ.get("UPD_CPU_MODEL", ""),
    "allocated_cpus": safe_int(os.environ.get("UPD_CPU_ALLOC", "0")),
    "logical_cpus_on_node": safe_int(os.environ.get("UPD_CPU_NODE", "0")),
    "gpu_model": os.environ.get("UPD_GPU", "None"),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    "started_at": os.environ.get("UPD_START", ""),
    "finished_at": os.environ.get("UPD_END", ""),
    "queue_wait_seconds": safe_int(os.environ.get("UPD_WAIT", "0")),
    "runtime_seconds": safe_int(os.environ.get("UPD_DUR", "0")),
    "exit_code": exit_code,
    "signal": caught_sig or None,
    "peak_ram_mb": safe_float(os.environ.get("UPD_RSS_MB", "0.0")),
    "cpu_utilization": os.environ.get("UPD_CPU_PCT", "N/A"),
    "major_page_faults": safe_int(os.environ.get("UPD_FAULTS", "0")),
}

if sweep_count > 1:
    node_tmp = repro_dir / f".node_{proc_id}.json.tmp.{os.getpid()}"
    node_tmp.write_text(json.dumps(exec_record, indent=2) + "\n", encoding="utf-8")
    os.replace(node_tmp, repro_dir / f"node_{proc_id}.json")

lock_path = repro_dir / ".run_json.lock"
with open(lock_path, "a", encoding="utf-8") as lf:
    fcntl.flock(lf, fcntl.LOCK_EX)
    try:
        data = (
            json.loads(run_json_path.read_text(encoding="utf-8"))
            if run_json_path.exists()
            else {}
        )
        if sweep_count == 1:
            if caught_sig:
                data["status"] = f"INTERRUPTED ({caught_sig})"
            else:
                data["status"] = "COMPLETED" if exit_code == 0 else "FAILED"
            data["execution"] = exec_record
        else:
            node_files = sorted(repro_dir.glob("node_*.json"))
            executions = [json.loads(nf.read_text(encoding="utf-8")) for nf in node_files]
            executions.sort(key=lambda x: x.get("process_id", 0))
            data["executions"] = executions
            if len(executions) < sweep_count:
                data["status"] = f"RUNNING ({len(executions)}/{sweep_count} finished)"
            else:
                any_sig = any(bool(x.get("signal")) for x in executions)
                any_failed = any(x.get("exit_code", 0) != 0 for x in executions)
                if any_sig:
                    data["status"] = "INTERRUPTED"
                elif any_failed:
                    data["status"] = "COMPLETED_WITH_ERRORS"
                else:
                    data["status"] = "COMPLETED"

        tmp_json = repro_dir / f".run.json.tmp.{proc_id}.{os.getpid()}"
        tmp_json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp_json, run_json_path)
    finally:
        fcntl.flock(lf, fcntl.LOCK_UN)
PYEOF

exit "${EXIT_CODE}"
"""


def normalize_fqdn(node: str) -> str:
    n = node.strip()
    return n if n.endswith(".i3a.es") else f"{n}.i3a.es"


def format_condor_tokens(tokens: List[str]) -> str:
    parts: List[str] = []
    for tok in tokens:
        escaped = tok.replace('"', '""')
        if tok == "" or any(c.isspace() or c == "'" for c in tok):
            escaped = "'" + escaped.replace("'", "''") + "'"
        parts.append(escaped)
    return " ".join(parts)


def format_condor_arg_line(arg_line: str) -> str:
    """Parses a shell-style argument string and re-encodes it for HTCondor."""
    s = arg_line.strip()
    if not s:
        return ""
    try:
        tokens = shlex.split(s)
    except ValueError as exc:
        console.print(f"[bold red]✘ Unbalanced quotes in arguments:[/bold red] {arg_line}\n  {exc}")
        sys.exit(1)
    return format_condor_tokens(tokens)


def generate_bash_wrapper(
    exec_type: str,
    target_exec_path: Path,
    venv_path: Optional[Path],
    pythonpath_dir: Optional[Path],
    project_cwd: Path,
    run_dir: Path,
    repro_dir: Path,
    submit_epoch: int,
    sweep_count: int,
    requested_cpus: Optional[int],
) -> str:
    updater_python = Path(sys.executable).resolve()
    header = (
        "#!/usr/bin/env bash\n"
        "# =============================================================================\n"
        "# Hermes 2 Execution Wrapper & Telemetry Auditor (Generated by hermes-tools)\n"
        "# =============================================================================\n"
        "set -o pipefail\n\n"
        f"EXEC_TYPE={shlex.quote(exec_type)}\n"
        f"TARGET_EXEC={shlex.quote(target_exec_path.as_posix())}\n"
        f"VENV_PATH={shlex.quote(venv_path.as_posix() if venv_path else '')}\n"
        f"SNAP_PY_ROOT={shlex.quote(pythonpath_dir.as_posix() if pythonpath_dir else '')}\n"
        f"PROJECT_CWD={shlex.quote(project_cwd.as_posix())}\n"
        f"REQUESTED_CPUS={shlex.quote(str(requested_cpus) if requested_cpus is not None else '')}\n"
        f"export HERMES_RUN_DIR={shlex.quote(run_dir.as_posix())}\n"
        f"REPRO_DIR={shlex.quote(repro_dir.as_posix())}\n"
        f'SUBMIT_EPOCH="${{SUBMIT_EPOCH:-{submit_epoch}}}"\n'
        f"SWEEP_COUNT={sweep_count}\n"
        f"UPDATER_PYTHON={shlex.quote(updater_python.as_posix())}\n"
    )
    return header + _JOB_SH_BODY


def generate_sub_file(
    sh_path: Path,
    project_cwd: Path,
    run_dir: Path,
    repro_dir: Path,
    script_stem: str,
    arch: str,
    cpus: Optional[int],
    mem_mb: Optional[int],
    gpus: Optional[int],
    sweep_args: List[str],
    nodes_include: List[str],
    nodes_exclude: List[str],
    project_code: Optional[str],
    scratch_files: List[str],
) -> str:
    is_sweep = len(sweep_args) > 1
    suffix = "_$(Process)" if is_sweep else ""

    lines = [
        "# =============================================================================",
        "# Hermes 2 HTCondor Submit File (Generated by hermes-tools)",
        "# =============================================================================",
        "Universe                = vanilla",
        f"Executable              = {sh_path.as_posix()}",
        f"Initialdir              = {project_cwd.as_posix()}",
        "",
        "# --- Hermes 2 Hardware & Architecture Directives ---",
        f'+Architecture           = "{arch}"',
    ]

    if arch == "GPU3090":
        lines.append("# Note: request_cpus & request_memory omitted for GPU3090 (full node lock)")
    elif arch == "A100":
        lines.append(f"request_gpus            = {gpus or 1}")
        if mem_mb and mem_mb > 65536:
            lines.append(f"request_memory          = {mem_mb}")
        else:
            lines.append("# Note: request_memory omitted for A100 (<= 64 GB default allocation)")
    else:
        lines.append(f"request_cpus            = {cpus or 4}")
        lines.append(f"request_memory          = {mem_mb or 8192}")

    req_clauses = []
    if nodes_include:
        inc_expr = " || ".join(f'(TARGET.Machine == "{normalize_fqdn(n)}")' for n in nodes_include)
        req_clauses.append(f"({inc_expr})")
    if nodes_exclude:
        exc_expr = " && ".join(f'(TARGET.Machine != "{normalize_fqdn(n)}")' for n in nodes_exclude)
        req_clauses.append(f"({exc_expr})")

    if req_clauses:
        req_joined = " && ".join(req_clauses)
        lines.append("")
        lines.append("# --- Target Machine Filtering ---")
        lines.append(f"requirements            = {req_joined}")

    if project_code:
        lines.append("")
        lines.append("# --- I3A Institutional Project Accounting ---")
        lines.append(f'+Proyecto               = "{project_code.strip()}"')

    if scratch_files:
        lines.append("")
        lines.append("# --- High-Speed RAMDisk Scratch Space Transfer ---")
        lines.append("should_transfer_files   = YES")
        lines.append("when_to_transfer_output = ON_EXIT")
        lines.append(f"transfer_input_files    = {', '.join(scratch_files)}")
        lines.append('transfer_output_files   = ""')

    lines.extend(
        [
            "",
            "# --- Top-Level Logs & Repro Scheduler Log ---",
            f"Output                  = {run_dir.as_posix()}/{script_stem}{suffix}.out",
            f"Error                   = {run_dir.as_posix()}/{script_stem}{suffix}.err",
            f"Log                     = {repro_dir.as_posix()}/condor.log",
            "",
            "# --- Job Queue & Arguments (HTCondor New Quoted Syntax) ---",
        ]
    )

    if not is_sweep:
        single_arg = format_condor_arg_line(sweep_args[0]) if sweep_args else ""
        if single_arg:
            lines.append(f'Arguments               = "$(Process) {single_arg}"')
        else:
            lines.append('Arguments               = "$(Process)"')
        lines.append("queue 1")
    else:
        lines.append('Arguments               = "$(Process) $(job_args)"')
        lines.append("queue job_args from (")
        for arg_line in sweep_args:
            encoded_line = format_condor_arg_line(arg_line)
            lines.append(f"    {encoded_line}")
        lines.append(")")

    return "\n".join(lines) + "\n"


def generate_retry_sub_file(
    orig_sub_path: Path,
    run_dir: Path,
    repro_dir: Path,
    script_stem: str,
    is_sweep: bool,
    targets: List[Tuple[int, str]],
    retry_epoch: int,
) -> str:
    """
    Builds `repro/job_retry.sub` with fresh SUBMIT_EPOCH and original proc ID mapping,
    preserving all hardware, architecture, node requirement, project, and scratch directives.
    """
    orig_sub = orig_sub_path.read_text(encoding="utf-8")
    header_lines: List[str] = []
    for line in orig_sub.splitlines():
        stripped = line.strip()
        if stripped.startswith(("environment", "Arguments", "queue", "# --- Job Queue")):
            break
        if stripped.startswith(("Output", "Error", "Log", "# --- Top-Level Logs")):
            continue
        header_lines.append(line)

    # Trim trailing blank lines from header_lines
    while header_lines and not header_lines[-1].strip():
        header_lines.pop()

    common_lines = header_lines + [
        "",
        "# --- Top-Level Logs & Repro Scheduler Log (In-Place Retry) ---",
        f"Log                     = {repro_dir.as_posix()}/condor.log",
        f'environment             = "SUBMIT_EPOCH={retry_epoch}"',
    ]

    if not is_sweep:
        single_encoded = format_condor_arg_line(targets[0][1]) if targets[0][1] else ""
        arg_str = f'"0 {single_encoded}"' if single_encoded else '"0"'
        retry_lines = common_lines + [
            f"Output                  = {run_dir.as_posix()}/{script_stem}.out",
            f"Error                   = {run_dir.as_posix()}/{script_stem}.err",
            "",
            "# --- Job Queue & Arguments (HTCondor New Quoted Syntax) ---",
            f"Arguments               = {arg_str}",
            "queue 1",
        ]
    else:
        retry_lines = common_lines + [
            f"Output                  = {run_dir.as_posix()}/{script_stem}_$(orig_proc).out",
            f"Error                   = {run_dir.as_posix()}/{script_stem}_$(orig_proc).err",
            "",
            "# --- Job Queue & Arguments (HTCondor New Quoted Syntax) ---",
            'Arguments               = "$(orig_proc) $(job_args)"',
            "queue orig_proc, job_args from (",
        ]
        for pid, raw_arg in targets:
            encoded = format_condor_arg_line(raw_arg)
            retry_lines.append(f"    {pid}, {encoded}" if encoded else f"    {pid},")
        retry_lines.append(")")

    return "\n".join(retry_lines) + "\n"


# ---------------------------------------------------------------------------
# Live HTCondor Queue & Multi-Cluster Reconciliation
# ---------------------------------------------------------------------------
def get_all_cluster_ids(data: Dict[str, Any]) -> List[int]:
    """Returns all HTCondor cluster IDs (initial + retries) associated with a run."""
    sub = data.get("submission", {})
    cids: List[int] = []
    for raw in sub.get("retry_clusters") or []:
        try:
            v = int(raw)
            if v not in cids:
                cids.append(v)
        except (ValueError, TypeError):
            pass
    main_cid = sub.get("cluster_id")
    if main_cid is not None:
        try:
            v = int(main_cid)
            if v not in cids:
                cids.append(v)
        except (ValueError, TypeError):
            pass
    return cids


def query_live_condor_clusters() -> Optional[Dict[int, Dict[str, int]]]:
    if not shutil.which("condor_q"):
        return None
    user = get_username()
    cmd = ["condor_q"]
    if user and user != "unknown":
        cmd.append(user)
    cmd.extend(["-json", "-attributes", "ClusterId,ProcId,JobStatus"])
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=6,
        )
        if res.returncode != 0:
            return None
        raw = res.stdout.strip()
        if not raw:
            return {}
        items = json.loads(raw)
        clusters: Dict[int, Dict[str, int]] = {}
        for job in items:
            cid = int(job.get("ClusterId", -1))
            st = int(job.get("JobStatus", 0))  # 1=Idle, 2=Running, 5=Held
            info = clusters.setdefault(cid, {"total": 0, "running": 0, "idle": 0, "held": 0})
            info["total"] += 1
            if st == 2:
                info["running"] += 1
            elif st == 1:
                info["idle"] += 1
            elif st == 5:
                info["held"] += 1
        return clusters
    except Exception:
        return None


def inspect_condor_log_fate(condor_log: Path, is_sweep: bool = False) -> Optional[str]:
    """Inspects events in `repro/condor.log` to determine why a job or sweep left `condor_q`."""
    if not condor_log.is_file():
        return None
    try:
        text = condor_log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    blocks = text.split("\n...\n")
    if not is_sweep:
        for block in reversed(blocks):
            if "Job was aborted" in block:
                return "ABORTED"
            if "Job terminated of its own accord" in block or "Normal termination" in block:
                if "return value 0" in block:
                    return None
                return "FAILED"
            if "Abnormal termination" in block or (
                "MemoryUsage" in block and "Hold reason" in block
            ):
                return "FAILED"
        return "VANISHED"

    # For multi-job sweeps that exited condor_q with missing node_*.json files:
    saw_abort = False
    saw_fail = False
    for block in blocks:
        if "Job was aborted" in block:
            saw_abort = True
        elif "Abnormal termination" in block or (
            "MemoryUsage" in block and "Hold reason" in block
        ):
            saw_fail = True
        elif (
            "Job terminated of its own accord" in block or "Normal termination" in block
        ) and "return value 0" not in block:
            saw_fail = True

    if saw_abort:
        return "ABORTED"
    if saw_fail:
        return "FAILED"
    return None


def reconcile_run(
    run_dir: Path,
    data: Dict[str, Any],
    live_clusters: Optional[Dict[int, Dict[str, int]]],
) -> Dict[str, Any]:
    """Synchronizes `run.json` with `repro/node_*.json` and all live `condor_q` clusters."""
    status = str(data.get("status", "UNKNOWN"))
    repro_dir = run_dir / "repro"
    run_json_path = run_dir / "run.json"

    sweep_count = int(data.get("resources_requested", {}).get("sweep_count", 1) or 1)
    synced_execs: Optional[List[Dict[str, Any]]] = None
    if sweep_count > 1 and repro_dir.is_dir():
        node_files = sorted(repro_dir.glob("node_*.json"))
        if node_files:
            loaded_execs: List[Dict[str, Any]] = []
            for nf in node_files:
                try:
                    loaded_execs.append(json.loads(nf.read_text(encoding="utf-8")))
                except Exception:
                    pass
            loaded_execs.sort(key=lambda x: x.get("process_id", 0))
            if loaded_execs != data.get("executions", []):
                synced_execs = loaded_execs
                data["executions"] = loaded_execs

    if live_clusters is None:
        return data

    all_cids = get_all_cluster_ids(data)
    active_cids = [c for c in all_cids if c in live_clusters]
    is_active_status = status in ("QUEUED", "RUNNING", "HELD") or status.startswith(
        ("QUEUED", "RUNNING", "HELD")
    )

    if active_cids:
        if synced_execs is not None:
            locked_mutate_run_json(
                run_json_path,
                repro_dir,
                lambda d: d.update({"executions": synced_execs}),
            )
        total_running = sum(live_clusters[c]["running"] for c in active_cids)
        total_held = sum(live_clusters[c]["held"] for c in active_cids)
        if total_held > 0:
            data["_live_badge"] = f"HELD ({total_held} held)"
        elif total_running > 0:
            if sweep_count > 1:
                done = len(data.get("executions", []))
                data["_live_badge"] = (
                    f"RUNNING ({total_running} active, {done}/{sweep_count} done)"
                )
            else:
                data["_live_badge"] = "RUNNING"
        else:
            data["_live_badge"] = "QUEUED"
        return data

    if is_active_status and all_cids:
        fate = inspect_condor_log_fate(repro_dir / "condor.log", is_sweep=(sweep_count > 1))
        if sweep_count > 1:
            execs = data.get("executions", [])
            if len(execs) >= sweep_count:
                any_sig = any(bool(x.get("signal")) for x in execs)
                any_err = any(x.get("exit_code", 0) != 0 for x in execs)
                if any_sig:
                    new_status = "INTERRUPTED"
                elif any_err:
                    new_status = "COMPLETED_WITH_ERRORS"
                else:
                    new_status = "COMPLETED"
            else:
                if fate == "ABORTED":
                    new_status = "ABORTED"
                elif fate == "FAILED":
                    new_status = f"FAILED ({len(execs)}/{sweep_count} finished)"
                else:
                    new_status = f"INCOMPLETE ({len(execs)}/{sweep_count})"
        else:
            if data.get("execution") is not None:
                ec = data["execution"].get("exit_code", 0)
                sig = data["execution"].get("signal")
                if sig:
                    new_status = f"INTERRUPTED ({sig})"
                else:
                    new_status = "COMPLETED" if ec == 0 else "FAILED"
            else:
                new_status = fate or "VANISHED"

        def _heal(d: Dict[str, Any]) -> None:
            if synced_execs is not None:
                d["executions"] = synced_execs
            if str(d.get("status", "")).startswith(("QUEUED", "RUNNING", "HELD")):
                d["status"] = new_status

        data = locked_mutate_run_json(run_json_path, repro_dir, _heal)
    elif synced_execs is not None:
        data = locked_mutate_run_json(
            run_json_path,
            repro_dir,
            lambda d: d.update({"executions": synced_execs}),
        )

    return data
