"""
condor.py — Hermes 2 architecture specs, job.sh/job.sub generators, and live condor_q reconciliation.
"""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_tools.core import (
    console,
    get_execution_records,
    get_username,
    load_run_json,
    locked_mutate_run_json,
    resolve_submit_python,
    set_execution_records,
)

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

# --- [1] Hardware & Environment Discovery ---
HOST_FQDN=$(hostname -f 2>/dev/null || hostname)
OS_NAME=$(lsb_release -ds 2>/dev/null || grep PRETTY_NAME /etc/os-release 2>/dev/null | cut -d= -f2 | tr -d '"')
KERNEL_VER=$(uname -sr 2>/dev/null)

LSCPU_OUT=$(LC_ALL=C lscpu 2>/dev/null || true)
CPU_MODEL=$(printf "%s\n" "${LSCPU_OUT}" | LC_ALL=C awk -F: '/Model name/ {gsub(/^[ \t]+/, "", $2); print $2; exit}')
CPU_ALLOC=$(nproc 2>/dev/null || echo "0")
CPU_CORES=$(nproc --all 2>/dev/null || printf "%s\n" "${LSCPU_OUT}" | LC_ALL=C awk -F: '/^CPU\(s\):/ {gsub(/^[ \t]+/, "", $2); print $2; exit}')

FREE_OUT=$(LC_ALL=C free -h 2>/dev/null || true)
MEM_TOTAL=$(printf "%s\n" "${FREE_OUT}" | LC_ALL=C awk '/^Mem:/ {print $2}')
MEM_AVAIL=$(printf "%s\n" "${FREE_OUT}" | LC_ALL=C awk '/^Mem:/ {print $7}')
DISK_INFO=$(LC_ALL=C df -h . 2>/dev/null | LC_ALL=C awk 'NR==2 {print $4 " free on " $1}')

GPU_SUMMARY="None"
GPU_LINES=""
if command -v nvidia-smi &>/dev/null && ( [ -n "${CUDA_VISIBLE_DEVICES:-}" ] || nvidia-smi -L &>/dev/null ); then
    GPU_SUMMARY=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -n 1 || echo "NVIDIA GPU")
    GPU_LINES=$(nvidia-smi --query-gpu=index,name,memory.free,memory.total,temperature.gpu,driver_version --format=csv,noheader 2>/dev/null | \
        LC_ALL=C awk -F', ' '{printf "  GPU      [%s] %s (%s / %s free · %s°C · Driver %s)\n", $1, $2, $3, $4, $5, $6}')
fi

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

if [ -n "${CUDA_HOME:-}" ]; then
    case ":${PATH}:" in
        *":${CUDA_HOME}/bin:"*) ;;
        *) export PATH="${CUDA_HOME}/bin:${PATH}" ;;
    esac
    case ":${LD_LIBRARY_PATH:-}:" in
        *":${CUDA_HOME}/lib64:"*) ;;
        *) export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
    esac
fi

SETUP_OK=1
EXIT_CODE=0
PYTHON_BIN=""
ENV_LABEL="System"
if [ -n "${VENV_PATH}" ]; then
    ENV_LABEL="${VENV_PATH##*/}"
    if [ -f "${VENV_PATH}/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "${VENV_PATH}/bin/activate"
        PYTHON_BIN="$(command -v python3 || command -v python)"
    elif [ -x "${VENV_PATH}/bin/python" ]; then
        export PATH="${VENV_PATH}/bin:${PATH}"
        PYTHON_BIN="${VENV_PATH}/bin/python"
    else
        echo "❌ ERROR: Virtual environment missing at ${VENV_PATH}/bin" >&2
        SETUP_OK=0
        EXIT_CODE=1
    fi
else
    if [ -n "${SUBMIT_PYTHON:-}" ] && [ -x "${SUBMIT_PYTHON}" ]; then
        PYTHON_BIN="${SUBMIT_PYTHON}"
    elif command -v python3 &>/dev/null; then
        PYTHON_BIN="$(command -v python3)"
    elif [ -n "${UPDATER_PYTHON:-}" ] && [ -x "${UPDATER_PYTHON}" ]; then
        PYTHON_BIN="${UPDATER_PYTHON}"
    else
        PYTHON_BIN="$(command -v python)"
    fi
    if [ -n "${PYTHON_BIN}" ]; then
        PY_DIR="${PYTHON_BIN%/*}"
        case ":${PATH}:" in
            *":${PY_DIR}:"*) ;;
            *) export PATH="${PY_DIR}:${PATH}" ;;
        esac
    fi
fi

PY_VER=""
if [ -n "${PYTHON_BIN}" ] && [ -x "${PYTHON_BIN}" ]; then
    PY_VER="$(${PYTHON_BIN} --version 2>&1)"
fi

TARGET_NAME="${TARGET_EXEC##*/}"
case "${EXEC_TYPE}" in
    python)
        CMD=("${PYTHON_BIN}" -u "${TARGET_EXEC}" "$@")
        SHORT_CMD="${PYTHON_BIN##*/} -u ${TARGET_NAME}${*:+ $*}"
        ;;
    bash)
        CMD=(bash "${TARGET_EXEC}" "$@")
        SHORT_CMD="bash ${TARGET_NAME}${*:+ $*}"
        ;;
    *)
        CMD=("${TARGET_EXEC}" "$@")
        SHORT_CMD="${TARGET_NAME}${*:+ $*}"
        ;;
esac

# --- [2] Compact Header ---
echo "── Hermes 2 Job · ${TARGET_NAME} [proc ${CONDOR_PROC_ID}/${SWEEP_COUNT}] ──────────────────────────────────────────────"
echo "  Host     ${HOST_FQDN} (${OS_NAME:-Linux} · ${KERNEL_VER})"
echo "  CPU/RAM  ${CPU_MODEL:-Unknown} (${CPU_ALLOC}/${CPU_CORES:-?} cores · ${MEM_AVAIL:-?}/${MEM_TOTAL:-?} free${OMP_NUM_THREADS:+ · OMP=${OMP_NUM_THREADS}})"
if [ -n "${GPU_LINES}" ]; then
    printf "%s\n" "${GPU_LINES}"
fi
echo "  Env      ${PY_VER:-${EXEC_TYPE}} (${ENV_LABEL})${DISK_INFO:+ · Disk: ${DISK_INFO}}"
if [ -n "${CUSTOM_ENV_SUMMARY:-}" ]; then
    echo "  Exports  ${CUSTOM_ENV_SUMMARY}"
fi
echo "  Run Dir  ${HERMES_RUN_DIR}"
echo "  Command  ${SHORT_CMD}"
echo "  Started  ${START_ISO} (queued ${QUEUE_WAIT}s)"
echo "────────────────────────────────────────────────────────────────────────────────"

if [ "${SETUP_OK}" -eq 1 ]; then
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

if [ -f "${TIME_TMP}" ]; then
    MAX_RSS_KB=$(LC_ALL=C awk -F: '/Maximum resident set size/ {gsub(/^[ \t]+/, "", $2); print $2}' "${TIME_TMP}")
    CPU_PCT=$(LC_ALL=C awk -F: '/Percent of CPU this job got/ {gsub(/^[ \t]+/, "", $2); print $2}' "${TIME_TMP}")
    MAJ_FAULTS=$(LC_ALL=C awk -F: '/Major \(requiring I\/O\) page faults/ {gsub(/^[ \t]+/, "", $2); print $2}' "${TIME_TMP}")
    rm -f "${TIME_TMP}"
fi

RAM_LABEL="N/A"
if [ -n "${MAX_RSS_KB}" ] && [ "${MAX_RSS_KB}" -gt 0 ] 2>/dev/null; then
    MAX_RSS_MB=$(LC_ALL=C awk -v kb="${MAX_RSS_KB}" 'BEGIN {printf "%.2f", kb / 1024}')
    if [ "${MAX_RSS_KB}" -ge 1048576 ] 2>/dev/null; then
        MAX_RSS_GB=$(LC_ALL=C awk -v kb="${MAX_RSS_KB}" 'BEGIN {printf "%.2f", kb / 1048576}')
        RAM_LABEL="${MAX_RSS_GB} GB"
    else
        RAM_LABEL="${MAX_RSS_MB} MB"
    fi
fi

if [ -n "${CAUGHT_SIG}" ]; then
    STATUS_WORD="Interrupted (${CAUGHT_SIG}, exit ${EXIT_CODE})"
elif [ "${EXIT_CODE}" -eq 0 ]; then
    STATUS_WORD="Completed (exit 0)"
else
    STATUS_WORD="Failed (exit ${EXIT_CODE})"
fi

echo "────────────────────────────────────────────────────────────────────────────────"
LC_ALL=C printf "── %s · %02dh %02dm %02ds · Peak RAM: %s · CPU: %s · Faults: %s\n" \
    "${STATUS_WORD}" "${HOURS}" "${MINS}" "${SECS}" "${RAM_LABEL}" "${CPU_PCT}" "${MAJ_FAULTS:-0}"

# --- Atomically Update Unified run.json (Env-isolated, NFS-safe flock) ---
export UPD_RUN_JSON="${RUN_JSON}"
export UPD_REPRO_DIR="${REPRO_DIR}"
export UPD_SWEEP_COUNT="${SWEEP_COUNT}"
export UPD_PROC_ID="${CONDOR_PROC_ID}"
export UPD_ARGS="$*"
export UPD_HOST="${HOST_FQDN}"
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
import random
import time
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

# Always write lock-free per-process node_<proc_id>.json atomically first
node_tmp = repro_dir / f".node_{proc_id}.json.tmp.{os.getpid()}"
node_tmp.write_text(json.dumps(exec_record, indent=2) + "\n", encoding="utf-8")
os.replace(node_tmp, repro_dir / f"node_{proc_id}.json")

# Zero-wait non-blocking flock: if another sweep worker holds the lock, exit in 0ms
# (our node_<proc_id>.json is already on disk and will be merged by the lock holder or reconcile_run)
lock_path = repro_dir / ".run_json.lock"
with open(lock_path, "a", encoding="utf-8") as lf:
    try:
        fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        pass
    else:
        try:
            data = (
                json.loads(run_json_path.read_text(encoding="utf-8"))
                if run_json_path.exists()
                else {}
            )
            data.pop("execution", None)
            if sweep_count == 1:
                data["executions"] = [exec_record]
                if caught_sig:
                    data["status"] = f"INTERRUPTED ({caught_sig})"
                else:
                    data["status"] = "COMPLETED" if exit_code == 0 else "FAILED"
            else:
                by_id = {
                    int(x.get("process_id", 0)): x
                    for x in data.get("executions", [])
                    if isinstance(x, dict)
                }
                by_id[proc_id] = exec_record
                # Only read node_<id>.json files not yet merged into data["executions"]
                for entry in os.scandir(repro_dir):
                    name = entry.name
                    if name.startswith("node_") and name.endswith(".json"):
                        try:
                            nid = int(name[5:-5])
                            if nid not in by_id:
                                by_id[nid] = json.loads(Path(entry.path).read_text(encoding="utf-8"))
                        except Exception:
                            pass
                executions = [by_id[k] for k in sorted(by_id)]
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


def _format_bash_export(key: str, val: str) -> str:
    """Formats an `export KEY=...` line, allowing shell `$VAR` expansion when `$` is present."""
    if "$" in val:
        escaped = val.replace('"', '\\"').replace("`", "\\`")
        return f'export {key}="{escaped}"'
    return f"export {key}={shlex.quote(val)}"


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
    env_vars: Optional[Dict[str, str]] = None,
) -> str:
    updater_python = Path(sys.executable).resolve()
    submit_python = resolve_submit_python()
    submit_home = Path.home().as_posix()

    merged_env: Dict[str, str] = {
        "PYTHONUNBUFFERED": "1",
        "HOME": submit_home,
    }
    custom_summary_parts: List[str] = []
    if env_vars:
        for k, v in env_vars.items():
            if k and v is not None:
                clean_k = str(k).strip()
                clean_v = str(v)
                merged_env[clean_k] = clean_v
                custom_summary_parts.append(f"{clean_k}={clean_v}")

    env_lines = "\n".join(_format_bash_export(k, v) for k, v in merged_env.items())
    custom_env_summary = " · ".join(custom_summary_parts)

    header = (
        "#!/usr/bin/env bash\n"
        "# =============================================================================\n"
        "# Hermes 2 Execution Wrapper & Telemetry Auditor (Generated by hermes-tools)\n"
        "# =============================================================================\n"
        "set -o pipefail\n\n"
        f"{env_lines}\n"
        f"CUSTOM_ENV_SUMMARY={shlex.quote(custom_env_summary)}\n"
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
        f"SUBMIT_PYTHON={shlex.quote(submit_python.as_posix())}\n"
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
    retry_targets: Optional[List[Tuple[int, str]]] = None,
    retry_epoch: Optional[int] = None,
) -> str:
    is_sweep = len(sweep_args) > 1

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
        clean_scratch = [sf.rstrip("/\\") for sf in scratch_files if sf.rstrip("/\\")]
        lines.append("")
        lines.append("# --- High-Speed RAMDisk Scratch Space Transfer ---")
        lines.append("should_transfer_files   = YES")
        lines.append("when_to_transfer_output = ON_EXIT")
        lines.append(f"transfer_input_files    = {', '.join(clean_scratch)}")
        lines.append('transfer_output_files   = ""')

    if retry_targets is not None:
        lines.extend(
            _format_retry_queue_section(
                run_dir=run_dir,
                repro_dir=repro_dir,
                script_stem=script_stem,
                is_sweep=is_sweep,
                targets=retry_targets,
                retry_epoch=retry_epoch or 0,
            )
        )
        return "\n".join(lines) + "\n"

    suffix = "_$(Process)" if is_sweep else ""
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


def _format_retry_queue_section(
    run_dir: Path,
    repro_dir: Path,
    script_stem: str,
    is_sweep: bool,
    targets: List[Tuple[int, str]],
    retry_epoch: int,
) -> List[str]:
    out = [
        "",
        "# --- Top-Level Logs & Repro Scheduler Log (In-Place Retry) ---",
        f"Log                     = {repro_dir.as_posix()}/condor.log",
        f'environment             = "SUBMIT_EPOCH={retry_epoch}"',
    ]
    if not is_sweep:
        single_encoded = format_condor_arg_line(targets[0][1]) if targets[0][1] else ""
        arg_str = f'"0 {single_encoded}"' if single_encoded else '"0"'
        out.extend(
            [
                f"Output                  = {run_dir.as_posix()}/{script_stem}.out",
                f"Error                   = {run_dir.as_posix()}/{script_stem}.err",
                "",
                "# --- Job Queue & Arguments (HTCondor New Quoted Syntax) ---",
                f"Arguments               = {arg_str}",
                "queue 1",
            ]
        )
    else:
        out.extend(
            [
                f"Output                  = {run_dir.as_posix()}/{script_stem}_$(orig_proc).out",
                f"Error                   = {run_dir.as_posix()}/{script_stem}_$(orig_proc).err",
                "",
                "# --- Job Queue & Arguments (HTCondor New Quoted Syntax) ---",
                'Arguments               = "$(orig_proc) $(job_args)"',
                "queue orig_proc, job_args from (",
            ]
        )
        for pid, raw_arg in targets:
            encoded = format_condor_arg_line(raw_arg)
            out.append(f"    {pid}, {encoded}" if encoded else f"    {pid},")
        out.append(")")
    return out


def build_send_command_tokens(
    *,
    send_bin: str = "send",
    target: str,
    arch: str,
    cpus: Optional[int] = None,
    mem_mb: Optional[int] = None,
    gpus: Optional[int] = None,
    tag: Optional[str] = None,
    no_venv: bool = False,
    venv: Optional[str] = None,
    nodes_include: Optional[List[str]] = None,
    nodes_exclude: Optional[List[str]] = None,
    project_code: Optional[str] = None,
    scratch_files: Optional[List[str]] = None,
    env_items: Optional[List[str]] = None,
    sweep_file: Optional[str] = None,
    sweep_specs: Optional[List[str]] = None,
    args_flag_val: Optional[str] = None,
    after: Optional[str] = None,
    after_ok: bool = False,
    yes: bool = False,
) -> List[str]:
    """Builds the canonical list of CLI tokens for `send` (used by both `send` and `runs retry --new-run`)."""
    cmd: List[str] = [send_bin, str(target), "-a", str(arch)]
    if cpus:
        cmd.extend(["-c", str(cpus)])
    if mem_mb:
        mem_str = f"{mem_mb // 1024}G" if mem_mb % 1024 == 0 else f"{mem_mb}M"
        cmd.extend(["-m", mem_str])
    if gpus and arch != "GPU3090":
        cmd.extend(["-g", str(gpus)])
    if tag:
        cmd.extend(["-t", str(tag)])
    if no_venv:
        cmd.append("--no-venv")
    elif venv:
        cmd.extend(["--venv", str(venv)])
    for n in nodes_include or []:
        cmd.extend(["--node", str(n)])
    for ex in nodes_exclude or []:
        cmd.extend(["--exclude", str(ex)])
    if project_code:
        cmd.extend(["--project", str(project_code)])
    if scratch_files:
        cmd.append("--scratch")
        cmd.extend(str(sf) for sf in scratch_files)
    for item in env_items or []:
        cmd.extend(["-e", str(item)])
    if sweep_file:
        cmd.extend(["--sweep-file", str(sweep_file)])
    for s in sweep_specs or []:
        cmd.extend(["-s", str(s)])
    if args_flag_val:
        cmd.append(f"--args={args_flag_val}")
    if after:
        cmd.extend(["--after", str(after)])
    if after_ok:
        cmd.append("--after-ok")
    if yes:
        cmd.append("-y")
    return cmd


_WAIT_PARENT_SCRIPT = r'''#!/usr/bin/env python3
"""wait_parent.py — HTCondor DAGMan PRE script to wait for a parent run to finish."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore[assignment]

# Ensure standard HTCondor and system paths are in PATH
for _p in ("/usr/bin", "/bin", "/usr/local/bin", "/usr/lib/condor/bin", "/usr/local/condor/bin"):
    if os.path.isdir(_p) and _p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{_p}:{os.environ.get('PATH', '')}"


def query_condor_jobs_for_cids(cids: Set[int], dagman_cids: Set[int]) -> List[Dict[str, Any]]:
    """Queries condor_q for any job matching specified ClusterIds or DAGManJobIds."""
    if not shutil.which("condor_q"):
        return []
    try:
        res = subprocess.run(
            ["condor_q", "-nobatch", "-json", "-attributes", "ClusterId,JobStatus,DAGManJobId"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        if res.returncode == 0 and res.stdout.strip():
            items = json.loads(res.stdout.strip())
            active = []
            for j in items:
                cid = int(j.get("ClusterId", -1))
                did = int(j.get("DAGManJobId") or -1)
                if cid in cids or (did > 0 and did in dagman_cids):
                    active.append(j)
            return active
    except Exception:
        pass
    return []


def query_single_cluster_active(cid: int) -> bool:
    """Checks if a single cluster ID is active in condor_q."""
    if not shutil.which("condor_q"):
        return False
    try:
        res = subprocess.run(
            ["condor_q", str(cid), "-nobatch", "-json", "-attributes", "ClusterId,JobStatus"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=8,
        )
        if res.returncode == 0 and res.stdout.strip():
            items = json.loads(res.stdout.strip())
            return bool(items)
    except Exception:
        pass
    return False


def get_parent_fate(parent_dir: Optional[Path], fallback_cid: Optional[int]) -> Tuple[bool, str]:
    """
    Returns (is_finished, fate_string).
    Handles dynamic parent cluster discovery, waiting states (A -> B -> C), and sweep executions.
    """
    if parent_dir is None:
        # Fallback for raw external cluster ID without local run.json
        if fallback_cid is None:
            return True, "UNKNOWN"
        if query_single_cluster_active(fallback_cid):
            return False, "RUNNING"
        # Check condor_history if available
        if shutil.which("condor_history"):
            try:
                res = subprocess.run(
                    ["condor_history", str(fallback_cid), "-json", "-attributes", "ExitCode,ExitStatus"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=8,
                )
                if res.returncode == 0 and res.stdout.strip():
                    items = json.loads(res.stdout.strip())
                    if items:
                        any_err = any(int(j.get("ExitCode", 0)) != 0 for j in items)
                        return True, "FAILED" if any_err else "COMPLETED"
            except Exception:
                pass
        return True, "COMPLETED"

    # --- Managed Run Directory ---
    rj_path = parent_dir / "run.json"
    data = {}
    if rj_path.is_file():
        try:
            data = json.loads(rj_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    status = str(data.get("status", "UNKNOWN"))
    dep = data.get("dependency", {})
    sub = data.get("submission", {})
    req = data.get("resources_requested", {})
    sweep_count = int(req.get("sweep_count", 1) or 1)

    # 1. Collect all known cluster IDs for parent
    cids: Set[int] = set()
    dagman_cids: Set[int] = set()
    if fallback_cid is not None:
        cids.add(fallback_cid)
    for k in ("job_cluster_id", "cluster_id"):
        v = dep.get(k) or sub.get(k)
        if v is not None:
            try:
                cids.add(int(v))
            except (ValueError, TypeError):
                pass
    for r in sub.get("retry_clusters") or []:
        try:
            cids.add(int(r))
        except (ValueError, TypeError):
            pass
    raw_dag = dep.get("dagman_cluster_id")
    if raw_dag is not None:
        try:
            did = int(raw_dag)
            cids.add(did)
            dagman_cids.add(did)
        except (ValueError, TypeError):
            pass

    # 2. Check if parent or its children are currently in condor_q
    if cids or dagman_cids:
        active_jobs = query_condor_jobs_for_cids(cids, dagman_cids)
        if active_jobs:
            return False, "RUNNING"

    # 3. If parent is WAITING and DAGMan is still active / launching
    repro_dir = parent_dir / "repro"
    dag_out = repro_dir / "job.dag.dagman.out"
    if status.startswith("WAITING"):
        # Parent compute job hasn't started yet!
        # Check if parent DAGMan is still running or if it terminated
        if dagman_cids and query_condor_jobs_for_cids(dagman_cids, set()):
            return False, "WAITING"
        # If DAGMan is not in condor_q, check if dagman.out shows terminal state
        if dag_out.is_file():
            try:
                txt = dag_out.read_text(encoding="utf-8", errors="replace")
                if "EXITING WITH STATUS" in txt or "failed with status" in txt:
                    # Parent DAGMan finished
                    pass
                else:
                    return False, "WAITING"
            except OSError:
                return False, "WAITING"
        else:
            return False, "WAITING"

    # 4. Check if parent's execution records are complete
    execs = list(data.get("executions") or [])
    if repro_dir.is_dir():
        node_files = sorted(repro_dir.glob("node_*.json"))
        if node_files:
            by_id = {int(x.get("process_id", 0)): x for x in execs if isinstance(x, dict)}
            for nf in node_files:
                try:
                    nd = json.loads(nf.read_text(encoding="utf-8"))
                    by_id[int(nd.get("process_id", 0))] = nd
                except Exception:
                    pass
            execs = [by_id[k] for k in sorted(by_id)]

    if len(execs) >= sweep_count and sweep_count > 0:
        any_sig = any(bool(x.get("signal")) for x in execs)
        any_err = any(int(x.get("exit_code", 0)) != 0 for x in execs)
        if any_sig:
            return True, "INTERRUPTED"
        if any_err:
            return True, "COMPLETED_WITH_ERRORS" if sweep_count > 1 else "FAILED"
        return True, "COMPLETED"

    # 5. Check if status is already terminal in run.json
    if status in (
        "COMPLETED",
        "FAILED",
        "COMPLETED_WITH_ERRORS",
        "INTERRUPTED",
        "ABORTED",
        "CANCELLED",
        "VANISHED",
        "SUBMIT_FAILED",
    ) or status.startswith(("INTERRUPTED", "CANCELLED", "FAILED", "COMPLETED_WITH_ERRORS")):
        return True, status

    # 6. Check condor.log or DAGMan out on disk
    condor_log = repro_dir / "condor.log"
    if condor_log.is_file():
        try:
            txt = condor_log.read_text(encoding="utf-8", errors="replace")
            if "Job was aborted" in txt:
                return True, "ABORTED"
            if "Abnormal termination" in txt:
                return True, "FAILED"
            if "Job terminated of its own accord" in txt or "Normal termination" in txt:
                if "return value 0" in txt:
                    return True, "COMPLETED"
                return True, "FAILED"
        except OSError:
            pass

    if dag_out.is_file():
        try:
            txt = dag_out.read_text(encoding="utf-8", errors="replace")
            if "failed with status" in txt or "EXITING WITH STATUS 1" in txt:
                return True, "FAILED"
            if "EXITING WITH STATUS 0" in txt:
                return True, "COMPLETED"
        except OSError:
            pass

    return False, status


def record_child_cancelled(child_dir: Optional[Path], parent_label: str, parent_fate: str) -> None:
    """Atomically updates child run.json to CANCELLED (parent failed) with lock."""
    if child_dir is None:
        return
    child_repro = child_dir / "repro"
    child_rj = child_dir / "run.json"
    child_lock = child_repro / ".run_json.lock"
    lock_f = None
    try:
        if fcntl is not None and child_lock.parent.is_dir():
            lock_f = open(child_lock, "a")
            fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            d = {}
            if child_rj.is_file():
                d = json.loads(child_rj.read_text(encoding="utf-8"))
            d["status"] = "CANCELLED (parent failed)"
            d["cancelled_reason"] = f"Parent run {parent_label} finished with status {parent_fate}"
            tmp_f = child_repro / f".run.json.tmp.cancel.{os.getpid()}"
            tmp_f.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp_f, child_rj)
        finally:
            if lock_f is not None:
                fcntl.flock(lock_f, fcntl.LOCK_UN)
                lock_f.close()
    except Exception as e:
        print(f"[WAIT_PARENT] Note: could not update child run.json: {e}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Wait for HTCondor parent run.")
    parser.add_argument("--child-dir", default=None, help="Path to child run directory")
    parser.add_argument("--parent-dir", default=None, help="Path to parent run directory")
    parser.add_argument("--parent-cluster", type=int, default=None, help="HTCondor cluster ID of parent")
    parser.add_argument("--parent-log", default=None, help="Path to parent condor.log")
    parser.add_argument("--require-ok", action="store_true", help="Abort if parent does not exit 0")
    parser.add_argument("--poll-interval", type=int, default=5, help="Polling interval in seconds")
    args = parser.parse_args()

    child_dir = Path(args.child_dir) if args.child_dir else None
    parent_dir = Path(args.parent_dir) if args.parent_dir else None
    require_ok = args.require_ok
    parent_cid = args.parent_cluster
    poll_sec = max(1, args.poll_interval)

    parent_label = parent_dir.name if parent_dir else f"#{parent_cid}"
    print(f"[WAIT_PARENT] Waiting for parent run {parent_label}...", flush=True)

    while True:
        finished, status = get_parent_fate(parent_dir, parent_cid)
        if finished:
            time.sleep(1.5)
            # Re-read to confirm final fate after filesystem settling
            _, final_fate = get_parent_fate(parent_dir, parent_cid)

            # Strict success check: MUST be COMPLETED
            is_ok = (final_fate == "COMPLETED")

            if require_ok and not is_ok:
                print(
                    f"[WAIT_PARENT] Parent run finished with status '{final_fate}' (not OK). "
                    f"Aborting dependent job (--require-ok).",
                    file=sys.stderr,
                    flush=True,
                )
                record_child_cancelled(child_dir, parent_label, final_fate)
                sys.exit(1)

            print(
                f"[WAIT_PARENT] Parent run finished with status '{final_fate}'. "
                f"Proceeding to submit dependent job.",
                flush=True,
            )
            sys.exit(0)

        time.sleep(poll_sec)


if __name__ == "__main__":
    main()
'''


def generate_wait_parent_script() -> str:
    """Returns the standalone Python PRE script placed in `repro/wait_parent.py`."""
    return _WAIT_PARENT_SCRIPT.lstrip()


def generate_dag_file(
    sub_path: Path,
    wait_script_path: Path,
    python_bin: str,
    child_dir: Optional[Path] = None,
    parent_dir: Optional[Path] = None,
    parent_cluster_id: Optional[int] = None,
    parent_log_path: Optional[Path] = None,
    require_ok: bool = False,
    poll_interval: int = 5,
) -> str:
    """Generates `repro/job.dag` for HTCondor DAGMan dependency chaining."""
    lines = [
        "# --- HTCondor DAGMan Workflow Definition generated by hermes-tools ---",
        f"JOB main {sub_path.resolve().as_posix()}",
    ]
    resolved_child_dir = child_dir if child_dir is not None else sub_path.parent.parent
    pre_args: List[str] = [
        wait_script_path.resolve().as_posix(),
        "--child-dir",
        resolved_child_dir.resolve().as_posix(),
        "--poll-interval",
        str(poll_interval),
    ]
    if parent_dir is not None:
        pre_args.extend(["--parent-dir", parent_dir.resolve().as_posix()])
    if parent_cluster_id is not None:
        pre_args.extend(["--parent-cluster", str(parent_cluster_id)])
    if parent_log_path is not None:
        pre_args.extend(["--parent-log", parent_log_path.resolve().as_posix()])
    if require_ok:
        pre_args.append("--require-ok")

    pre_cmd = f"SCRIPT PRE main {python_bin} " + " ".join(
        f'"{a}"' if any(c.isspace() or c in ('"', "'") for c in a) else a
        for a in pre_args
    )
    lines.append(pre_cmd)
    return "\n".join(lines) + "\n"


def extract_child_cluster_id_from_logs(repro_dir: Path) -> Optional[int]:
    """Inspects condor.log, job.dag.dagman.out, and job.dag.nodes.log to discover child job cluster ID."""
    condor_log = repro_dir / "condor.log"
    if condor_log.is_file():
        try:
            txt = condor_log.read_text(encoding="utf-8", errors="replace")
            m = re.search(r"000\s+\((\d+)\.\d+\.\d+\)", txt)
            if m:
                return int(m.group(1))
        except OSError:
            pass

    dag_out = repro_dir / "job.dag.dagman.out"
    if dag_out.is_file():
        try:
            txt = dag_out.read_text(encoding="utf-8", errors="replace")
            m = re.search(r"(?:submitted to cluster|Submitted job\s+\S+\s+\(cluster)\s+(\d+)", txt, re.IGNORECASE)
            if m:
                return int(m.group(1))
        except OSError:
            pass

    nodes_log = repro_dir / "job.dag.nodes.log"
    if nodes_log.is_file():
        try:
            txt = nodes_log.read_text(encoding="utf-8", errors="replace")
            m = re.search(r"000\s+\((\d+)\.\d+\.\d+\)", txt)
            if m:
                return int(m.group(1))
        except OSError:
            pass

    return None


def generate_retry_sub_file(
    orig_sub_path: Optional[Path] = None,
    *,
    run_dir: Path,
    repro_dir: Path,
    script_stem: str,
    is_sweep: bool,
    targets: List[Tuple[int, str]],
    retry_epoch: int,
    run_data: Optional[Dict[str, Any]] = None,
) -> str:
    """Builds `repro/job_retry.sub` directly from structured `run.json` metadata via `generate_sub_file`."""
    data = run_data if run_data is not None else load_run_json(run_dir)
    req = data.get("resources_requested", {})
    sub = data.get("submission", {})
    project_cwd = Path(sub.get("working_directory") or run_dir.parent.parent)
    sweep_args = list(req.get("sweep_arguments") or ([""] * (2 if is_sweep else 1)))
    return generate_sub_file(
        sh_path=repro_dir / "job.sh",
        project_cwd=project_cwd,
        run_dir=run_dir,
        repro_dir=repro_dir,
        script_stem=script_stem,
        arch=str(req.get("architecture", "AMD")),
        cpus=req.get("cpus"),
        mem_mb=req.get("memory_mb"),
        gpus=req.get("gpus"),
        sweep_args=sweep_args,
        nodes_include=list(req.get("node_include") or []),
        nodes_exclude=list(req.get("node_exclude") or []),
        project_code=req.get("project_code"),
        scratch_files=list(req.get("scratch_files") or []),
        retry_targets=targets,
        retry_epoch=retry_epoch,
    )


# ---------------------------------------------------------------------------
# Live HTCondor Queue & Multi-Cluster Reconciliation
# ---------------------------------------------------------------------------
def get_all_cluster_ids(data: Dict[str, Any]) -> List[int]:
    """Returns all HTCondor cluster IDs (initial + retries + dependency DAGMan/job) associated with a run."""
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
    dep = data.get("dependency", {})
    if isinstance(dep, dict):
        for k in ("dagman_cluster_id", "job_cluster_id"):
            raw_dep_cid = dep.get(k)
            if raw_dep_cid is not None:
                try:
                    v = int(raw_dep_cid)
                    if v not in cids:
                        cids.append(v)
                except (ValueError, TypeError):
                    pass
    return cids


def query_live_condor_clusters() -> Optional[Dict[int, Dict[str, Any]]]:
    if not shutil.which("condor_q"):
        return None
    import time as _time

    user = get_username()
    cmd = ["condor_q"]
    if user and user != "unknown":
        cmd.append(user)
    cmd.extend(
        [
            "-nobatch",
            "-json",
            "-attributes",
            "ClusterId,ProcId,JobStatus,HoldReason,RemoteHost,"
            "JobCurrentStartExecutingDate,JobCurrentStartDate,EnteredCurrentStatus,"
            "ServerTime,MemoryUsage,ResidentSetSize,DAGManJobId",
        ]
    )
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
        now_fallback = int(_time.time())
        clusters: Dict[int, Dict[str, Any]] = {}
        dag_parent_map: Dict[int, List[int]] = {}
        for job in items:
            cid = int(job.get("ClusterId", -1))
            st = int(job.get("JobStatus", 0))  # 1=Idle, 2=Running, 5=Held
            raw_dag = job.get("DAGManJobId")
            if raw_dag is not None:
                try:
                    did = int(raw_dag)
                    if did > 0:
                        dag_parent_map.setdefault(did, []).append(cid)
                except (ValueError, TypeError):
                    pass
            info = clusters.setdefault(
                cid,
                {
                    "total": 0,
                    "running": 0,
                    "idle": 0,
                    "held": 0,
                    "hold_reasons": [],
                    "hosts": [],
                    "max_runtime_sec": 0,
                    "max_ram_mb": 0.0,
                    "child_clusters": [],
                },
            )
            info["total"] += 1
            if st == 2:
                info["running"] += 1
                raw_host = str(job.get("RemoteHost") or "").strip()
                if raw_host:
                    host_part = raw_host.split("@")[-1].split(".")[0]
                    if host_part and host_part not in info["hosts"]:
                        info["hosts"].append(host_part)
                start_ts = int(
                    job.get("JobCurrentStartExecutingDate")
                    or job.get("JobCurrentStartDate")
                    or job.get("EnteredCurrentStatus")
                    or 0
                )
                srv_now = int(job.get("ServerTime") or now_fallback)
                if start_ts > 0 and srv_now >= start_ts:
                    info["max_runtime_sec"] = max(int(info["max_runtime_sec"]), srv_now - start_ts)
                mem_mb = job.get("MemoryUsage")
                rss_kb = job.get("ResidentSetSize")
                live_mb = 0.0
                if isinstance(mem_mb, (int, float)) and mem_mb > 0:
                    live_mb = float(mem_mb)
                elif isinstance(rss_kb, (int, float)) and rss_kb > 0:
                    live_mb = float(rss_kb) / 1024.0
                if live_mb > 0:
                    info["max_ram_mb"] = max(float(info["max_ram_mb"]), live_mb)
            elif st == 1:
                info["idle"] += 1
            elif st == 5:
                info["held"] += 1
                hr = str(job.get("HoldReason") or "").strip()
                if hr and hr not in info["hold_reasons"]:
                    info["hold_reasons"].append(hr)
        for did, ch_list in dag_parent_map.items():
            dag_info = clusters.setdefault(
                did,
                {
                    "total": 0,
                    "running": 0,
                    "idle": 0,
                    "held": 0,
                    "hold_reasons": [],
                    "hosts": [],
                    "max_runtime_sec": 0,
                    "max_ram_mb": 0.0,
                    "child_clusters": [],
                },
            )
            existing_ch = dag_info.setdefault("child_clusters", [])
            for c in ch_list:
                if c not in existing_ch:
                    existing_ch.append(c)
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


def get_failed_or_missing_procs(data: Dict[str, Any]) -> List[Tuple[int, str]]:
    """Returns `[(process_id, arg_line), ...]` for any sweep/job process that did not complete with exit_code 0."""
    req = data.get("resources_requested", {})
    sweep_args: List[str] = list(req.get("sweep_arguments") or [""])
    execs = {int(x.get("process_id", 0)): x for x in get_execution_records(data)}
    failed: List[Tuple[int, str]] = []
    for pid, arg_line in enumerate(sweep_args):
        rec = execs.get(pid)
        if rec is None or int(rec.get("exit_code", -1)) != 0 or rec.get("signal"):
            failed.append((pid, arg_line))
    return failed


def run_needs_live_reconcile(data: Dict[str, Any]) -> bool:
    """Returns True if a run is active or has incomplete execution records that may need reconciliation."""
    status = str(data.get("status", "UNKNOWN"))
    if status in ("DRY_RUN", "SUBMIT_FAILED"):
        return False
    if status.startswith(("QUEUED", "RUNNING", "HELD", "WAITING")):
        return True
    sweep_count = int(data.get("resources_requested", {}).get("sweep_count", 1) or 1)
    return len(get_execution_records(data)) < sweep_count and status not in (
        "ABORTED",
        "VANISHED",
        "CANCELLED",
    ) and not status.startswith("CANCELLED")


def reconcile_run(
    run_dir: Path,
    data: Dict[str, Any],
    live_clusters: Optional[Dict[int, Dict[str, Any]]],
) -> Dict[str, Any]:
    """Synchronizes `run.json` with `repro/node_*.json` and all live `condor_q` clusters."""
    status = str(data.get("status", "UNKNOWN"))
    is_active_status = status in ("QUEUED", "RUNNING", "HELD", "WAITING") or status.startswith(
        ("QUEUED", "RUNNING", "HELD", "WAITING")
    )
    sweep_count = int(data.get("resources_requested", {}).get("sweep_count", 1) or 1)

    # Fast path for terminal runs: skip CephFS globbing and condor_q checks completely
    # (unless an HTCondor cluster for this run is still actively running in condor_q)
    has_live_cluster = live_clusters is not None and any(c in live_clusters for c in get_all_cluster_ids(data))
    if not is_active_status and not has_live_cluster and (
        status in ("DRY_RUN", "SUBMIT_FAILED", "ABORTED", "CANCELLED")
        or status.startswith("CANCELLED")
        or len(get_execution_records(data)) >= sweep_count
    ):
        return data

    repro_dir = run_dir / "repro"
    run_json_path = run_dir / "run.json"

    dep = data.get("dependency")
    is_dependent = isinstance(dep, dict)

    # Check if a dependent run was prematurely marked CANCELLED while its DAGMan is actively running in condor_q
    dagman_cid = dep.get("dagman_cluster_id") if is_dependent else None
    if status.startswith("CANCELLED") and is_dependent and dagman_cid and live_clusters and dagman_cid in live_clusters:
        locked_mutate_run_json(run_json_path, repro_dir, lambda d: d.update({"status": "WAITING"}))
        data["status"] = "WAITING"
        status = "WAITING"
        is_active_status = True

    synced_execs: Optional[List[Dict[str, Any]]] = None
    if repro_dir.is_dir():
        node_files = sorted(repro_dir.glob("node_*.json"))
        if node_files:
            loaded_execs: List[Dict[str, Any]] = []
            for nf in node_files:
                try:
                    loaded_execs.append(json.loads(nf.read_text(encoding="utf-8")))
                except Exception:
                    pass
            loaded_execs.sort(key=lambda x: x.get("process_id", 0))
            if loaded_execs != get_execution_records(data):
                synced_execs = loaded_execs
                set_execution_records(data, loaded_execs)

    if live_clusters is None:
        return data

    all_cids = get_all_cluster_ids(data)
    active_cids = [c for c in all_cids if c in live_clusters]

    # --- DAGMan Dependency Workflow Handling ---
    dep = data.get("dependency")
    if isinstance(dep, dict):
        dagman_cid = dep.get("dagman_cluster_id")
        job_cid = dep.get("job_cluster_id")
        parent_cid = dep.get("parent_cluster_id")

        # Discover job_cid from live_clusters DAGMan child_clusters if not yet recorded
        if dagman_cid is not None and dagman_cid in live_clusters:
            ch_list = live_clusters[dagman_cid].get("child_clusters") or []
            if ch_list and (job_cid is None or job_cid not in ch_list):
                job_cid = ch_list[0]

                def _update_job_cid(d: Dict[str, Any]) -> None:
                    d.setdefault("dependency", {})["job_cluster_id"] = job_cid
                    d.setdefault("submission", {})["cluster_id"] = job_cid

                locked_mutate_run_json(run_json_path, repro_dir, _update_job_cid)
                dep["job_cluster_id"] = job_cid
                data.setdefault("submission", {})["cluster_id"] = job_cid
                if job_cid not in all_cids:
                    all_cids.append(job_cid)
                if job_cid in live_clusters and job_cid not in active_cids:
                    active_cids.append(job_cid)

        # Fallback: discover job_cid from disk log files if DAGMan already submitted child
        if job_cid is None:
            extracted = extract_child_cluster_id_from_logs(repro_dir)
            if extracted is not None:
                job_cid = extracted

                def _update_log_job_cid(d: Dict[str, Any]) -> None:
                    d.setdefault("dependency", {})["job_cluster_id"] = extracted
                    d.setdefault("submission", {})["cluster_id"] = extracted

                locked_mutate_run_json(run_json_path, repro_dir, _update_log_job_cid)
                dep["job_cluster_id"] = extracted
                data.setdefault("submission", {})["cluster_id"] = extracted
                if extracted not in all_cids:
                    all_cids.append(extracted)
                if extracted in live_clusters and extracted not in active_cids:
                    active_cids.append(extracted)

        # Check if DAGMan has finished / exited on disk
        dag_out = repro_dir / "job.dag.dagman.out"
        dag_has_exited = False
        if dag_out.is_file():
            try:
                txt = dag_out.read_text(encoding="utf-8", errors="replace")
                if "EXITING WITH STATUS" in txt or "failed with status" in txt:
                    dag_has_exited = True
            except OSError:
                pass

        # Check if parent run is still active
        parent_dir_str = dep.get("parent_run_dir")
        parent_active = False
        if parent_dir_str:
            try:
                p_rj = load_run_json(Path(parent_dir_str))
                p_st = str(p_rj.get("_live_badge") or p_rj.get("status", ""))
                if p_st.startswith(("QUEUED", "RUNNING", "HELD", "WAITING")):
                    parent_active = True
            except Exception:
                pass
        if parent_cid is not None and live_clusters and parent_cid in live_clusters:
            parent_active = True

        # Still waiting for parent if child job not active in queue and executions not yet finished
        if (
            (job_cid is None or job_cid not in live_clusters)
            and len(get_execution_records(data)) < sweep_count
        ):
            is_dag_active = bool(dagman_cid is not None and dagman_cid in live_clusters)
            if is_dag_active or (not dag_has_exited and parent_active):
                parent_tag = f"after #{parent_cid}" if parent_cid else "after parent"
                data["_live_badge"] = f"WAITING ({parent_tag})"
                return data

        # If child job has started in condor_q, transition out of WAITING status
        if job_cid is not None and job_cid in live_clusters:
            if status == "WAITING":
                child_running = int(live_clusters[job_cid].get("running", 0))
                new_st = "RUNNING" if child_running > 0 else "QUEUED"
                locked_mutate_run_json(
                    run_json_path, repro_dir, lambda d: d.update({"status": new_st})
                )
                data["status"] = new_st
                status = new_st
            # Don't attribute DAGMan host/RAM/walltime to the workload telemetry
            active_cids = [c for c in active_cids if c != dagman_cid]

    if active_cids:
        if synced_execs is not None:
            locked_mutate_run_json(
                run_json_path,
                repro_dir,
                lambda d: set_execution_records(d, synced_execs),
            )
        total_running = sum(int(live_clusters[c].get("running", 0)) for c in active_cids)
        total_held = sum(int(live_clusters[c].get("held", 0)) for c in active_cids)
        if total_held > 0:
            data["_live_badge"] = f"HELD ({total_held} held)"
            reasons: List[str] = []
            for c in active_cids:
                for hr in live_clusters[c].get("hold_reasons") or []:
                    if hr and hr not in reasons:
                        reasons.append(str(hr))
            if reasons:
                data["_hold_reasons"] = reasons
        elif total_running > 0:
            if sweep_count > 1:
                done = len(get_execution_records(data))
                data["_live_badge"] = (
                    f"RUNNING ({total_running} active, {done}/{sweep_count} done)"
                )
            else:
                data["_live_badge"] = "RUNNING"
            live_hosts: List[str] = []
            for c in active_cids:
                for h in live_clusters[c].get("hosts") or []:
                    if h and h not in live_hosts:
                        live_hosts.append(str(h))
            if live_hosts:
                data["_live_hosts"] = live_hosts
            max_rt = max(
                (int(live_clusters[c].get("max_runtime_sec") or 0) for c in active_cids),
                default=0,
            )
            if max_rt > 0:
                data["_live_runtime_sec"] = max_rt
            max_ram = max(
                (float(live_clusters[c].get("max_ram_mb") or 0.0) for c in active_cids),
                default=0.0,
            )
            if max_ram > 0:
                data["_live_ram_mb"] = max_ram
        else:
            data["_live_badge"] = "QUEUED"
        return data

    if is_active_status and all_cids:
        fate = inspect_condor_log_fate(repro_dir / "condor.log", is_sweep=(sweep_count > 1))
        execs = get_execution_records(data)
        if len(execs) >= sweep_count:
            any_sig = any(bool(x.get("signal")) for x in execs)
            any_err = any(x.get("exit_code", 0) != 0 for x in execs)
            if any_sig:
                first_sig = next((x.get("signal") for x in execs if x.get("signal")), None)
                new_status = (
                    f"INTERRUPTED ({first_sig})"
                    if sweep_count == 1 and first_sig
                    else "INTERRUPTED"
                )
            elif any_err:
                new_status = "COMPLETED_WITH_ERRORS" if sweep_count > 1 else "FAILED"
            else:
                new_status = "COMPLETED"
        else:
            if fate in ("COMPLETED", "FAILED", "ABORTED"):
                new_status = fate
            elif dep and isinstance(dep, dict) and not get_execution_records(data):
                require_ok = bool(dep.get("require_ok", False))
                parent_dir_str = dep.get("parent_run_dir")
                parent_ok = False
                parent_active = False
                if parent_dir_str:
                    try:
                        p_rj = load_run_json(Path(parent_dir_str))
                        p_st = str(p_rj.get("_live_badge") or p_rj.get("status", ""))
                        if p_st.startswith(("QUEUED", "RUNNING", "HELD", "WAITING")):
                            parent_active = True
                        p_execs = get_execution_records(p_rj)
                        if p_st == "COMPLETED" or (
                            p_execs
                            and all(
                                int(x.get("exit_code", -1)) == 0 and not x.get("signal")
                                for x in p_execs
                            )
                        ):
                            parent_ok = True
                    except Exception:
                        pass
                if parent_active:
                    parent_tag = f"after #{parent_cid}" if parent_cid else "after parent"
                    data["_live_badge"] = f"WAITING ({parent_tag})"
                    return data
                if require_ok and not parent_ok:
                    new_status = "CANCELLED (parent failed)"
                elif fate:
                    new_status = fate
                else:
                    new_status = "CANCELLED"
            elif fate == "FAILED":
                new_status = (
                    f"FAILED ({len(execs)}/{sweep_count} finished)"
                    if sweep_count > 1
                    else "FAILED"
                )
            else:
                new_status = (
                    f"INCOMPLETE ({len(execs)}/{sweep_count})"
                    if sweep_count > 1
                    else (fate or "VANISHED")
                )

        def _heal(d: Dict[str, Any]) -> None:
            if synced_execs is not None:
                set_execution_records(d, synced_execs)
            if str(d.get("status", "")).startswith(("QUEUED", "RUNNING", "HELD", "WAITING")):
                d["status"] = new_status

        data = locked_mutate_run_json(run_json_path, repro_dir, _heal)
    elif synced_execs is not None:
        data = locked_mutate_run_json(
            run_json_path,
            repro_dir,
            lambda d: set_execution_records(d, synced_execs),
        )

    return data


# ---------------------------------------------------------------------------
# Live Cluster Node & Pool Availability (`condor_status`)
# ---------------------------------------------------------------------------
_POOL_ORDER: Dict[str, int] = {
    "GPU3090": 0,
    "A100": 1,
    "EPYC7662": 2,
    "AMD": 3,
    "INTEL": 4,
}


def _infer_node_arch(ad: Dict[str, Any], host: str) -> str:
    raw = str(ad.get("Architecture") or "").strip()
    if raw and raw.upper() not in ("X86_64", "INTEL/LINUX", "UNDEFINED", "NONE"):
        return raw.upper() if raw.upper() in ARCH_SPECS else raw
    h_low = host.lower()
    if "3090" in h_low:
        return "GPU3090"
    if "a100" in h_low:
        return "A100"
    if "7662" in h_low:
        return "EPYC7662"
    return "AMD"


def _count_assigned_gpus(val: Any) -> int:
    if isinstance(val, int):
        return max(0, val)
    if isinstance(val, str):
        s = val.strip()
        if not s or s.lower() == "none":
            return 0
        if s.isdigit():
            return int(s)
        return len([tok for tok in s.split(",") if tok.strip()])
    return 0


def _clean_condor_user(raw_user: Any) -> str:
    s = str(raw_user or "").strip()
    if not s or s.lower() in ("none", "undefined"):
        return ""
    return s.split("@")[0]


def _extract_gpu_model(ad: Dict[str, Any]) -> str:
    raw_name = str(
        ad.get("CUDADeviceName")
        or ad.get("GPUs_DeviceName")
        or ad.get("CUDA0DeviceName")
        or ""
    ).strip()
    if not raw_name or raw_name.lower() in ("none", "undefined"):
        return ""
    short = (
        raw_name.replace("NVIDIA GeForce ", "")
        .replace("NVIDIA ", "")
        .replace("Tesla ", "")
        .strip()
    )
    vram_mb = int(
        ad.get("CUDAGlobalMemoryMb")
        or ad.get("GPUs_GlobalMemoryMb")
        or ad.get("CUDA0GlobalMemoryMb")
        or 0
    )
    if vram_mb > 0:
        vram_gb = round(vram_mb / 1024)
        if f"{vram_gb}GB" not in short.upper():
            return f"{short} ({vram_gb}GB)"
    return short


def parse_condor_status_ads(
    items: List[Dict[str, Any]], now_epoch: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Aggregates raw `condor_status -json` slot ClassAds into per-machine hardware,
    user, and active-claim records, handling both Partitionable+Dynamic and Static slots.
    """
    import time as _time

    ref_now = int(now_epoch if now_epoch is not None else _time.time())
    by_machine: Dict[str, List[Dict[str, Any]]] = {}
    for ad in items:
        if not isinstance(ad, dict):
            continue
        machine = str(ad.get("Machine") or "").strip()
        if not machine:
            name_val = str(ad.get("Name") or "").strip()
            machine = name_val.split("@")[-1] if "@" in name_val else name_val
        if machine:
            by_machine.setdefault(machine, []).append(ad)

    nodes: List[Dict[str, Any]] = []
    for machine, ads in by_machine.items():
        host = machine.split(".")[0]
        arch = "AMD"
        gpu_model = ""
        for ad in ads:
            cand = _infer_node_arch(ad, host)
            if cand != "AMD" or arch == "AMD":
                arch = cand
            if not gpu_model:
                gpu_model = _extract_gpu_model(ad)

        cpus_free = 0
        cpus_used = 0
        cpus_rep_total = 0

        mem_free = 0
        mem_used = 0
        mem_rep_total = 0

        gpus_free = 0
        gpus_used = 0
        gpus_rep_total = 0

        disk_free_kb = 0
        load_avg = 0.0
        sum_slot_load = 0.0

        active_jobs = 0
        draining = False
        users_count: Dict[str, int] = {}
        claims: List[Dict[str, Any]] = []

        for ad in ads:
            state = str(ad.get("State") or "").strip()
            slot_type = str(ad.get("SlotType") or "").strip()
            slot_raw_name = str(ad.get("Name") or "").strip()
            slot_short = slot_raw_name.split("@")[0] if "@" in slot_raw_name else (slot_raw_name or "slot")
            is_draining = bool(ad.get("Draining")) or state == "Drained"
            if is_draining:
                draining = True

            c = int(ad.get("Cpus") or 0)
            m = int(ad.get("Memory") or 0)
            g = int(ad.get("GPUs") or 0)

            tc = int(ad.get("TotalSlotCpus") or ad.get("TotalCpus") or 0)
            tm = int(ad.get("TotalSlotMemory") or ad.get("TotalMemory") or 0)
            tg = int(
                ad.get("TotalSlotGPUs")
                or ad.get("TotalGPUs")
                or _count_assigned_gpus(ad.get("AssignedGPUs"))
                or 0
            )
            if slot_type != "Dynamic":
                cpus_rep_total = max(cpus_rep_total, tc)
                mem_rep_total = max(mem_rep_total, tm)
                gpus_rep_total = max(gpus_rep_total, tg)
            else:
                cpus_rep_total = max(cpus_rep_total, int(ad.get("TotalCpus") or 0))
                mem_rep_total = max(mem_rep_total, int(ad.get("TotalMemory") or 0))
                gpus_rep_total = max(gpus_rep_total, int(ad.get("TotalGPUs") or 0))
            disk_free_kb = max(disk_free_kb, int(ad.get("Disk") or 0))

            tot_load = ad.get("TotalLoadAvg") or ad.get("TotalCondorLoadAvg")
            if tot_load is not None:
                try:
                    load_avg = max(load_avg, float(tot_load))
                except (ValueError, TypeError):
                    pass
            try:
                sum_slot_load += float(ad.get("LoadAvg") or 0.0)
            except (ValueError, TypeError):
                pass

            if state == "Claimed":
                active_jobs += 1
                cpus_used += c
                mem_used += m
                gpus_used += g
                user = _clean_condor_user(ad.get("RemoteUser") or ad.get("RemoteOwner"))
                if user:
                    users_count[user] = users_count.get(user, 0) + 1
                entered = int(ad.get("EnteredCurrentActivity") or 0)
                ad_now = int(ad.get("MyCurrentTime") or ref_now)
                runtime_sec = max(0, ad_now - entered) if entered > 0 else None
                claims.append(
                    {
                        "host": host,
                        "arch": arch,
                        "slot": slot_short,
                        "user": user or "unknown",
                        "job_id": str(ad.get("JobId") or "—"),
                        "gpus": g or (1 if arch == "GPU3090" else 0),
                        "cpus": c,
                        "mem_mb": m,
                        "runtime_seconds": runtime_sec,
                        "activity": str(ad.get("Activity") or "Busy"),
                    }
                )
            elif state == "Unclaimed" and not is_draining:
                cpus_free += c
                mem_free += m
                gpus_free += g

        if load_avg == 0.0 and sum_slot_load > 0.0:
            load_avg = sum_slot_load

        cpus_total = max(cpus_free + cpus_used, cpus_rep_total)
        mem_total = max(mem_free + mem_used, mem_rep_total)
        gpus_total = max(gpus_free + gpus_used, gpus_rep_total)
        if arch == "GPU3090" and gpus_total == 0:
            gpus_total = 1
            if active_jobs == 0 and not draining:
                gpus_free = 1

        # GPU3090 locks the entire node when claimed by a job
        if arch == "GPU3090" and (active_jobs > 0 or gpus_free == 0) and not draining:
            node_state = "BUSY"
            gpus_free = 0
            cpus_free = 0
            mem_free = 0
        elif draining:
            node_state = "DRAINING"
        elif active_jobs == 0 and (cpus_free > 0 or gpus_free > 0):
            node_state = "FREE"
        elif (gpus_total > 0 and gpus_free == 0) or (gpus_total == 0 and cpus_free == 0):
            node_state = "BUSY"
        else:
            node_state = "MIXED"

        nodes.append(
            {
                "machine": machine,
                "host": host,
                "arch": arch,
                "state": node_state,
                "active_jobs": active_jobs,
                "gpus_free": gpus_free,
                "gpus_total": gpus_total,
                "cpus_free": cpus_free,
                "cpus_total": cpus_total,
                "mem_free_mb": mem_free,
                "mem_total_mb": mem_total,
                "disk_free_gb": round(disk_free_kb / (1024 * 1024)) if disk_free_kb > 0 else 0,
                "load_avg": round(load_avg, 1),
                "gpu_model": gpu_model,
                "users": users_count,
                "claims": claims,
            }
        )

    nodes.sort(key=lambda n: (_POOL_ORDER.get(str(n["arch"]), 99), str(n["host"])))
    return nodes


def query_condor_nodes(constraint: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
    """Queries `condor_status -json` and returns per-node hardware, user, and slot availability."""
    if not shutil.which("condor_status"):
        return None
    attrs = (
        "Machine,Name,SlotType,State,Activity,Architecture,Cpus,TotalSlotCpus,TotalCpus,"
        "Memory,TotalSlotMemory,TotalMemory,GPUs,TotalSlotGPUs,TotalGPUs,AssignedGPUs,Draining,"
        "RemoteUser,RemoteOwner,JobId,EnteredCurrentActivity,MyCurrentTime,"
        "TotalLoadAvg,TotalCondorLoadAvg,LoadAvg,Disk,"
        "CUDADeviceName,GPUs_DeviceName,CUDA0DeviceName,"
        "CUDAGlobalMemoryMb,GPUs_GlobalMemoryMb,CUDA0GlobalMemoryMb"
    )
    cmd = ["condor_status", "-json", "-attributes", attrs]
    if constraint and constraint.strip():
        cmd.extend(["-constraint", constraint.strip()])
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
            err_detail = (res.stderr or res.stdout or "").strip()
            if constraint and constraint.strip() and err_detail:
                raise ValueError(err_detail)
            return None
        raw = res.stdout.strip()
        if not raw:
            return []
        items = json.loads(raw)
        if not isinstance(items, list):
            return None
        return parse_condor_status_ads(items)
    except ValueError:
        raise
    except Exception:
        return None


def summarize_condor_pools(nodes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregates per-node availability and active users into per-architecture pool totals."""
    pools: Dict[str, Dict[str, Any]] = {}
    for n in nodes:
        arch = str(n["arch"])
        p = pools.setdefault(
            arch,
            {
                "nodes_total": 0,
                "nodes_avail": 0,
                "gpus_free": 0,
                "gpus_total": 0,
                "cpus_free": 0,
                "cpus_total": 0,
                "mem_free_mb": 0,
                "mem_total_mb": 0,
                "active_jobs": 0,
                "users": {},
            },
        )
        p["nodes_total"] += 1
        if n["state"] in ("FREE", "MIXED"):
            p["nodes_avail"] += 1
        p["gpus_free"] += int(n["gpus_free"])
        p["gpus_total"] += int(n["gpus_total"])
        p["cpus_free"] += int(n["cpus_free"])
        p["cpus_total"] += int(n["cpus_total"])
        p["mem_free_mb"] += int(n["mem_free_mb"])
        p["mem_total_mb"] += int(n["mem_total_mb"])
        p["active_jobs"] += int(n["active_jobs"])
        for u, cnt in (n.get("users") or {}).items():
            p["users"][u] = p["users"].get(u, 0) + int(cnt)
    return pools


# ---------------------------------------------------------------------------
# Queue Match Analysis (`condor_q -better-analyze`)
# ---------------------------------------------------------------------------
def parse_condor_better_analyze(raw_text: str) -> Dict[str, Any]:
    """
    Parses `condor_q -better-analyze` output into structured requirement steps,
    attribute definitions, machine summary counts, and a plain-English verdict.
    """
    text = (raw_text or "").strip()
    job_id = ""
    m_job = re.search(r"Requirements expression for job (\d+\.\d+)", text) or re.search(
        r"^(\d+\.\d+):", text, re.MULTILINE
    )
    if m_job:
        raw_jid = m_job.group(1)
        parts = raw_jid.split(".", 1)
        job_id = f"{parts[0]}.{int(parts[1])}" if len(parts) == 2 and parts[1].isdigit() else raw_jid

    attributes: Dict[str, str] = {}
    in_attrs = False
    for line in text.splitlines():
        s = line.strip()
        if "defines the following attributes:" in s:
            in_attrs = True
            continue
        if in_attrs:
            if not s:
                if attributes:
                    in_attrs = False
                continue
            if s.startswith("The Requirements expression") or s.startswith("Slots"):
                in_attrs = False
                continue
            if "=" in s:
                k, v = s.split("=", 1)
                attributes[k.strip()] = v.strip().strip('"')

    steps: List[Dict[str, Any]] = []
    for m_step in re.finditer(r"^\s*(\[\d+\])\s+(\d+)\s+(.+?)\s*$", text, re.MULTILINE):
        step_id = m_step.group(1)
        slots = int(m_step.group(2))
        cond = m_step.group(3).strip()
        hint = ""
        m_lit = re.search(r'==\s*"([^"]+)"', cond)
        if m_lit:
            hint = m_lit.group(1)
        elif "RequestMemory" in cond and "RequestMemory" in attributes:
            rm_val = attributes["RequestMemory"]
            if rm_val.isdigit():
                mb = int(rm_val)
                hint = f"{mb / 1024:g} GB ({mb} MB)" if mb >= 1024 else f"{mb} MB"
        elif "RequestCpus" in cond and "RequestCpus" in attributes:
            rc_val = attributes["RequestCpus"]
            hint = f"{rc_val} CPUs" if rc_val.isdigit() else rc_val
        elif "RequestGPUs" in cond and "RequestGPUs" in attributes:
            rg_val = attributes["RequestGPUs"]
            hint = f"{rg_val} GPUs" if rg_val.isdigit() else rg_val
        elif "RequestDisk" in cond:
            rd_val = attributes.get("RequestDisk", "")
            if rd_val in attributes:
                rd_val = attributes[rd_val]
            if rd_val:
                hint = f"{rd_val} KB" if rd_val.isdigit() else rd_val
        elif "FileSystemDomain" in cond and "FileSystemDomain" in attributes:
            hint = attributes["FileSystemDomain"]
        else:
            for k, v in attributes.items():
                if k in ("FileSystemDomain", "DiskUsage"):
                    continue
                if re.search(rf"\b{re.escape(k)}\b", cond):
                    hint = v
                    break
        steps.append(
            {
                "step": step_id,
                "slots": slots,
                "condition": cond,
                "hint": hint,
            }
        )

    m_last = re.search(r"Last failed match:\s*(.+)", text)
    last_failed_match = m_last.group(1).strip() if m_last else ""

    m_reason = re.search(r"Reason for last match failure:\s*(.+)", text)
    failure_reason = m_reason.group(1).strip() if m_reason else ""

    def _extract_int(pattern: str) -> int:
        m = re.search(pattern, text)
        return int(m.group(1)) if m else 0

    machines_total = _extract_int(r"Of\s+(\d+)\s+machines")
    rejected_by_job = _extract_int(r"(\d+)\s+are rejected by your job's requirements")
    rejected_by_node = _extract_int(r"(\d+)\s+reject your job because of their own requirements")
    running_your_jobs = _extract_int(r"(\d+)\s+match and are already running your jobs")
    serving_others = _extract_int(r"(\d+)\s+match but are serving other users")
    able_to_run = _extract_int(r"(\d+)\s+are able to run your job")

    zero_steps = [s for s in steps if s["slots"] == 0]
    scheduler_note = ""
    if able_to_run > 0:
        verdict = f"● ELIGIBLE — {able_to_run} of {machines_total} machines match all requirements"
        verdict_style = "bold green"
        if "PREEMPTION_REQUIREMENTS" in failure_reason:
            scheduler_note = f"Waiting for next negotiator cycle (busy slots cannot be preempted: {failure_reason})"
        elif failure_reason:
            scheduler_note = f"Waiting for next negotiator cycle ({failure_reason})"
        else:
            scheduler_note = "Waiting for next HTCondor negotiator dispatch cycle"
    elif zero_steps:
        zero_conds = ", ".join(s["condition"] for s in zero_steps)
        verdict = f"● BLOCKED — 0 slots matched [{zero_conds}]"
        verdict_style = "bold red"
        scheduler_note = "Requested resources exceed available cluster hardware"
    elif serving_others > 0 or running_your_jobs > 0:
        verdict = (
            f"● WAITING — All {serving_others + running_your_jobs} compatible machines are currently busy"
        )
        verdict_style = "bold yellow"
        scheduler_note = f"{serving_others} serving other users · {running_your_jobs} running your jobs"
    elif machines_total > 0:
        verdict = f"● WAITING — 0 of {machines_total} machines immediately available"
        verdict_style = "bold yellow"
        scheduler_note = f"{rejected_by_job} lacked requested resources · {rejected_by_node} rejected by node/pool policy"
    elif failure_reason:
        verdict = f"● WAITING — {failure_reason}"
        verdict_style = "bold yellow"
    else:
        verdict = "○ No structured match analysis available (job may already be running or finished)"
        verdict_style = "dim"

    return {
        "job_id": job_id,
        "attributes": attributes,
        "steps": steps,
        "last_failed_match": last_failed_match,
        "failure_reason": failure_reason,
        "machines_total": machines_total,
        "summary": {
            "rejected_by_job": rejected_by_job,
            "rejected_by_node": rejected_by_node,
            "running_your_jobs": running_your_jobs,
            "serving_others": serving_others,
            "able_to_run": able_to_run,
        },
        "verdict": verdict,
        "verdict_style": verdict_style,
        "scheduler_note": scheduler_note,
        "raw": text,
    }


def query_condor_better_analyze(job_target: str) -> Optional[Dict[str, Any]]:
    """Runs `condor_q -better-analyze <job_target>` and returns parsed diagnosis."""
    if not shutil.which("condor_q"):
        return None
    try:
        res = subprocess.run(
            ["condor_q", "-better-analyze", str(job_target)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=12,
        )
        combined = (res.stdout or "").strip()
        if not combined:
            combined = (res.stderr or "").strip()
        if not combined:
            return None
        return parse_condor_better_analyze(combined)
    except Exception:
        return None


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
        # Supports both 8-col (pid ppid stat pcpu rss etime comm args)
        # and legacy 6-col (pid pcpu rss etime comm args)
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
        # Fallback to kernel `/proc/<pid>/stat` ticks when lxcfs/container uptime zeroes out `ps`
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

    # Group multiprocessing / DataLoader workers under their root command / parent PID
    by_pid: Dict[int, Dict[str, Any]] = {p["pid"]: p for p in raw_procs}

    def _find_root_key(p: Dict[str, Any]) -> str:
        cmd = str(p["command"])
        # If this is a Python multiprocessing helper child, attach to its parent PID if present
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
        # Root representative is the longest-running (oldest) process, tie-broken by lowest PID
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

    # Sort processes: GPU VRAM users first, then highest CPU%, then highest RSS
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
