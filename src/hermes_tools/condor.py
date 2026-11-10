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
    if yes:
        cmd.append("-y")
    return cmd


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


def query_live_condor_clusters() -> Optional[Dict[int, Dict[str, Any]]]:
    if not shutil.which("condor_q"):
        return None
    user = get_username()
    cmd = ["condor_q"]
    if user and user != "unknown":
        cmd.append(user)
    cmd.extend(["-json", "-attributes", "ClusterId,ProcId,JobStatus,HoldReason"])
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
        clusters: Dict[int, Dict[str, Any]] = {}
        for job in items:
            cid = int(job.get("ClusterId", -1))
            st = int(job.get("JobStatus", 0))  # 1=Idle, 2=Running, 5=Held
            info = clusters.setdefault(
                cid,
                {"total": 0, "running": 0, "idle": 0, "held": 0, "hold_reasons": []},
            )
            info["total"] += 1
            if st == 2:
                info["running"] += 1
            elif st == 1:
                info["idle"] += 1
            elif st == 5:
                info["held"] += 1
                hr = str(job.get("HoldReason") or "").strip()
                if hr and hr not in info["hold_reasons"]:
                    info["hold_reasons"].append(hr)
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
    if status.startswith(("QUEUED", "RUNNING", "HELD")):
        return True
    sweep_count = int(data.get("resources_requested", {}).get("sweep_count", 1) or 1)
    return len(get_execution_records(data)) < sweep_count and status not in (
        "ABORTED",
        "VANISHED",
    )


def reconcile_run(
    run_dir: Path,
    data: Dict[str, Any],
    live_clusters: Optional[Dict[int, Dict[str, Any]]],
) -> Dict[str, Any]:
    """Synchronizes `run.json` with `repro/node_*.json` and all live `condor_q` clusters."""
    status = str(data.get("status", "UNKNOWN"))
    is_active_status = status in ("QUEUED", "RUNNING", "HELD") or status.startswith(
        ("QUEUED", "RUNNING", "HELD")
    )
    sweep_count = int(data.get("resources_requested", {}).get("sweep_count", 1) or 1)

    # Fast path for terminal runs: skip CephFS globbing and condor_q checks completely
    if not is_active_status and (
        status in ("DRY_RUN", "SUBMIT_FAILED", "ABORTED")
        or len(get_execution_records(data)) >= sweep_count
    ):
        return data

    repro_dir = run_dir / "repro"
    run_json_path = run_dir / "run.json"

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
            if fate == "ABORTED":
                new_status = "ABORTED"
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
            if str(d.get("status", "")).startswith(("QUEUED", "RUNNING", "HELD")):
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


def parse_condor_status_ads(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggregates raw `condor_status -json` slot ClassAds into per-machine hardware
    availability records, handling both Partitionable+Dynamic slots and Static slots.
    """
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
        for ad in ads:
            cand = _infer_node_arch(ad, host)
            if cand != "AMD" or arch == "AMD":
                arch = cand

        cpus_free = 0
        cpus_used = 0
        cpus_rep_total = 0

        mem_free = 0
        mem_used = 0
        mem_rep_total = 0

        gpus_free = 0
        gpus_used = 0
        gpus_rep_total = 0

        active_jobs = 0
        draining = False

        for ad in ads:
            state = str(ad.get("State") or "").strip()
            slot_type = str(ad.get("SlotType") or "").strip()
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

            if state == "Claimed":
                active_jobs += 1
                cpus_used += c
                mem_used += m
                gpus_used += g
            elif state == "Unclaimed" and not is_draining:
                cpus_free += c
                mem_free += m
                gpus_free += g

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
            }
        )

    nodes.sort(key=lambda n: (_POOL_ORDER.get(str(n["arch"]), 99), str(n["host"])))
    return nodes


def query_condor_nodes() -> Optional[List[Dict[str, Any]]]:
    """Queries `condor_status -json` and returns per-node hardware availability."""
    if not shutil.which("condor_status"):
        return None
    attrs = (
        "Machine,Name,SlotType,State,Activity,Architecture,Cpus,TotalSlotCpus,TotalCpus,"
        "Memory,TotalSlotMemory,TotalMemory,GPUs,TotalSlotGPUs,TotalGPUs,AssignedGPUs,Draining"
    )
    try:
        res = subprocess.run(
            ["condor_status", "-json", "-attributes", attrs],
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
            return []
        items = json.loads(raw)
        if not isinstance(items, list):
            return None
        return parse_condor_status_ads(items)
    except Exception:
        return None


def summarize_condor_pools(nodes: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """Aggregates per-node availability into per-architecture pool totals."""
    pools: Dict[str, Dict[str, int]] = {}
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
    return pools
