# hermes-tools

Definitive HTCondor job launcher (`send`) and run provenance / telemetry manager (`runs`) for the **Hermes 2 (I3A)** cluster.

## Features

- **Interactive & Fast CLI (`send`)**: Launch single jobs or parallel sweeps across `GPU3090`, `A100`, `AMD`, `EPYC7662`, `INTEL`, or `UNDEFINED` pools with automatic hardware rule enforcement.
- **Zero-Breakage Code Snapshots**: Copies tracked/untracked source files (`<= 15 MB` per file, `<= 100 MB` total) into `runs/<run_id>/repro/code/`, records `git` commit + `uncommitted.patch`, and freezes `repro/env.txt` so editing your workspace while a job sits in `condor_q` never corrupts queued jobs.
- **Locale- & NFS-Safe Telemetry (`job.sh`)**: Captures CPU/GPU hardware profiles, queue wait time, wall-clock duration, peak RSS (`MaxRSS`), CPU utilization, page faults, and preemption signals (`SIGTERM`/`SIGINT`/`SIGHUP`) into `run.json` using NFS-safe `flock` + atomic file replacement.
- **Live Run Management & Recovery (`runs`)**: Reconcile `run.json` against live `condor_q`, watch active sweeps (`runs watch`), inspect per-process telemetry and stderr tails (`runs inspect`), hot-sync code into a queued run (`runs sync`), and resume only failed sweep processes in-place (`runs retry`).
- **Project Config Defaults**: Define `[tool.hermes]` in `pyproject.toml` or `.hermes.toml` so common flags (`arch`, `project`, `cpus`, `mem`, `gpus`, `exclude`) don't need to be retyped.

---

## Installation

```bash
uv tool install .
# or in development mode:
uv sync
```

---

## Quick Start

### 1. Submitting Jobs (`send`)

```bash
# Interactive TUI wizard
send

# Direct RTX 3090 submission with script arguments after `--`
send train.py -a GPU3090 -- --epochs 100 --lr 1e-4

# Dual NVIDIA A100 (40GB) with 120 GB system RAM
send train.py -a A100 -g 2 -m 120G

# 16-core AMD EPYC job with 32 GB RAM and run tag
send sim.py -a AMD -c 16 -m 32G -t baseline

# 10-job parallel sweep with zero-padded seeds
send train.py -a GPU3090 -s '--seed {00..09} --lr {1e-3,1e-4}'

# Generate run capsule locally without submitting to HTCondor
send train.py -a GPU3090 --dry-run
```

### 2. Project Defaults (`pyproject.toml` or `.hermes.toml`)

Add `[tool.hermes]` to your project's `pyproject.toml` (or create `.hermes.toml`):

```toml
[tool.hermes]
arch = "GPU3090"
project = "100-100"
exclude = ["her07-100"]
```

### 3. Managing & Recovering Runs (`runs`)

```bash
runs ls                  # Table of runs reconciled with live condor_q
runs watch               # Live-updating dashboard of active runs
runs inspect latest      # Full telemetry breakdown + stderr tracebacks
runs logs latest -e -f   # Follow stderr (auto-selects failed process in sweeps)
runs sync latest         # Hot-patch repro/code/ snapshot while job is still queued
runs retry latest        # Re-run only failed/incomplete sweep processes in-place
runs diff 1 2            # Compare provenance, metrics, and code snapshot diff
runs stop latest         # Kill active HTCondor cluster(s) via condor_rm
runs clean --failed      # Prune failed/aborted/dry-run capsules
```

### 4. Environment Variables Inside Jobs

Your script automatically receives:
- `HERMES_RUN_DIR`: Absolute path to `runs/<run_id>/` for saving checkpoints and artifacts.
- `HERMES_PROC_ID`: HTCondor process index (`0` for single jobs, `0..N-1` for sweeps).
