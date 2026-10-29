# hermes-tools

Definitive HTCondor job submission (`send`), code provenance, live telemetry, and run recovery toolkit (`runs`) for the **Hermes 2 (I3A)** cluster.

---

## Table of Contents

1. [Installation & Updating](#1-installation--updating)
2. [`send` — Job Submission Reference](#2-send--job-submission-reference)
   - [Interactive vs. Direct CLI Mode](#21-interactive-vs-direct-cli-mode)
   - [All `send` Options & Defaults](#22-all-send-options--defaults)
   - [Hermes 2 Architecture Pools & Hardware Rules](#23-hermes-2-architecture-pools--hardware-rules)
   - [Parallel Sweeps & Array Jobs](#24-parallel-sweeps--array-jobs)
   - [Passing Script Arguments Safely (`--` vs `--args` vs `-s`)](#25-passing-script-arguments-safely----vs---args-vs--s)
   - [Node Filtering, Project Accounting & Scratch Mode](#26-node-filtering-project-accounting--scratch-mode)
3. [Project Configuration (`.hermes.toml` / `pyproject.toml`)](#3-project-configuration-hermestoml--pyprojecttoml)
4. [Discovery & Fallback Chains](#4-discovery--fallback-chains)
   - [Virtual Environment & Conda Discovery](#41-virtual-environment--conda-discovery)
   - [Worker Node Python 3 Resolution Chain](#42-worker-node-python-3-resolution-chain)
   - [Run Target Query Resolution (`runs <query>`)](#43-run-target-query-resolution-runs-query)
   - [`runs/` Directory Discovery](#44-runs-directory-discovery)
5. [Run Capsule Layout & Environment Variables](#5-run-capsule-layout--environment-variables)
   - [Directory Structure](#51-directory-structure)
   - [Environment Variables Exported to Your Job](#52-environment-variables-exported-to-your-job)
   - [Code Snapshot Rules & Safety Caps](#53-code-snapshot-rules--safety-caps)
6. [`runs` — Run Management & Recovery Reference](#6-runs--run-management--recovery-reference)
   - [`runs ls` & `runs watch`](#61-runs-ls--runs-watch)
   - [`runs inspect` (`info`, `show`)](#62-runs-inspect-info-show)
   - [`runs logs` (`tail`)](#63-runs-logs-tail)
   - [`runs sync`](#64-runs-sync)
   - [`runs retry` (`resume`, `continue`)](#65-runs-retry-resume-continue)
   - [`runs diff`](#66-runs-diff)
   - [`runs stop` (`kill`, `rm`)](#67-runs-stop-kill-rm)
   - [`runs clean` (`prune`)](#68-runs-clean-prune)
7. [Under the Hood: Telemetry, Locking & Status Lifecycle](#7-under-the-hood-telemetry-locking--status-lifecycle)

---

## 1. Installation & Updating

On `hermes2` (or locally):

```bash
# Install globally as CLI tools (`send` and `runs`)
uv tool install .

# Update after pulling new changes
git pull
uv tool install --force .
```

To run the test suite during development:

```bash
uv run pytest -v
```

---

## 2. `send` — Job Submission Reference

### 2.1 Interactive vs. Direct CLI Mode

- **Interactive Wizard**: If you omit the target script (`send`) or omit the target architecture (`send train.py` without `-a` or a project config default), `send` launches an interactive TUI wizard that guides you through selecting the executable, virtual environment, architecture pool, GPUs/CPUs/RAM, sweep arguments, and run tag.
- **Direct CLI**: When both the target script and architecture are provided (via CLI flags or project config), `send` skips the wizard, shows the submission summary panel (with the reproducible one-liner CLI command), and submits upon confirmation (or immediately with `-y`).

```bash
send                                                # Full interactive TUI wizard
send train.py -a GPU3090 -- --epochs 100 --lr 1e-4  # Direct RTX 3090 submission
send train.py -a A100 -g 2 -m 120G -y               # Dual A100, 120 GB RAM, no prompt
send sim.py -a AMD -c 16 -m 32G -t exp1             # 16-core AMD CPU job with tag
send train.py -a GPU3090 -s '--fold {0..4}'         # 5-job parallel sweep
send train.py -a GPU3090 --dry-run                  # Generate capsule without submitting
```

### 2.2 All `send` Options & Defaults

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `target` | `path` | Interactive prompt | Path to Python script (`.py`), Bash script (`.sh`), or executable binary. |
| `-a`, `--arch` | `choice` | Interactive / Config | Target Hermes 2 `+Architecture` pool (`GPU3090`, `A100`, `AMD`, `EPYC7662`, `INTEL`, `UNDEFINED`). |
| `-c`, `--cpus` | `int` | `4` (CPU pools) | Number of CPU cores (`request_cpus`). Automatically omitted on `GPU3090` and `A100`. |
| `-m`, `--mem` | `str` | `8G` (CPU pools) | System RAM request (`8G`, `32GB`, `4096M`, `1T`). Omitted on `GPU3090` and on `A100` when `<= 64G`. |
| `-g`, `--gpus` | `1..4` | `1` (on `A100`) | Number of GPUs (`request_gpus`). Used on `A100` (`1`–`4`); automatic (`1`) on `GPU3090`; ignored on CPU pools. |
| `--args` | `str` | `""` | Base arguments passed to your script (kept literal; not split by sweep braces). |
| `-- <args...>` | `tokens` | `""` | Passthrough arguments after `--` (e.g. `send train.py -a GPU3090 -- --lr 1e-4`). |
| `-s`, `--sweep` | `str` (repeatable) | `[]` | Parallel sweep specification (supports `{start..end}`, `{00..09}`, `{a,b}`, `;`, or multiple `-s`). |
| `--sweep-file` | `path` | `None` | Text file with one argument string per line (`#` comments and blank lines ignored). |
| `-t`, `--tag` | `str` | `""` | Optional suffix sanitized and appended to the run folder: `<timestamp>_<script>_<arch>_<tag>`. |
| `--venv` | `path` | Auto-discovered | Explicit path to a Python virtual environment or Conda environment. |
| `--no-venv` | `flag` | `False` | Do not activate a virtual environment; run with the submitter's default Python 3. |
| `--node` | `str` (repeatable) | `[]` | Restrict execution to specific worker node(s), e.g. `--node nv3090-01` (auto-appends `.i3a.es`). |
| `--exclude` | `str` (repeatable) | `[]` | Exclude specific worker node(s), e.g. `--exclude her07-100`. |
| `--project` | `str` | `None` | Inject I3A institutional accounting directive `+Proyecto = "<code>"`. |
| `--scratch` | `files...` | `[]` | Enable HTCondor file transfer (`should_transfer_files = YES`) and copy listed input files to worker scratch space. |
| `-y`, `--yes` | `flag` | `False` | Skip the final `"Submit job to HTCondor queue now?"` confirmation prompt. |
| `--dry-run` | `flag` | `False` | Materialize `runs/<run_id>/` (`job.sh`, `job.sub`, `run.json`, `repro/code/`) and print `job.sub` without calling `condor_submit`. |

### 2.3 Hermes 2 Architecture Pools & Hardware Rules

Hermes 2 scheduler policies require specific resource directives to be omitted on certain GPU pools so jobs do not get stuck idle in `condor_q`. `send` enforces these rules automatically:

| `+Architecture` | Hardware | `request_cpus` | `request_memory` | `request_gpus` |
| :--- | :--- | :--- | :--- | :--- |
| **`GPU3090`** | NVIDIA RTX 3090 (24 GB) | **Omitted** (Full-node lock) | **Omitted** (Full-node lock) | **Omitted** (Implicit full node) |
| **`A100`** | NVIDIA A100 (40 GB) | **Omitted** (Auto-managed) | **Omitted** if `<= 64 GB`; emitted only if `> 64 GB` | Emitted (`1`–`4`, default `1`) |
| **`AMD`** | AMD EPYC Pool | Emitted (default `4`) | Emitted (default `8192` MB) | **Omitted** |
| **`EPYC7662`** | AMD EPYC 7662 | Emitted (default `4`) | Emitted (default `8192` MB) | **Omitted** |
| **`INTEL`** | Intel Xeon 6226R | Emitted (default `4`) | Emitted (default `8192` MB) | **Omitted** |
| **`UNDEFINED`** | Any available node | Emitted (default `4`) | Emitted (default `8192` MB) | **Omitted** |

### 2.4 Parallel Sweeps & Array Jobs

Use `-s` / `--sweep` or `--sweep-file` to queue a single HTCondor cluster with multiple parallel processes (`queue job_args from (...)`):

- **Integer ranges**: `'{0..4}'` expands to `0, 1, 2, 3, 4`.
- **Stepped ranges**: `'{0..10..2}'` expands to `0, 2, 4, 6, 8, 10` (descending `'{10..0..2}'` also works).
- **Zero-padded ranges**: `'{00..04}'` preserves leading zeros: `00, 01, 02, 03, 04`.
- **Comma lists**: `'{adam,sgd,adamw}'` expands to `adam`, `sgd`, `adamw`.
- **Cartesian products**: Multiple braces in the same string expand to all combinations:
  ```bash
  send train.py -a GPU3090 -s '--seed {00..02} --lr {1e-3,1e-4}'
  # Queues 3 x 2 = 6 parallel processes (Proc 0..5)
  ```
- **Semicolon-separated explicit runs**:
  ```bash
  send train.py -a GPU3090 -s '--mode fast --epochs 10; --mode accurate --epochs 100'
  ```
- **Combining base arguments (`--` / `--args`) with sweeps (`-s`)**:
  Base arguments are prepended to every sweep item:
  ```bash
  send train.py -a GPU3090 -s '--fold {0..4}' -- --data /mnt/cephfs/dataset
  # Proc i receives: --data /mnt/cephfs/dataset --fold <i>
  ```

### 2.5 Passing Script Arguments Safely (`--` vs `--args` vs `-s`)

- **Literal arguments (`--` and `--args`)**: Never undergo brace or semicolon splitting. Passing JSON or Python dicts such as:
  ```bash
  send train.py -a GPU3090 -- --config '{"lr": 1e-4, "betas": [0.9, 0.999]}'
  ```
  keeps the JSON argument intact as a single run.
- **Sweep arguments (`-s` and interactive prompt)**: Undergo brace/semicolon expansion, while still ignoring JSON-like `{key: val}` blocks containing colons (`:`).

### 2.6 Node Filtering, Project Accounting & Scratch Mode

- **Include / Exclude Nodes**:
  ```bash
  send train.py -a GPU3090 --node nv3090-01 --node nv3090-02
  send train.py -a AMD --exclude her07-100 --exclude her07-101
  ```
  Generates HTCondor `requirements = ((TARGET.Machine == "nv3090-01.i3a.es") || ...)` expressions automatically.
- **Project Accounting**: `--project 100-100` adds `+Proyecto = "100-100"` to `job.sub`.
- **Scratch Transfer Mode**: `--scratch data.tar weights.pt` enables `should_transfer_files = YES`, `when_to_transfer_output = ON_EXIT`, and `transfer_input_files = data.tar, weights.pt`.

---

## 3. Project Configuration (`.hermes.toml` / `pyproject.toml`)

You can configure per-project defaults so you don't have to repeat flags on every submission. `hermes-tools` walks up to 4 parent directories (stopping at the Git root) looking for:
1. `.hermes.toml`
2. `pyproject.toml` under `[tool.hermes]`

Example `pyproject.toml`:

```toml
[tool.hermes]
arch = "GPU3090"
project = "100-100"
cpus = 8
mem = "32G"
gpus = 1
tag = "baseline"
venv = ".venv"
no_venv = false
node = []
exclude = ["her07-100"]
scratch = []
```

**Precedence Order (Highest to Lowest):**
1. Explicit CLI flags (`-a`, `-c`, `-m`, `-g`, `--project`, `--venv`, `--no-venv`, `--node`, `--exclude`, `-t`, `--scratch`)
2. `.hermes.toml` in current or parent directory
3. `[tool.hermes]` in `pyproject.toml`
4. Built-in defaults / interactive prompt

---

## 4. Discovery & Fallback Chains

### 4.1 Virtual Environment & Conda Discovery

Unless `--no-venv` is passed, `send` searches for a Python environment in this exact order (always ignoring `hermes-tools`'s own tool venv):

1. **Explicit `--venv <path>`** (or `venv` in `.hermes.toml` / `pyproject.toml`). Exits with an error if invalid.
2. **Active `$VIRTUAL_ENV`** environment variable in your current shell.
3. **Local project directories**: Searches `.venv`, `venv`, and `env` in the current working directory and up to 4 parent directories (stopping at the `.git` root).
4. **Active `$CONDA_PREFIX`** environment variable (Conda / Mamba / Micromamba environments).
5. **Interactive fallback**: If none of the above are found and `send` is running interactively, prompts for a custom venv path (or Enter to use system/login Python 3).

### 4.2 Worker Node Python 3 Resolution Chain

On Hermes 2 worker nodes (such as `her09-*`), HTCondor starts jobs with a minimal `PATH=/usr/bin:/bin` where `/usr/bin/python` is legacy **Python 2.7**. To guarantee your script always runs under the intended Python 3 interpreter:

1. **If a Virtual Env (`bin/activate`) is configured**: Sources `${VENV_PATH}/bin/activate` and uses `python3` / `python` from the activated environment.
2. **If a Conda Env (`bin/python` without `bin/activate`) is configured**: Prepends `${VENV_PATH}/bin` to `PATH` and uses `${VENV_PATH}/bin/python`.
3. **If running without a venv (`--no-venv` or no local venv)**:
   - At submission time on `hermes2`, `send` resolves your login shell's `python3` / `python` binary (verifying `sys.version_info[0] == 3`) and embeds its path as `SUBMIT_PYTHON` in `job.sh`.
   - On the worker node, `job.sh` checks in order:
     1. `${SUBMIT_PYTHON}` (your login shell's exact Python 3 on CephFS/NFS)
     2. `command -v python3` on the worker node
     3. `${UPDATER_PYTHON}` (the Python 3 interpreter that runs `hermes-tools`)
     4. `command -v python` (last resort)
4. **Telemetry Updater Python (`UPDATER_PYTHON`)**: At the end of `job.sh`, `run.json` is updated using `${UPDATER_PYTHON}` if mounted on the worker node, falling back to `python3` -> `python` (the updater snippet uses only the Python standard library).

### 4.3 Run Target Query Resolution (`runs <query>`)

Every `runs` subcommand that accepts a `<run>` target (`inspect`, `logs`, `sync`, `retry`, `stop`, `diff`) resolves your query in this order:

1. **`latest`**: Follows the `runs/latest` symlink (or falls back to the newest run folder).
2. **Numeric index (`1`, `2`, ...)**: Matches the row number `#` shown in `runs ls` (`1` = newest run).
3. **Exact Run ID**: Matches `runs/<query>` exactly.
4. **Substring or HTCondor Cluster ID**: Case-insensitive substring match against folder names (e.g. `runs inspect smoke` or `runs inspect A100`) **or** exact match against any HTCondor `cluster_id` (including `retry_clusters`, e.g. `runs inspect 1071044`).
   - If a single run matches, it is selected immediately.
   - If multiple runs match (or no `<query>` was given), an interactive selector menu is shown (or the newest matching run is chosen when `stdin` is not a TTY).

### 4.4 `runs/` Directory Discovery

When you run `runs` from a subdirectory of your project, [`get_runs_root()`](file:///C:/Users/diego/Desktop/Programming/Python/hermes-tools/src/hermes_tools/core.py#L168-L176) walks up to 4 parent directories (stopping at `.git`) to find the existing `runs/` folder.

---

## 5. Run Capsule Layout & Environment Variables

### 5.1 Directory Structure

Every submission creates a self-contained directory under `runs/` and atomically updates the `runs/latest` symlink:

```text
runs/
├── latest -> 2026-10-29_15-34-31_train_GPU3090_exp1/
└── 2026-10-29_15-34-31_train_GPU3090_exp1/
    ├── run.json                  # Unified provenance, hardware, & per-process telemetry
    ├── train.out / train_0.out   # Job stdout (includes Hermes 2 hardware/telemetry banner)
    ├── train.err / train_0.err   # Job stderr
    └── repro/
        ├── job.sh                # Generated Bash wrapper & signal/resource auditor
        ├── job.sub               # Generated HTCondor submit file
        ├── job_retry.sub         # Generated on `runs retry --in-place`
        ├── condor.log            # HTCondor scheduler event log
        ├── env.txt               # Frozen `pip freeze` / `uv pip freeze` package list
        ├── uncommitted.patch     # `git diff HEAD` at submission time (if repo was dirty)
        ├── node_0.json           # Per-process telemetry record (for parallel sweeps)
        └── code/                 # Isolated snapshot of project source code
```

### 5.2 Environment Variables Exported to Your Job

Inside `job.sh`, the following variables are automatically exported before your workload starts:

| Variable | Value | Purpose |
| :--- | :--- | :--- |
| **`HERMES_RUN_DIR`** | `/abs/path/to/runs/<run_id>` | Save checkpoints, plots, and CSVs directly inside the run folder (`os.environ["HERMES_RUN_DIR"]`). |
| **`HERMES_PROC_ID`** | `0` (or `0..N-1` in sweeps) | Process index within the sweep (`$(Process)` or original process ID on retry). |
| **`PYTHONPATH`** | `.../repro/code:<project_cwd>:$PYTHONPATH` | Ensures Python imports resolve against the frozen `repro/code/` snapshot first. |
| **`OMP_NUM_THREADS`** | `<requested_cpus>` | Caps OpenMP threads on CPU pools to prevent thread oversubscription (respects pre-existing env var). Also sets `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `NUMEXPR_NUM_THREADS`. |

### 5.3 Code Snapshot Rules & Safety Caps

Why does `send` snapshot your code into `repro/code/`?
When HTCondor queues are busy, your job may wait minutes or hours before starting. Because `job.sh` executes the copy inside `repro/code/` (and prepends `repro/code` to `PYTHONPATH`), you can freely edit your working tree or switch Git branches while jobs are queued without breaking them.

To prevent accidental slowdowns or disk quota exhaustion from untracked datasets:
- **Tracked files first**: Git-tracked files (`git ls-files --cached`) are copied before untracked files (`git ls-files --others --exclude-standard`).
- **Per-file limit**: Regular files `<= 15 MB` are copied; symlinks are recreated without dereferencing.
- **Total snapshot limit**: Capped at **`100 MB` total** or **`2500 files`** (prints a note if extra untracked files were skipped).
- **Ignored directories**: `.git`, `runs`, `.venv`, `venv`, `env`, `__pycache__`, `.mypy_cache`, `.pytest_cache`, `.ruff_cache`, `node_modules`, and `wandb` are always excluded.

---

## 6. `runs` — Run Management & Recovery Reference

Running `runs` with no arguments defaults to `runs ls`.

### 6.1 `runs ls` & `runs watch`

Lists runs reconciled against live `condor_q` clusters and `repro/condor.log`:

```bash
runs ls                        # Show latest 25 runs (★ marks `runs/latest`)
runs ls -n 50                  # Show latest 50 runs
runs ls -s RUNNING             # Filter by status substring (RUNNING, FAILED, QUEUED, etc.)
runs ls -t baseline            # Filter by run ID / tag substring
runs watch                     # Live-updating dashboard (refreshes every 2s)
runs watch -i 5 -s RUNNING     # Refresh every 5s, showing only running jobs
```

### 6.2 `runs inspect` (`info`, `show`)

Displays full provenance metadata, per-process hardware/runtime/RAM/CPU breakdown, and the last 25 lines of any non-empty `.err` files:

```bash
runs inspect latest            # Inspect most recent run
runs inspect 1                 # Inspect run #1 from `runs ls`
runs inspect 1071044           # Inspect run by HTCondor cluster ID
```

### 6.3 `runs logs` (`tail`)

Views or follows `.out`, `.err`, or `repro/condor.log`:

```bash
runs logs latest               # Last 60 lines of stdout (.out)
runs logs latest -e            # Last 60 lines of stderr (.err)
runs logs latest -e -p 3       # Stderr of process #3 in a sweep
runs logs latest -c            # HTCondor scheduler log (repro/condor.log)
runs logs latest -f -n 100     # Live `tail -f` starting with last 100 lines
```
> **Smart Sweep Selection**: When you run `runs logs <run> -e` on a sweep without specifying `-p <proc>`, `runs` automatically selects the first process with a non-zero exit code (or the first non-empty `.err` file).

### 6.4 `runs sync`

Hot-patches a queued run's `repro/code/` snapshot, `repro/uncommitted.patch`, `repro/env.txt`, and `run.json["git"]` from your current working tree (removing any deleted files from `repro/code/` and recording `initial_commit` + `synced_at`):

```bash
runs sync latest
```

### 6.5 `runs retry` (`resume`, `continue`)

Recovers failed, interrupted, or incomplete runs/sweeps. Automatically detects which processes did not finish with `exit_code == 0`:

```bash
runs retry latest              # Interactive recovery mode selector
runs retry latest --in-place -u -y  # Update code snapshot & retry failed processes in-place
runs retry latest --in-place -y     # Retry failed processes with frozen code snapshot
runs retry latest --new-run -y      # Relaunch failed processes as a brand-new run via `send`
runs retry latest --all             # Force re-run of all processes (even exit_code 0)
runs retry latest --force           # Allow retry even if cluster still appears in condor_q
```

**How In-Place Retry Works (`--in-place`)**:
1. Guards against colliding with still-running jobs in `condor_q` (unless `--force` is passed).
2. Optionally refreshes `repro/code/` if `-u` / `--update-code` is chosen.
3. Archives previous `.out`/`.err` logs for retried processes into `repro/<name>.prev_<HHMMSS>` and removes stale `repro/node_<pid>.json`.
4. Generates `repro/job_retry.sub` preserving all hardware, architecture, node requirement, project, and scratch directives while mapping HTCondor's `$(orig_proc)` back to the original sweep process IDs.
5. Appends the previous `cluster_id` to `submission.retry_clusters` so `runs` tracks all clusters belonging to the run.

### 6.6 `runs diff`

Compares two runs side-by-side (status, target, architecture, arguments, git commit, peak RAM, wall time) and displays a syntax-highlighted unified diff between their `repro/code/` snapshots:

```bash
runs diff 1 2
```

### 6.7 `runs stop` (`kill`, `rm`)

Calls `condor_rm` on all active HTCondor cluster IDs associated with the run and marks `run.json` as `ABORTED`:

```bash
runs stop latest
runs stop 1071044 -y
```

### 6.8 `runs clean` (`prune`)

Deletes inactive run directories and repairs the `runs/latest` symlink (active `QUEUED`, `RUNNING`, or `HELD` runs are never deleted):

```bash
runs clean                     # Interactive multi-select checkbox TUI
runs clean --failed -y         # Delete FAILED, SUBMIT_FAILED, and VANISHED runs
runs clean --aborted -y        # Delete ABORTED and INTERRUPTED runs
runs clean --dry -y            # Delete DRY_RUN capsules
runs clean --older-than 7d -y  # Delete inactive runs older than 7 days (supports d, h, m)
```

---

## 7. Under the Hood: Telemetry, Locking & Status Lifecycle

- **Signal & Preemption Auditing**: `job.sh` traps `SIGTERM`, `SIGINT`, and `SIGHUP`. When wrapped by `/usr/bin/time -v`, signals are forwarded directly to the child workload process (`pkill -<SIG> -P <time_pid>`) so your Python script can flush/checkpoint cleanly and `/usr/bin/time` still writes peak RSS (`MaxRSS`), CPU utilization, page faults, and context switches before exiting.
- **Locale Isolation**: Hardware probes (`lscpu`, `free`, `df`), `/usr/bin/time -v`, and `awk` run under `LC_ALL=C` / `LC_MESSAGES=C LC_NUMERIC=C` so Spanish locales (`es_ES.UTF-8`) on cluster nodes never break regex parsing or floating-point decimal dots, while your workload keeps its native UTF-8 locale.
- **CephFS / NFS Atomic Locking**: Concurrent sweep processes write per-process `repro/node_<proc>.json` files via atomic `os.replace` and coordinate updates to `run.json` through an exclusive `fcntl.flock` on `repro/.run_json.lock` (never unlinking the lock inode).
- **Status Lifecycle**:
  - `DRY_RUN`: Generated locally with `--dry-run`.
  - `SUBMIT_FAILED`: `condor_submit` returned an error or was not found.
  - `QUEUED` / `QUEUED (RETRY)`: Submitted and idle in `condor_q`.
  - `RUNNING` / `RUNNING (k active, m/N done)`: Executing on worker node(s).
  - `HELD (k held)`: Held by HTCondor (e.g. exceeded memory request or spool issue).
  - `COMPLETED`: All processes exited with code `0`.
  - `COMPLETED_WITH_ERRORS`: All sweep processes finished, but at least one exited non-zero.
  - `FAILED`: Process exited non-zero or terminated abnormally in `condor.log`.
  - `INCOMPLETE (m/N)`: Sweep left `condor_q` before all `N` processes recorded telemetry.
  - `INTERRUPTED`: Job caught `SIGTERM` / `SIGINT` / `SIGHUP` (eviction or preemption).
  - `ABORTED`: Removed via `runs stop` or `condor_rm`.
  - `VANISHED`: Left `condor_q` with no telemetry and no termination event in `condor.log`.
