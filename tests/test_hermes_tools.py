from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_tools.condor import (
    format_condor_arg_line,
    generate_bash_wrapper,
    generate_retry_sub_file,
    generate_sub_file,
    inspect_condor_log_fate,
    reconcile_run,
)
from hermes_tools.core import (
    atomic_write_json,
    expand_arg_sweeps,
    load_project_config,
    load_run_json,
    parse_memory_to_mb,
    sanitize_tag,
)
from hermes_tools.snapshot import create_code_snapshot


def test_parse_memory_to_mb() -> None:
    assert parse_memory_to_mb("16G") == 16384
    assert parse_memory_to_mb("16GB") == 16384
    assert parse_memory_to_mb("1.5G") == 1536
    assert parse_memory_to_mb("4096M") == 4096
    assert parse_memory_to_mb("4096") == 4096
    assert parse_memory_to_mb("1T") == 1048576
    with pytest.raises(ValueError):
        parse_memory_to_mb("0G")
    with pytest.raises(ValueError):
        parse_memory_to_mb("")


def test_expand_arg_sweeps() -> None:
    assert expand_arg_sweeps("--fold {0..3}") == [
        "--fold 0",
        "--fold 1",
        "--fold 2",
        "--fold 3",
    ]
    # Zero-padded ranges
    assert expand_arg_sweeps("--seed {00..03}") == [
        "--seed 00",
        "--seed 01",
        "--seed 02",
        "--seed 03",
    ]
    # Cartesian product + semicolon
    assert expand_arg_sweeps("--lr {1e-3,1e-4} --opt {adam,sgd}; --eval-only") == [
        "--lr 1e-3 --opt adam",
        "--lr 1e-3 --opt sgd",
        "--lr 1e-4 --opt adam",
        "--lr 1e-4 --opt sgd",
        "--eval-only",
    ]
    # JSON dict with colons is not split as a sweep
    assert expand_arg_sweeps('--cfg \'{"a": 1, "b": 2}\'') == ['--cfg \'{"a": 1, "b": 2}\'']


def test_sanitize_tag() -> None:
    assert sanitize_tag("  exp 1 / lr=1e-4  ") == "exp_1_lr_1e-4"


def test_format_condor_arg_line() -> None:
    assert format_condor_arg_line("--epochs 100 --name 'my run'") == "--epochs 100 --name 'my run'"
    assert format_condor_arg_line('--msg "hello \\"world\\""') == "--msg 'hello \"\"world\"\"'"


def test_generate_sub_and_retry_preserves_directives(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "2026-10-29_train_A100"
    repro_dir = run_dir / "repro"
    repro_dir.mkdir(parents=True)
    sh_path = repro_dir / "job.sh"

    sub_text = generate_sub_file(
        sh_path=sh_path,
        project_cwd=tmp_path,
        run_dir=run_dir,
        repro_dir=repro_dir,
        script_stem="train",
        arch="A100",
        cpus=None,
        mem_mb=131072,
        gpus=2,
        sweep_args=["--fold 0", "--fold 1", "--fold 2"],
        nodes_include=["nv-a100-01"],
        nodes_exclude=["her07-100.i3a.es"],
        project_code="100-100",
        scratch_files=["dataset.tar"],
    )
    orig_sub_path = repro_dir / "job.sub"
    orig_sub_path.write_text(sub_text, encoding="utf-8")

    retry_text = generate_retry_sub_file(
        orig_sub_path=orig_sub_path,
        run_dir=run_dir,
        repro_dir=repro_dir,
        script_stem="train",
        is_sweep=True,
        targets=[(1, "--fold 1"), (2, "--fold 2")],
        retry_epoch=1700000000,
    )

    assert '+Architecture           = "A100"' in retry_text
    assert "request_gpus            = 2" in retry_text
    assert "request_memory          = 131072" in retry_text
    assert 'TARGET.Machine == "nv-a100-01.i3a.es"' in retry_text
    assert 'TARGET.Machine != "her07-100.i3a.es"' in retry_text
    assert '+Proyecto               = "100-100"' in retry_text
    assert "transfer_input_files    = dataset.tar" in retry_text
    assert 'environment             = "SUBMIT_EPOCH=1700000000"' in retry_text
    assert "1, --fold 1" in retry_text
    assert "2, --fold 2" in retry_text


def test_bash_wrapper_cluster_hardening(tmp_path: Path) -> None:
    sh = generate_bash_wrapper(
        exec_type="python",
        target_exec_path=tmp_path / "train.py",
        venv_path=tmp_path / ".venv",
        pythonpath_dir=tmp_path / "repro" / "code",
        project_cwd=tmp_path,
        run_dir=tmp_path / "runs" / "r1",
        repro_dir=tmp_path / "runs" / "r1" / "repro",
        submit_epoch=1700000000,
        sweep_count=4,
        requested_cpus=8,
    )
    assert "pkill -\"${SHORT_SIG}\" -P \"${CHILD_PID}\"" in sh
    assert "LC_ALL=C lscpu" in sh
    assert "LC_MESSAGES=C LC_NUMERIC=C /usr/bin/time -v" in sh
    assert 'if [ ! -x "${UPDATER_PYTHON}" ]; then' in sh
    assert "SUBMIT_PYTHON=" in sh
    assert 'CMD=("${PYTHON_BIN}" -u "${TARGET_EXEC}" "$@")' in sh
    assert "── Hermes 2 Job ·" in sh


def test_reconcile_run_persists_sweep_nodes_and_heals_status(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "sweep_run"
    repro_dir = run_dir / "repro"
    repro_dir.mkdir(parents=True)

    initial_data = {
        "run_id": "sweep_run",
        "status": "RUNNING",
        "submission": {"cluster_id": 42},
        "resources_requested": {"architecture": "GPU3090", "sweep_count": 2},
        "executions": [],
    }
    atomic_write_json(run_dir / "run.json", initial_data, repro_dir)

    # Simulate two worker processes writing node_0.json and node_1.json
    (repro_dir / "node_0.json").write_text(
        json.dumps({"process_id": 0, "exit_code": 0, "hostname": "n1.i3a.es"}),
        encoding="utf-8",
    )
    (repro_dir / "node_1.json").write_text(
        json.dumps({"process_id": 1, "exit_code": 1, "hostname": "n2.i3a.es"}),
        encoding="utf-8",
    )

    # Cluster 42 has left condor_q (live_clusters == {})
    reconciled = reconcile_run(run_dir, load_run_json(run_dir), live_clusters={})
    assert reconciled["status"] == "COMPLETED_WITH_ERRORS"
    assert len(reconciled["executions"]) == 2

    # Verify it was persisted to disk as well
    persisted = load_run_json(run_dir)
    assert persisted["status"] == "COMPLETED_WITH_ERRORS"
    assert len(persisted["executions"]) == 2


def test_inspect_condor_log_fate_sweep_vs_single(tmp_path: Path) -> None:
    log_file = tmp_path / "condor.log"
    log_file.write_text(
        "005 (050.000.000) 2026-10-29 12:00:00 Job terminated.\n"
        "    (1) Normal termination (return value 1)\n"
        "...\n"
        "005 (050.001.000) 2026-10-29 12:01:00 Job terminated.\n"
        "    (1) Normal termination (return value 0)\n"
        "...\n",
        encoding="utf-8",
    )
    # In a sweep, even though proc 1 succeeded last, proc 0 failed earlier:
    assert inspect_condor_log_fate(log_file, is_sweep=True) == "FAILED"


def test_code_snapshot_removes_deleted_files_on_sync(tmp_path: Path) -> None:
    src_a = tmp_path / "main.py"
    src_b = tmp_path / "old_helper.py"
    src_a.write_text("print('v1')\n", encoding="utf-8")
    src_b.write_text("x = 1\n", encoding="utf-8")

    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    repro_dir = tmp_path / "runs" / "r1" / "repro"
    repro_dir.mkdir(parents=True)

    from hermes_tools.snapshot import get_git_info

    create_code_snapshot(src_a, tmp_path, repro_dir, get_git_info(tmp_path))
    assert (repro_dir / "code" / "old_helper.py").is_file()

    # Delete old_helper.py and re-snapshot (as in `runs sync`)
    src_b.unlink()
    create_code_snapshot(src_a, tmp_path, repro_dir, get_git_info(tmp_path))
    assert not (repro_dir / "code" / "old_helper.py").exists()


def test_end_to_end_send_dry_run_and_runs_cli(tmp_path: Path) -> None:
    script = tmp_path / "train.py"
    script.write_text("print('hello hermes')\n", encoding="utf-8")
    (tmp_path / ".hermes.toml").write_text(
        'arch = "GPU3090"\nproject = "200-300"\n', encoding="utf-8"
    )

    cfg = load_project_config(tmp_path)
    assert cfg["arch"] == "GPU3090"
    assert cfg["project"] == "200-300"

    # Run `send train.py --dry-run --no-venv -- --cfg '{"a": 1, "b": 2}'`
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_tools.send_cli",
            "train.py",
            "--dry-run",
            "--no-venv",
            "-t",
            "smoke",
            "--",
            "--cfg",
            '{"a": 1, "b": 2}',
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert res.returncode == 0, res.stderr
    runs_dir = tmp_path / "runs"
    run_folders = [p for p in runs_dir.iterdir() if p.is_dir() and not p.is_symlink()]
    assert len(run_folders) == 1
    rj = load_run_json(run_folders[0])
    assert rj["status"] == "DRY_RUN"
    assert rj["resources_requested"]["architecture"] == "GPU3090"
    assert rj["resources_requested"]["project_code"] == "200-300"
    # Verify JSON arg was not split into a sweep
    assert rj["resources_requested"]["sweep_count"] == 1

    # Verify `runs ls` and `runs inspect`
    res_ls = subprocess.run(
        [sys.executable, "-m", "hermes_tools.runs_cli", "ls"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert res_ls.returncode == 0
    assert "DRY_RUN" in res_ls.stdout

    res_ins = subprocess.run(
        [sys.executable, "-m", "hermes_tools.runs_cli", "inspect", "1"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert res_ins.returncode == 0
    assert "GPU3090" in res_ins.stdout


def test_sweep_equivalent_cli_stays_compact(tmp_path: Path) -> None:
    script = tmp_path / "train.py"
    script.write_text("print('sweep')\n", encoding="utf-8")

    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_tools.send_cli",
            "train.py",
            "-a",
            "GPU3090",
            "--dry-run",
            "--no-venv",
            "-s",
            "{0..5}",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert res.returncode == 0, res.stderr
    runs_dir = tmp_path / "runs"
    run_folders = [p for p in runs_dir.iterdir() if p.is_dir() and not p.is_symlink()]
    rj = load_run_json(run_folders[0])
    assert rj["resources_requested"]["sweep_count"] == 6
    assert "-s '{0..5}'" in rj["submission"]["equivalent_cli"] or "-s {0..5}" in rj["submission"]["equivalent_cli"]
    assert "-s 0 -s 1" not in rj["submission"]["equivalent_cli"]


def test_stop_does_not_mark_completed_run_as_aborted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse
    import hermes_tools.runs_cli as runs_cli

    run_dir = tmp_path / "runs" / "2026-11-03_train_GPU3090"
    repro_dir = run_dir / "repro"
    repro_dir.mkdir(parents=True)
    atomic_write_json(
        run_dir / "run.json",
        {
            "run_id": run_dir.name,
            "status": "COMPLETED",
            "submission": {"cluster_id": 999},
            "resources_requested": {"architecture": "GPU3090", "sweep_count": 1},
            "executions": [{"process_id": 0, "exit_code": 0}],
        },
        repro_dir,
    )
    monkeypatch.setattr(runs_cli, "query_live_condor_clusters", lambda: {})
    args = argparse.Namespace(run=run_dir.name, yes=True)
    runs_cli.cmd_stop(args, tmp_path / "runs")
    assert load_run_json(run_dir)["status"] == "COMPLETED"


def test_freeze_python_env_uses_mtime_cache_and_dir_size_skips_code(tmp_path: Path) -> None:
    from hermes_tools.core import dir_size_mb
    from hermes_tools.snapshot import freeze_python_env

    fake_venv = tmp_path / ".venv"
    fake_venv.mkdir()
    (fake_venv / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    cache_file = fake_venv / ".hermes_env_cache.txt"
    cache_file.write_text("torch==2.5.0\n", encoding="utf-8")

    repro_dir = tmp_path / "runs" / "r1" / "repro"
    code_dir = repro_dir / "code"
    code_dir.mkdir(parents=True)
    # Write 2 MB into repro/code and 1 MB into run output root
    (code_dir / "big_code.py").write_bytes(b"x" * (2 * 1024 * 1024))
    (tmp_path / "runs" / "r1" / "checkpoint.bin").write_bytes(b"y" * (1 * 1024 * 1024))

    freeze_python_env(fake_venv, repro_dir)
    assert (repro_dir / "env.txt").read_text(encoding="utf-8") == "torch==2.5.0\n"

    # dir_size_mb skips repro/code by default so only the 1 MB checkpoint is counted
    mb_default = dir_size_mb(tmp_path / "runs" / "r1")
    mb_full = dir_size_mb(tmp_path / "runs" / "r1", include_code_snapshot=True)
    assert 0.95 <= mb_default <= 1.15
    assert mb_full >= 2.95


def test_env_config_and_auto_cuda_home_exports(tmp_path: Path) -> None:
    (tmp_path / ".hermes.toml").write_text(
        'arch = "GPU3090"\n\n'
        "[env]\n"
        'CUDA_HOME = "/mnt/cephfs/home/bio/dbouchet/cuda-12.1"\n'
        'TORCH_CUDA_ARCH_LIST = "8.6"\n'
        'WANDB_MODE = "offline"\n',
        encoding="utf-8",
    )
    cfg = load_project_config(tmp_path)
    assert cfg["env"]["CUDA_HOME"] == "/mnt/cephfs/home/bio/dbouchet/cuda-12.1"
    assert cfg["env"]["TORCH_CUDA_ARCH_LIST"] == "8.6"
    assert cfg["env"]["WANDB_MODE"] == "offline"

    sh = generate_bash_wrapper(
        exec_type="bash",
        target_exec_path=tmp_path / "pipeline.sh",
        venv_path=None,
        pythonpath_dir=tmp_path / "repro" / "code",
        project_cwd=tmp_path,
        run_dir=tmp_path / "runs" / "r1",
        repro_dir=tmp_path / "runs" / "r1" / "repro",
        submit_epoch=1700000000,
        sweep_count=1,
        requested_cpus=None,
        env_vars=cfg["env"],
    )
    assert "export PYTHONUNBUFFERED=1" in sh
    assert "export HOME=" in sh
    assert "export CUDA_HOME=/mnt/cephfs/home/bio/dbouchet/cuda-12.1" in sh
    assert "export TORCH_CUDA_ARCH_LIST=8.6" in sh
    assert "export WANDB_MODE=offline" in sh
    assert 'export PATH="${CUDA_HOME}/bin:${PATH}"' in sh
    assert 'export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"' in sh


def test_watch_terminal_cache_detects_in_place_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse
    import hermes_tools.runs_cli as runs_cli

    run_dir = tmp_path / "runs" / "2026-11-05_16-00-00_train_GPU3090"
    repro_dir = run_dir / "repro"
    repro_dir.mkdir(parents=True)
    atomic_write_json(
        run_dir / "run.json",
        {
            "run_id": run_dir.name,
            "status": "FAILED",
            "submission": {"cluster_id": 100},
            "resources_requested": {"architecture": "GPU3090", "sweep_count": 1},
            "executions": [{"process_id": 0, "exit_code": 1}],
        },
        repro_dir,
    )
    monkeypatch.setattr(runs_cli, "query_live_condor_clusters", lambda: {101: {"total": 1, "running": 1, "idle": 0, "held": 0}})

    args = argparse.Namespace(limit=25, status=None, tag=None)
    cache: dict = {}
    _, sig1 = runs_cli._build_ls_table_with_state(args, tmp_path / "runs", cache)
    assert run_dir in cache
    assert any("FAILED" in cell for cell in sig1[0])

    # Now simulate in-place retry updating run.json -> QUEUED (RETRY) with cluster 101
    atomic_write_json(
        run_dir / "run.json",
        {
            "run_id": run_dir.name,
            "status": "QUEUED (RETRY)",
            "submission": {"cluster_id": 101, "retry_clusters": [100]},
            "resources_requested": {"architecture": "GPU3090", "sweep_count": 1},
            "executions": [],
        },
        repro_dir,
    )
    _, sig2 = runs_cli._build_ls_table_with_state(args, tmp_path / "runs", cache)
    assert sig2 != sig1
    assert any("RUNNING" in cell for cell in sig2[0])


def test_tail_file_lines_o1_seek_and_diff_symlinks(tmp_path: Path) -> None:
    import hermes_tools.runs_cli as runs_cli
    from hermes_tools.core import tail_file_lines

    log_f = tmp_path / "big.log"
    lines = [f"line-{i:05d}" for i in range(2500)]
    log_f.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Small block size exercises multi-chunk backward seeking
    assert tail_file_lines(log_f, 5, block_size=64) == lines[-5:]
    assert tail_file_lines(tmp_path / "missing.log", 10) == []

    # Verify _diff_directories does not crash on broken symlinks or read binary blobs
    dir_a = tmp_path / "snap_a"
    dir_b = tmp_path / "snap_b"
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_a / "train.py").write_text("lr = 1e-3\n", encoding="utf-8")
    (dir_b / "train.py").write_text("lr = 1e-4\n", encoding="utf-8")
    (dir_a / "weights.bin").write_bytes(b"\x00\x01\x02" * 100)
    (dir_b / "weights.bin").write_bytes(b"\x00\x01\x02" * 200)

    diff_out = runs_cli._diff_directories(dir_a, dir_b)
    assert "-lr = 1e-3" in diff_out
    assert "+lr = 1e-4" in diff_out
    assert "[binary file:" in diff_out


def test_hold_reason_surfacing_and_batch_stop_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse
    import hermes_tools.runs_cli as runs_cli

    run_dir = tmp_path / "runs" / "2026-11-05_17-00-00_train_A100"
    repro_dir = run_dir / "repro"
    repro_dir.mkdir(parents=True)
    atomic_write_json(
        run_dir / "run.json",
        {
            "run_id": run_dir.name,
            "status": "QUEUED",
            "submission": {"cluster_id": 505},
            "resources_requested": {"architecture": "A100", "sweep_count": 1},
            "executions": [],
        },
        repro_dir,
    )

    live_map = {
        505: {
            "total": 1,
            "running": 0,
            "idle": 0,
            "held": 1,
            "hold_reasons": ["Transfer input file dataset.tar not found"],
        }
    }
    rec = reconcile_run(run_dir, load_run_json(run_dir), live_clusters=live_map)
    assert rec["_live_badge"] == "HELD (1 held)"
    assert rec["_hold_reasons"] == ["Transfer input file dataset.tar not found"]

    monkeypatch.setattr(runs_cli, "query_live_condor_clusters", lambda: live_map)
    monkeypatch.setattr(
        runs_cli.subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(args=a[0], returncode=0, stdout="Cluster 505 removed", stderr=""),
    )
    args = argparse.Namespace(run=None, all=True, yes=True)
    runs_cli.cmd_stop(args, tmp_path / "runs")
    assert load_run_json(run_dir)["status"] == "ABORTED"

