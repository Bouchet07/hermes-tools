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
        run_data={
            "submission": {"working_directory": str(tmp_path)},
            "resources_requested": {
                "architecture": "A100",
                "cpus": None,
                "memory_mb": 131072,
                "gpus": 2,
                "sweep_arguments": ["--fold 0", "--fold 1", "--fold 2"],
                "node_include": ["nv-a100-01"],
                "node_exclude": ["her07-100.i3a.es"],
                "project_code": "100-100",
                "scratch_files": ["dataset.tar"],
            },
        },
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

    single_ok = tmp_path / "single_ok.log"
    single_ok.write_text(
        "005 (051.000.000) 2026-10-29 12:00:00 Job terminated.\n"
        "    (1) Normal termination (return value 0)\n"
        "...\n",
        encoding="utf-8",
    )
    assert inspect_condor_log_fate(single_ok, is_sweep=False) == "COMPLETED"


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


def test_code_snapshot_nongit_subdirectory_skips_runs_root(tmp_path: Path) -> None:
    project_root = tmp_path / "nongit_project"
    project_root.mkdir()
    runs_dir = project_root / "runs"
    runs_dir.mkdir()
    (runs_dir / "old_artifact.txt").write_text("old", encoding="utf-8")

    sub_dir = project_root / "deep" / "nested"
    sub_dir.mkdir(parents=True)
    target = sub_dir / "train.py"
    target.write_text("print('train')\n", encoding="utf-8")

    repro_dir = runs_dir / "r1" / "repro"
    repro_dir.mkdir(parents=True)

    create_code_snapshot(target, sub_dir, repro_dir, {"is_git": False})
    # Target in subdir must be copied
    assert (repro_dir / "code" / "train.py").is_file()
    # Old runs contents must not be copied
    assert not (repro_dir / "code" / "runs").exists()
    assert not (repro_dir / "code" / "old_artifact.txt").exists()


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


def test_follow_run_log_waits_for_start_and_auto_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import hermes_tools.runs_cli as runs_cli

    run_dir = tmp_path / "runs" / "2026-11-05_18-00-00_train_GPU3090"
    repro_dir = run_dir / "repro"
    repro_dir.mkdir(parents=True)
    log_file = run_dir / "train.out"

    atomic_write_json(
        run_dir / "run.json",
        {
            "run_id": run_dir.name,
            "status": "QUEUED",
            "submission": {"cluster_id": 600, "target": "train.py"},
            "resources_requested": {"architecture": "GPU3090", "sweep_count": 1},
            "executions": [],
        },
        repro_dir,
    )

    sleep_calls = 0

    def fake_sleep(sec: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls == 2:
            # Job starts and writes initial output
            log_file.write_text("epoch 1\n", encoding="utf-8")
        elif sleep_calls == 4:
            # Job writes more output and completes
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("epoch 2\n")
            atomic_write_json(
                run_dir / "run.json",
                {
                    "run_id": run_dir.name,
                    "status": "COMPLETED",
                    "submission": {"cluster_id": 600, "target": "train.py"},
                    "resources_requested": {"architecture": "GPU3090", "sweep_count": 1},
                    "executions": [{"process_id": 0, "exit_code": 0}],
                },
                repro_dir,
            )

    monkeypatch.setattr(runs_cli.time, "sleep", fake_sleep)
    monkeypatch.setattr(runs_cli, "query_live_condor_clusters", lambda: {})

    runs_cli.follow_run_log(run_dir, log_file, initial_lines=10)
    captured = capsys.readouterr().out
    assert "epoch 1" in captured
    assert "epoch 2" in captured
    assert "COMPLETED" in captured


def test_parse_condor_status_ads_and_runs_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import argparse
    import hermes_tools.runs_cli as runs_cli
    from hermes_tools.condor import parse_condor_status_ads, summarize_condor_pools

    fake_ads = [
        # Node 1: Free GPU3090
        {
            "Machine": "nv3090-01.i3a.es",
            "SlotType": "Partitionable",
            "State": "Unclaimed",
            "Architecture": "GPU3090",
            "Cpus": 32,
            "TotalSlotCpus": 32,
            "Memory": 128000,
            "TotalSlotMemory": 128000,
            "GPUs": 1,
            "TotalSlotGPUs": 1,
        },
        # Node 2: Busy GPU3090 (1 Dynamic slot claimed, Partitionable slot has 0 GPUs left)
        {
            "Machine": "nv3090-02.i3a.es",
            "SlotType": "Partitionable",
            "State": "Unclaimed",
            "Architecture": "GPU3090",
            "Cpus": 31,
            "TotalSlotCpus": 32,
            "Memory": 120000,
            "TotalSlotMemory": 128000,
            "GPUs": 0,
            "TotalSlotGPUs": 1,
        },
        {
            "Machine": "nv3090-02.i3a.es",
            "Name": "slot1_1@nv3090-02.i3a.es",
            "SlotType": "Dynamic",
            "State": "Claimed",
            "Activity": "Busy",
            "Architecture": "GPU3090",
            "Cpus": 1,
            "Memory": 8000,
            "GPUs": 1,
            "RemoteUser": "diego@i3a.es",
            "JobId": "1402.0",
            "EnteredCurrentActivity": 1000,
            "MyCurrentTime": 4660,
            "CUDADeviceName": "NVIDIA GeForce RTX 3090",
            "CUDAGlobalMemoryMb": 24576,
            "TotalLoadAvg": 4.25,
            "Disk": 500 * 1024 * 1024,
        },
        # Node 3: Mixed A100 (2 GPUs free, 2 GPUs claimed)
        {
            "Machine": "nv-a100-01.i3a.es",
            "SlotType": "Partitionable",
            "State": "Unclaimed",
            "Architecture": "A100",
            "Cpus": 64,
            "TotalSlotCpus": 128,
            "Memory": 256000,
            "TotalSlotMemory": 512000,
            "GPUs": 2,
            "TotalSlotGPUs": 4,
            "CUDADeviceName": "NVIDIA A100-SXM4-40GB",
        },
        {
            "Machine": "nv-a100-01.i3a.es",
            "Name": "slot1_1@nv-a100-01.i3a.es",
            "SlotType": "Dynamic",
            "State": "Claimed",
            "Activity": "Busy",
            "Architecture": "A100",
            "Cpus": 64,
            "Memory": 256000,
            "GPUs": 2,
            "RemoteUser": "alice@i3a.es",
            "JobId": "1550.3",
            "EnteredCurrentActivity": 2000,
            "MyCurrentTime": 2600,
        },
    ]

    nodes = parse_condor_status_ads(fake_ads)
    assert len(nodes) == 3
    by_host = {n["host"]: n for n in nodes}
    assert by_host["nv3090-01"]["state"] == "FREE"
    assert by_host["nv3090-01"]["gpus_free"] == 1
    assert by_host["nv3090-02"]["state"] == "BUSY"
    assert by_host["nv3090-02"]["gpus_free"] == 0
    assert by_host["nv3090-02"]["cpus_free"] == 0  # GPU3090 full-node lock zeroes free CPUs when busy
    assert by_host["nv3090-02"]["users"] == {"diego": 1}
    assert by_host["nv3090-02"]["claims"][0]["job_id"] == "1402.0"
    assert by_host["nv3090-02"]["claims"][0]["runtime_seconds"] == 3660.0
    assert "RTX 3090" in by_host["nv3090-02"]["gpu_model"]
    assert by_host["nv-a100-01"]["state"] == "MIXED"
    assert by_host["nv-a100-01"]["gpus_free"] == 2
    assert by_host["nv-a100-01"]["gpus_total"] == 4
    assert by_host["nv-a100-01"]["users"] == {"alice": 1}

    pools = summarize_condor_pools(nodes)
    assert pools["GPU3090"]["gpus_free"] == 1
    assert pools["GPU3090"]["gpus_total"] == 2
    assert pools["GPU3090"]["users"] == {"diego": 1}
    assert pools["A100"]["gpus_free"] == 2
    assert pools["A100"]["gpus_total"] == 4
    assert pools["A100"]["users"] == {"alice": 1}

    monkeypatch.setattr(runs_cli, "query_condor_nodes", lambda constraint=None: nodes)
    args = argparse.Namespace(
        arch=None,
        node=None,
        user=None,
        mine=False,
        free=False,
        busy=False,
        detail=False,
        where=None,
        watch=False,
        interval=2,
    )
    runs_cli.cmd_nodes(args)
    out = capsys.readouterr().out
    assert "GPU3090" in out
    assert "A100" in out
    assert "nv3090-01" in out
    assert "nv-a100-01" in out
    assert "diego" in out
    assert "alice" in out

    # Detail mode + user filter + local run tag cross-linking
    runs_root = tmp_path / "runs"
    rd = runs_root / "20261110_100000_ecg_train"
    rd.mkdir(parents=True)
    (rd / "run.json").write_text(
        json.dumps(
            {
                "run_id": rd.name,
                "status": "RUNNING",
                "submission": {"cluster_id": 1402, "target": "train.py", "tag": "ecg_train"},
            }
        ),
        encoding="utf-8",
    )

    detail_args = argparse.Namespace(
        arch=None,
        node=None,
        user="diego",
        mine=False,
        free=False,
        busy=False,
        detail=True,
        where=None,
        watch=False,
        interval=2,
    )
    runs_cli.cmd_nodes(detail_args, runs_root=runs_root)
    detail_out = capsys.readouterr().out
    assert "Active Slot Claims" in detail_out
    assert "1402.0" in detail_out
    assert "(ecg_train)" in detail_out
    assert "1h 01m 00s" in detail_out
    assert "nv-a100-01" not in detail_out

    # Invalid --where constraint error surfacing
    def _fail_query(constraint=None):
        raise ValueError("Can't parse constraint: GPUs ==")

    monkeypatch.setattr(runs_cli, "query_condor_nodes", _fail_query)
    bad_args = argparse.Namespace(
        arch=None,
        node=None,
        user=None,
        mine=False,
        free=False,
        busy=False,
        detail=False,
        where="GPUs ==",
        watch=False,
        interval=2,
    )
    with pytest.raises(SystemExit):
        runs_cli.cmd_nodes(bad_args)
    err_out = capsys.readouterr().out
    assert "Can't parse constraint: GPUs ==" in err_out


def test_parse_condor_better_analyze_and_runs_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import argparse
    import hermes_tools.runs_cli as runs_cli
    from hermes_tools.condor import parse_condor_better_analyze

    sample_output = """
-- Schedd: hermes2.i3a.es : <172.16.1.122:9618?...
The Requirements expression for job 1074510.000 is

    (TARGET.Arch == "X86_64") && (TARGET.OpSys == "LINUX") && (TARGET.Disk >= RequestDisk) && (TARGET.Memory >= RequestMemory) &&
    (TARGET.Cpus >= RequestCpus) && ((TARGET.FileSystemDomain == MY.FileSystemDomain) || (TARGET.HasFileTransfer))

Job 1074510.000 defines the following attributes:

    DiskUsage = 15
    FileSystemDomain = "i3a.es"
    RequestCpus = 16
    RequestDisk = DiskUsage
    RequestMemory = 16384

The Requirements expression for job 1074510.000 reduces to these conditions:

         Slots
Step    Matched  Condition
-----  --------  ---------
[0]         392  TARGET.Arch == "X86_64"
[1]         392  TARGET.OpSys == "LINUX"
[3]         392  TARGET.Disk >= RequestDisk
[5]          71  TARGET.Memory >= RequestMemory
[7]          28  TARGET.Cpus >= RequestCpus
[9]         392  TARGET.FileSystemDomain == MY.FileSystemDomain

No successful match recorded.
Last failed match: Tue Nov 10 11:45:46 2026

Reason for last match failure: PREEMPTION_REQUIREMENTS == False

1074510.000:  Run analysis summary ignoring user priority.  Of 55 machines,
     35 are rejected by your job's requirements
     14 reject your job because of their own requirements
      0 match and are already running your jobs
      0 match but are serving other users
      6 are able to run your job
"""

    parsed = parse_condor_better_analyze(sample_output)
    assert parsed["job_id"] == "1074510.0"
    assert parsed["attributes"]["RequestCpus"] == "16"
    assert parsed["attributes"]["RequestMemory"] == "16384"
    assert len(parsed["steps"]) == 6
    assert parsed["steps"][4]["slots"] == 28
    assert parsed["steps"][4]["hint"] == "16 CPUs"
    assert parsed["steps"][3]["hint"] == "16 GB (16384 MB)"
    assert parsed["machines_total"] == 55
    assert parsed["summary"]["able_to_run"] == 6
    assert parsed["summary"]["rejected_by_job"] == 35
    assert parsed["summary"]["rejected_by_node"] == 14
    assert "● ELIGIBLE — 6 of 55 machines match all requirements" in parsed["verdict"]
    assert "PREEMPTION_REQUIREMENTS == False" in parsed["scheduler_note"]

    runs_root = tmp_path / "runs"
    rd = runs_root / "2026-11-10_11-45-00_train_GPU3090"
    rd.mkdir(parents=True)
    (rd / "run.json").write_text(
        json.dumps(
            {
                "run_id": rd.name,
                "status": "QUEUED",
                "submission": {"cluster_id": 1074510, "target": "train.py"},
                "resources_requested": {"architecture": "GPU3090", "cpus": 16, "memory_mb": 16384, "gpus": 1},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(runs_cli, "query_live_condor_clusters", lambda: {1074510: {"total": 1, "running": 0, "idle": 1, "held": 0, "hold_reasons": []}})
    monkeypatch.setattr(runs_cli, "query_condor_better_analyze", lambda jid: parsed)

    # 1. Test `runs why` auto-selecting the QUEUED run
    why_args = argparse.Namespace(run=None, proc=0, raw=False)
    runs_cli.cmd_why(why_args, runs_root)
    why_out = capsys.readouterr().out
    assert "Queue Match Analysis" in why_out
    assert "1074510.0" in why_out
    assert "6 eligible" in why_out
    assert "TARGET.Cpus >= RequestCpus" in why_out

    # 2. Test `runs inspect` automatically embedding the Queue Match Analysis panel when QUEUED
    ins_args = argparse.Namespace(run=rd.name)
    runs_cli.cmd_inspect(ins_args, runs_root)
    ins_out = capsys.readouterr().out
    assert "Run Provenance & Telemetry" in ins_out
    assert "Queue Match Analysis" in ins_out
    assert "6 eligible" in ins_out


def test_parse_ssh_telemetry_and_runs_top(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import argparse
    import hermes_tools.runs_cli as runs_cli
    from hermes_tools.condor import parse_ssh_telemetry_frame

    raw_frame = """
===HERMES_FRAME_BEGIN===
---HOST---
nv3090-02.i3a.es
4.12 3.80 2.95 2/1024 149201
---GPU---
0, NVIDIA GeForce RTX 3090, 00000000:01:00.0, 96, 18841, 24576, 67, 312.4, 350.0
---GPU_APPS---
148920, 00000000:01:00.0, 18841
---PS---
148920 785.0 14889779 01:12:44 python python -u train.py --lr 0.001 --batch-size 64
148900   0.0     3200 01:12:45 bash /bin/bash /home/diego/runs/r1/repro/job.sh
149205   0.0     1800    00:00 ps ps -u diego -o pid=,pcpu=,rss=,etime=,comm=,args=
===HERMES_FRAME_END===
"""
    frame = parse_ssh_telemetry_frame(raw_frame)
    assert frame["host"] == "nv3090-02"
    assert frame["hostname"] == "nv3090-02.i3a.es"
    assert frame["load_avg"] == 4.12
    assert len(frame["gpus"]) == 1
    assert frame["gpus"][0]["model"] == "RTX 3090"
    assert frame["gpus"][0]["util_pct"] == 96.0
    assert frame["gpus"][0]["mem_used_mb"] == 18841.0
    assert len(frame["processes"]) == 1
    assert frame["processes"][0]["pid"] == 148920
    assert frame["processes"][0]["vram_mb"] == 18841.0
    assert "train.py" in frame["processes"][0]["command"]

    runs_root = tmp_path / "runs"
    rd = runs_root / "2026-11-10_12-00-00_train_GPU3090"
    rd.mkdir(parents=True)
    (rd / "run.json").write_text(
        json.dumps(
            {
                "run_id": rd.name,
                "status": "RUNNING",
                "submission": {"cluster_id": 1074510, "target": "train.py"},
                "resources_requested": {"architecture": "GPU3090", "cpus": 16, "memory_mb": 16384, "gpus": 1},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        runs_cli,
        "query_live_condor_clusters",
        lambda: {1074510: {"total": 1, "running": 1, "idle": 0, "held": 0, "hold_reasons": []}},
    )
    monkeypatch.setattr(runs_cli, "query_job_live_telemetry", lambda jid: frame)

    top_args = argparse.Namespace(run=None, proc=0, watch=False, interval=2)
    runs_cli.cmd_top(top_args, runs_root)
    top_out = capsys.readouterr().out
    assert "Live Worker Telemetry" in top_out
    assert "nv3090-02" in top_out
    assert "Allocated GPUs" in top_out
    assert "RTX 3090" in top_out
    assert "96 %" in top_out
    assert "Active Job Processes" in top_out
    assert "148920" in top_out
    assert "785.0%" in top_out

    # Test multiprocessing worker collapsing, parent Elapsed preservation, and DISK I/O state
    mp_frame_raw = """
===HERMES_FRAME_BEGIN===
---HOST---
her09-27.i3a.es
49.01 42.10 30.00 12/2048 1125000
---GPU---
---GPU_APPS---
---PS---
1124936 1124749 D   0.0  470016    00:00 python python prepare_dataset.py --dataset code15
1124914 1124749 D   0.0  470016    00:00 python python prepare_dataset.py --dataset code15
1124749 1124700 S  12.5  602112    16:42 python python prepare_dataset.py --dataset code15
===HERMES_FRAME_END===
"""
    mp_frame = parse_ssh_telemetry_frame(mp_frame_raw)
    assert len(mp_frame["processes"]) == 1
    proc_grp = mp_frame["processes"][0]
    assert proc_grp["pid"] == 1124749
    assert proc_grp["workers"] == 2
    assert proc_grp["elapsed"] == "16:42"
    assert proc_grp["state"] == "DISK I/O (2)"
    assert proc_grp["cpu_pct"] == 12.5

    # Test kernel `/proc/<pid>/stat` fallback when container/lxcfs `/proc/uptime` zeroes out `ps` etime/pcpu,
    # and verify `condor_ssh_to_job` helper processes are filtered out
    lxc_frame_raw = """
===HERMES_FRAME_BEGIN===
---HOST---
nv3090-09.i3a.es
0.99 1.00 1.00 1/1024 2935500
---GPU---
---GPU_APPS---
---PS---
2935294 2935200 D   0.0  647168    00:00 python python train_all.py --config_file configs/task.yaml
2935455 2935400 S   0.0    3072    00:00 condor_ssh_to_j /bin/bash /usr/lib/condor/libexec/condor_ssh_to_job_shell_setup
---PROC_STAT---
2935501 (cat) R 2935455 2935455 2935455 0 -1 4194304 90 0 0 0 0 0 0 0 20 0 1 0 10050000 8000000 200 18446744073709551615
2935294 (python) D 2935200 2935200 2935200 0 -1 4194304 1000 0 0 0 9000 3000 0 0 20 0 8 0 10026000 900000000 161792 18446744073709551615
===HERMES_FRAME_END===
"""
    lxc_frame = parse_ssh_telemetry_frame(lxc_frame_raw)
    assert len(lxc_frame["processes"]) == 1  # condor_ssh_to_job helper filtered out
    p_lxc = lxc_frame["processes"][0]
    assert p_lxc["pid"] == 2935294
    assert p_lxc["elapsed"] == "04:00"  # (10050000 - 10026000) / 100 = 240s = 04:00
    assert p_lxc["cpu_pct"] == 50.0  # (9000 + 3000) / 24000 * 100 = 50.0%

    # Verify live Host(s), Wall Time, and RAM for RUNNING runs in `runs ls` / `summarize_metrics`
    from hermes_tools.condor import reconcile_run
    from hermes_tools.core import summarize_metrics

    rec_data = reconcile_run(
        rd,
        json.loads((rd / "run.json").read_text(encoding="utf-8")),
        {
            1074510: {
                "total": 1,
                "running": 1,
                "idle": 0,
                "held": 0,
                "hold_reasons": [],
                "hosts": ["nv3090-09"],
                "max_runtime_sec": 754,
                "max_ram_mb": 2048.0,
            }
        },
    )
    host_s, dur_s, ram_s = summarize_metrics(rec_data)
    assert host_s == "nv3090-09"
    assert dur_s == "12m 34s"
    assert ram_s == "2.00 GB"

    # Verify `runs path` outputs clean unwrapped paths for run dir, --code, and --err
    p_args = argparse.Namespace(
        run="1", code=False, repro=False, out=False, err=False, condor=False, json=False, proc=None
    )
    runs_cli.cmd_path(p_args, runs_root)
    assert capsys.readouterr().out.strip() == str(rd.resolve())

    p_code_args = argparse.Namespace(
        run="latest", code=True, repro=False, out=False, err=False, condor=False, json=False, proc=None
    )
    runs_cli.cmd_path(p_code_args, runs_root)
    assert capsys.readouterr().out.strip() == str((rd / "repro" / "code").resolve())

    p_err_args = argparse.Namespace(
        run="latest", code=False, repro=False, out=False, err=True, condor=False, json=False, proc=None
    )
    runs_cli.cmd_path(p_err_args, runs_root)
    assert capsys.readouterr().out.strip() == str((rd / "train.err").resolve())


def test_dagman_generation_and_wait_script(tmp_path: Path) -> None:
    from hermes_tools.condor import generate_dag_file, generate_wait_parent_script

    sub_path = tmp_path / "job.sub"
    wait_py = tmp_path / "wait_parent.py"
    parent_dir = tmp_path / "runs" / "parent_run"
    parent_dir.mkdir(parents=True)

    dag_str = generate_dag_file(
        sub_path=sub_path,
        wait_script_path=wait_py,
        python_bin="/usr/bin/python3",
        parent_dir=parent_dir,
        parent_cluster_id=12345,
        parent_log_path=parent_dir / "condor.log",
        require_ok=True,
        poll_interval=2,
    )
    assert f"JOB main {sub_path.resolve().as_posix()}" in dag_str
    assert "SCRIPT PRE main /usr/bin/python3" in dag_str
    assert f"--parent-dir {parent_dir.resolve().as_posix()}" in dag_str
    assert "--parent-cluster 12345" in dag_str
    assert "--require-ok" in dag_str
    assert "--poll-interval 2" in dag_str

    script_content = generate_wait_parent_script()
    assert "#!/usr/bin/env python3" in script_content
    assert "def get_parent_fate(" in script_content
    assert "--require-ok" in script_content

    # Validate that script compiles without syntax errors
    compile(script_content, "wait_parent.py", "exec")

    # Verify execution behavior of wait_parent.py
    wait_script_file = tmp_path / "wait_parent.py"
    wait_script_file.write_text(script_content, encoding="utf-8")

    # 1. Success parent with --require-ok -> exit 0
    (parent_dir / "run.json").write_text(
        json.dumps({"status": "COMPLETED", "executions": [{"exit_code": 0}]}),
        encoding="utf-8",
    )
    res_ok = subprocess.run(
        [sys.executable, str(wait_script_file), "--parent-dir", str(parent_dir), "--require-ok", "--poll-interval", "1"],
        capture_output=True,
        text=True,
    )
    assert res_ok.returncode == 0
    assert "Proceeding to submit dependent job" in res_ok.stdout

    # 2. Failed parent with --require-ok -> exit 1
    (parent_dir / "run.json").write_text(
        json.dumps({"status": "FAILED", "executions": [{"exit_code": 1}]}),
        encoding="utf-8",
    )
    res_fail_req = subprocess.run(
        [sys.executable, str(wait_script_file), "--parent-dir", str(parent_dir), "--require-ok", "--poll-interval", "1"],
        capture_output=True,
        text=True,
    )
    assert res_fail_req.returncode == 1
    assert "Aborting dependent job (--require-ok)" in res_fail_req.stderr

    # 3. Failed parent without --require-ok -> exit 0
    res_fail_noreq = subprocess.run(
        [sys.executable, str(wait_script_file), "--parent-dir", str(parent_dir), "--poll-interval", "1"],
        capture_output=True,
        text=True,
    )
    assert res_fail_noreq.returncode == 0


def test_send_after_dependency_workflow_e2e(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    import argparse
    import hermes_tools.runs_cli as runs_cli
    import hermes_tools.send_cli as send_cli
    from hermes_tools.condor import get_all_cluster_ids, reconcile_run
    from hermes_tools.core import atomic_write_json, load_run_json

    monkeypatch.chdir(tmp_path)
    runs_root = tmp_path / "runs"
    parent_run = runs_root / "2026-11-12_10-00-00_train_GPU3090"
    p_repro = parent_run / "repro"
    p_repro.mkdir(parents=True)
    atomic_write_json(
        parent_run / "run.json",
        {
            "run_id": parent_run.name,
            "status": "RUNNING",
            "submission": {"cluster_id": 1000},
            "resources_requested": {"architecture": "GPU3090", "sweep_count": 1},
            "executions": [],
        },
        p_repro,
    )

    # Mock condor_q: parent cluster 1000 is running
    monkeypatch.setattr(
        send_cli,
        "query_live_condor_clusters",
        lambda: {1000: {"total": 1, "running": 1, "idle": 0, "held": 0}},
    )

    target_script = tmp_path / "step2.py"
    target_script.write_text("print('step2')\n", encoding="utf-8")

    # 1. Test dry-run with --after and --after-ok
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "send",
            "step2.py",
            "-a",
            "AMD",
            "-c",
            "8",
            "-m",
            "16G",
            "--after",
            "1",
            "--after-ok",
            "--dry-run",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        send_cli.main()
    assert exc.value.code == 0
    dry_out = capsys.readouterr().out
    assert "Dry run complete!" in dry_out
    assert "HTCondor DAGMan Workflow Definition (job.dag)" in dry_out
    dry_dirs = [d for d in runs_cli.list_run_dirs(runs_root) if "step2" in d.name]
    assert len(dry_dirs) == 1
    assert "--require-ok" in (dry_dirs[0] / "repro" / "job.dag").read_text(encoding="utf-8")
    import shutil
    shutil.rmtree(dry_dirs[0])

    # 2. Test live DAGMan submission
    submitted_dag_calls = []
    submitted_sub_calls = []

    def mock_subprocess_run(cmd, **kwargs):
        if cmd[0] == "condor_submit_dag":
            submitted_dag_calls.append(cmd)
            # Simulate HTCondor 9.0.9 `condor_submit_dag -no_submit` output
            dag_sub = Path(kwargs.get("cwd", ".")) / f"{cmd[-1]}.condor.sub"
            dag_sub.write_text(
                "universe        = scheduler\n"
                "executable      = /usr/bin/condor_dagman\n"
                f"arguments       = \"-p 0 -f -l . -Lockfile {cmd[-1]}.lock -AutoRescue 1 "
                f"-DoRescueFrom 0 -Dag {cmd[-1]} -Suppress_notification "
                "-CsdVersion $CondorVersion:' '9.0.9' '$ -Force -Dagman /usr/bin/condor_dagman\"\n"
                "queue\n",
                encoding="utf-8",
            )
            import subprocess

            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout="File for submitting this DAG to HTCondor: ...\n",
                stderr="",
            )
        if cmd[0] == "condor_submit":
            submitted_sub_calls.append(cmd)
            # Simulate the Hermes 2 wrapper: sed -i "/notification/d" on the submit file
            sub_file = Path(kwargs.get("cwd", ".")) / cmd[-1]
            kept = [
                ln for ln in sub_file.read_text(encoding="utf-8").splitlines()
                if "notification" not in ln
            ]
            sub_file.write_text("\n".join(kept) + "\n", encoding="utf-8")
            import subprocess

            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout="1 job(s) submitted to cluster 2000.\n",
                stderr="",
            )
        raise RuntimeError(f"Unexpected subprocess call: {cmd}")

    monkeypatch.setattr(send_cli.subprocess, "run", mock_subprocess_run)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "send",
            "step2.py",
            "-a",
            "AMD",
            "-c",
            "8",
            "-m",
            "16G",
            "--after",
            "1",
            "--after-ok",
            "-y",
        ],
    )
    send_cli.main()
    assert len(submitted_dag_calls) == 1
    assert "-f" in submitted_dag_calls[0]
    assert "-no_submit" in submitted_dag_calls[0]
    assert "-import_env" not in submitted_dag_calls[0]
    assert not Path(submitted_dag_calls[0][-1]).is_absolute()
    assert "job.dag" in submitted_dag_calls[0][-1]

    assert len(submitted_sub_calls) == 1
    assert "job.dag.condor.sub" in submitted_sub_calls[0][-1]

    # The DAGMan arguments line must survive the wrapper's `sed -i "/notification/d"`
    created_sub = Path(submitted_dag_calls[0][-1] + ".condor.sub")
    assert created_sub.is_file()
    sub_content = created_sub.read_text(encoding="utf-8")
    assert "Suppress_notification" not in sub_content
    arg_lines = [ln for ln in sub_content.splitlines() if ln.startswith("arguments")]
    assert len(arg_lines) == 1
    assert "-Dag " in arg_lines[0] and "-Lockfile " in arg_lines[0]
    assert "-CsdVersion $CondorVersion:' '9.0.9' '$" in arg_lines[0]

    # DAG pins DAGMAN_CONDOR_SUBMIT_EXE via a CONFIG file when condor_submit is on PATH
    dag_text = Path(submitted_dag_calls[0][-1]).read_text(encoding="utf-8")
    cfg_file = Path(submitted_dag_calls[0][-1]).parent / "dagman.config"
    if send_cli.shutil.which("condor_submit"):
        assert "CONFIG " in dag_text and cfg_file.is_file()
    else:
        assert "CONFIG " not in dag_text

    # Find the created dependent run directory
    child_run_dirs = [d for d in runs_cli.list_run_dirs(runs_root) if "step2" in d.name]
    assert len(child_run_dirs) == 1
    child_dir = child_run_dirs[0]
    child_rj = load_run_json(child_dir)

    assert child_rj["status"] == "WAITING"
    assert child_rj["dependency"]["parent_cluster_id"] == 1000
    assert child_rj["dependency"]["require_ok"] is True
    assert child_rj["dependency"]["dagman_cluster_id"] == 2000
    assert child_rj["dependency"]["job_cluster_id"] is None
    assert (child_dir / "repro" / "code" / "step2.py").is_file()  # Snapshot verified
    assert "--after" in child_rj["submission"]["equivalent_cli"]
    assert "--after-ok" in child_rj["submission"]["equivalent_cli"]

    # 3. Test reconciliation while DAGMan is waiting (dagman_cid in live_clusters, child not yet spawned)
    live_wait = {
        1000: {"total": 1, "running": 1, "idle": 0, "held": 0},
        2000: {"total": 1, "running": 1, "idle": 0, "held": 0, "child_clusters": []},
    }
    rec_wait = reconcile_run(child_dir, child_rj, live_wait)
    assert rec_wait["_live_badge"] == "WAITING (after #1000)"

    # Test `runs why` on WAITING run
    monkeypatch.setattr(runs_cli, "query_live_condor_clusters", lambda: live_wait)
    why_args = argparse.Namespace(run=child_dir.name, proc=0, raw=False)
    runs_cli.cmd_why(why_args, runs_root)
    why_out = capsys.readouterr().out
    assert "WAITING FOR PARENT RUN" in why_out
    assert "Parent must exit 0 (--after-ok)" in why_out

    # 4. Test reconciliation when DAGMan spawns child job 2001
    live_dispatched = {
        2000: {"total": 1, "running": 1, "idle": 0, "held": 0, "child_clusters": [2001]},
        2001: {
            "total": 1,
            "running": 1,
            "idle": 0,
            "held": 0,
            "hosts": ["her09-20"],
            "max_runtime_sec": 45,
            "max_ram_mb": 512.0,
        },
    }
    rec_active = reconcile_run(child_dir, child_rj, live_dispatched)
    assert rec_active["status"] == "RUNNING"
    assert rec_active["_live_badge"] == "RUNNING"
    assert rec_active["_live_hosts"] == ["her09-20"]
    assert 2001 in get_all_cluster_ids(rec_active)

    # 5. Test `runs stop` aborts DAGMan cluster
    rm_calls = []

    def mock_condor_rm(cmd, **kwargs):
        rm_calls.append(cmd)
        import subprocess

        return subprocess.CompletedProcess(cmd, 0, stdout="Cluster 2000 was removed.\n", stderr="")

    monkeypatch.setattr(runs_cli.subprocess, "run", mock_condor_rm)
    monkeypatch.setattr(runs_cli, "query_live_condor_clusters", lambda: live_dispatched)
    stop_args = argparse.Namespace(run=child_dir.name, yes=True, all=False)
    runs_cli.cmd_stop(stop_args, runs_root)
    assert any("2000" in cmd for cmd in rm_calls)
    stopped_rj = load_run_json(child_dir)
    assert stopped_rj["status"] == "ABORTED"


def test_send_after_parent_failure_cancelled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    import argparse
    import hermes_tools.runs_cli as runs_cli
    import hermes_tools.send_cli as send_cli
    from hermes_tools.condor import reconcile_run
    from hermes_tools.core import atomic_write_json, load_run_json

    monkeypatch.chdir(tmp_path)
    runs_root = tmp_path / "runs"
    parent_run = runs_root / "2026-11-12_11-00-00_train_A100"
    p_repro = parent_run / "repro"
    p_repro.mkdir(parents=True)
    atomic_write_json(
        parent_run / "run.json",
        {
            "run_id": parent_run.name,
            "status": "FAILED",
            "submission": {"cluster_id": 3000},
            "resources_requested": {"architecture": "A100", "sweep_count": 1},
            "executions": [{"process_id": 0, "exit_code": 1}],
        },
        p_repro,
    )

    child_run = runs_root / "2026-11-12_11-05-00_eval_GPU3090"
    c_repro = child_run / "repro"
    c_repro.mkdir(parents=True)
    atomic_write_json(
        child_run / "run.json",
        {
            "run_id": child_run.name,
            "status": "WAITING",
            "submission": {"cluster_id": 4000},
            "resources_requested": {"architecture": "GPU3090", "sweep_count": 1},
            "dependency": {
                "parent_run_id": parent_run.name,
                "parent_run_dir": str(parent_run),
                "parent_cluster_id": 3000,
                "require_ok": True,
                "dagman_cluster_id": 4000,
                "job_cluster_id": None,
            },
            "executions": [],
        },
        c_repro,
    )

    # When DAGMan leaves condor_q after PRE script exited 1 because parent failed:
    rec = reconcile_run(child_run, load_run_json(child_run), live_clusters={})
    assert rec["status"] == "CANCELLED (parent failed)"

    # Test `runs clean --failed` picks up CANCELLED (parent failed) run
    clean_args = argparse.Namespace(failed=True, aborted=False, dry=False, older_than=None, yes=True)
    runs_cli.cmd_clean(clean_args, runs_root)
    clean_out = capsys.readouterr().out
    assert "Deleted 2 run(s)" in clean_out or "Deleted 1 run(s)" in clean_out

    # Test direct submit when parent already completed successfully
    p_ok = runs_root / "2026-11-12_12-00-00_done_AMD"
    p_ok_repro = p_ok / "repro"
    p_ok_repro.mkdir(parents=True)
    atomic_write_json(
        p_ok / "run.json",
        {
            "run_id": p_ok.name,
            "status": "COMPLETED",
            "submission": {"cluster_id": 5000},
            "resources_requested": {"architecture": "AMD", "sweep_count": 1},
            "executions": [{"process_id": 0, "exit_code": 0}],
        },
        p_ok_repro,
    )

    submit_normal_calls = []

    def mock_submit(cmd, **kwargs):
        submit_normal_calls.append(cmd)
        import subprocess

        return subprocess.CompletedProcess(cmd, 0, stdout="1 job(s) submitted to cluster 6000.\n", stderr="")

    monkeypatch.setattr(send_cli.subprocess, "run", mock_submit)
    target = tmp_path / "post.py"
    target.write_text("print('post')\n", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["send", "post.py", "-a", "AMD", "-c", "2", "-m", "4G", "--after", p_ok.name, "-y"])
    send_cli.main()
    assert len(submit_normal_calls) == 1
    assert submit_normal_calls[0][0] == "condor_submit"  # Directly submitted without DAGMan delay!

    # Test rejection when parent failed and --after-ok is passed
    p_fail = runs_root / "2026-11-12_13-00-00_fail_AMD"
    p_fail_repro = p_fail / "repro"
    p_fail_repro.mkdir(parents=True)
    atomic_write_json(
        p_fail / "run.json",
        {
            "run_id": p_fail.name,
            "status": "FAILED",
            "submission": {"cluster_id": 7000},
            "resources_requested": {"architecture": "AMD", "sweep_count": 1},
            "executions": [{"process_id": 0, "exit_code": 1}],
        },
        p_fail_repro,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["send", "post.py", "-a", "AMD", "-c", "2", "-m", "4G", "--after", p_fail.name, "--after-ok", "-y"],
    )
    with pytest.raises(SystemExit) as exc:
        send_cli.main()
    assert exc.value.code == 1
    err_out = capsys.readouterr().out
    assert "Aborting because --after-ok was specified" in err_out


def test_multistage_pipeline_dynamic_tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that wait_parent.py handles multi-stage chains (A -> B -> C) where B is WAITING on A."""
    from hermes_tools.condor import generate_wait_parent_script

    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True)

    # 1. Run A (Running)
    run_a = runs_root / "2026-11-12_10-00-00_stageA_AMD"
    a_repro = run_a / "repro"
    a_repro.mkdir(parents=True)
    atomic_write_json(
        run_a / "run.json",
        {
            "run_id": run_a.name,
            "status": "RUNNING",
            "submission": {"cluster_id": 1001},
            "resources_requested": {"sweep_count": 1},
            "executions": [],
        },
        a_repro,
    )

    # 2. Run B (Waiting on A)
    run_b = runs_root / "2026-11-12_10-05-00_stageB_AMD"
    b_repro = run_b / "repro"
    b_repro.mkdir(parents=True)
    atomic_write_json(
        run_b / "run.json",
        {
            "run_id": run_b.name,
            "status": "WAITING",
            "submission": {"cluster_id": 2001},
            "dependency": {
                "parent_run_id": run_a.name,
                "parent_run_dir": str(run_a),
                "parent_cluster_id": 1001,
                "dagman_cluster_id": 2001,
                "job_cluster_id": None,
                "require_ok": True,
            },
            "resources_requested": {"sweep_count": 1},
            "executions": [],
        },
        b_repro,
    )

    # 3. Run C (Waiting on B)
    run_c = runs_root / "2026-11-12_10-10-00_stageC_AMD"
    c_repro = run_c / "repro"
    c_repro.mkdir(parents=True)
    atomic_write_json(
        run_c / "run.json",
        {
            "run_id": run_c.name,
            "status": "WAITING",
            "submission": {"cluster_id": 3001},
            "dependency": {
                "parent_run_id": run_b.name,
                "parent_run_dir": str(run_b),
                "parent_cluster_id": 2001,
                "dagman_cluster_id": 3001,
                "job_cluster_id": None,
                "require_ok": True,
            },
            "resources_requested": {"sweep_count": 1},
            "executions": [],
        },
        c_repro,
    )

    script_path = tmp_path / "wait_parent.py"
    script_path.write_text(generate_wait_parent_script(), encoding="utf-8")

    # Import helper function dynamically from the generated script
    import importlib.util

    spec = importlib.util.spec_from_file_location("wait_parent", str(script_path))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # C waiting on B: B is still WAITING on A -> must NOT be finished!
    fin, fate = mod.get_parent_fate(run_b, fallback_cid=2001)
    assert fin is False
    assert fate == "WAITING"

    # Now A completes
    atomic_write_json(
        run_a / "run.json",
        {
            "run_id": run_a.name,
            "status": "COMPLETED",
            "submission": {"cluster_id": 1001},
            "resources_requested": {"sweep_count": 1},
            "executions": [{"process_id": 0, "exit_code": 0}],
        },
        a_repro,
    )

    # B waiting on A: A is finished
    fin_a, fate_a = mod.get_parent_fate(run_a, fallback_cid=1001)
    assert fin_a is True
    assert fate_a == "COMPLETED"

    # B transitions to RUNNING
    atomic_write_json(
        run_b / "run.json",
        {
            "run_id": run_b.name,
            "status": "RUNNING",
            "submission": {"cluster_id": 2002},
            "dependency": {
                "parent_run_id": run_a.name,
                "parent_run_dir": str(run_a),
                "parent_cluster_id": 1001,
                "dagman_cluster_id": 2001,
                "job_cluster_id": 2002,
                "require_ok": True,
            },
            "resources_requested": {"sweep_count": 1},
            "executions": [],
        },
        b_repro,
    )

    # C waiting on B: B is now RUNNING -> must NOT be finished!
    # Mock condor_q to show cluster 2002 running
    monkeypatch.setattr(mod, "query_condor_jobs_for_cids", lambda cids, dids: [{"ClusterId": 2002}] if 2002 in cids else [])
    fin_b, fate_b = mod.get_parent_fate(run_b, fallback_cid=2001)
    assert fin_b is False
    assert fate_b == "RUNNING"

    # B finishes execution with success
    atomic_write_json(
        run_b / "run.json",
        {
            "run_id": run_b.name,
            "status": "COMPLETED",
            "submission": {"cluster_id": 2002},
            "resources_requested": {"sweep_count": 1},
            "executions": [{"process_id": 0, "exit_code": 0}],
        },
        b_repro,
    )
    monkeypatch.setattr(mod, "query_condor_jobs_for_cids", lambda cids, dids: [])

    # Now C checking B sees COMPLETED
    fin_b_done, fate_b_done = mod.get_parent_fate(run_b, fallback_cid=2001)
    assert fin_b_done is True
    assert fate_b_done == "COMPLETED"


def test_sweep_parent_partial_failure_cancelled(tmp_path: Path) -> None:
    """Verifies that when parent sweep finishes COMPLETED_WITH_ERRORS, child is cancelled."""
    from hermes_tools.condor import generate_wait_parent_script

    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True)

    parent_run = runs_root / "2026-11-12_10-00-00_sweep_AMD"
    p_repro = parent_run / "repro"
    p_repro.mkdir(parents=True)
    atomic_write_json(
        parent_run / "run.json",
        {
            "run_id": parent_run.name,
            "status": "COMPLETED_WITH_ERRORS",
            "submission": {"cluster_id": 5001},
            "resources_requested": {"sweep_count": 2},
            "executions": [
                {"process_id": 0, "exit_code": 0},
                {"process_id": 1, "exit_code": 1},
            ],
        },
        p_repro,
    )

    child_run = runs_root / "2026-11-12_10-05-00_child_AMD"
    c_repro = child_run / "repro"
    c_repro.mkdir(parents=True)
    atomic_write_json(
        child_run / "run.json",
        {
            "run_id": child_run.name,
            "status": "WAITING",
            "submission": {"cluster_id": 6001},
            "executions": [],
        },
        c_repro,
    )

    wait_py = tmp_path / "wait_parent.py"
    wait_py.write_text(generate_wait_parent_script(), encoding="utf-8")

    res = subprocess.run(
        [
            sys.executable,
            str(wait_py),
            "--child-dir",
            str(child_run),
            "--parent-dir",
            str(parent_run),
            "--require-ok",
            "--poll-interval",
            "1",
        ],
        capture_output=True,
        text=True,
    )

    assert res.returncode == 1
    assert "Aborting dependent job (--require-ok)" in res.stderr

    # Child run.json must be atomically marked CANCELLED (parent failed)
    child_rj = load_run_json(child_run)
    assert child_rj["status"] == "CANCELLED (parent failed)"
    assert "finished with status COMPLETED_WITH_ERRORS" in child_rj.get("cancelled_reason", "")


def test_fast_child_completion_discovery_from_logs(tmp_path: Path) -> None:
    """Verifies that reconcile_run discovers child cluster ID and completes when condor_q is empty."""
    from hermes_tools.condor import extract_child_cluster_id_from_logs, reconcile_run

    run_dir = tmp_path / "runs" / "2026-11-12_12-00-00_fast_AMD"
    repro_dir = run_dir / "repro"
    repro_dir.mkdir(parents=True)

    # repro/condor.log shows cluster 777777 executed and completed
    (repro_dir / "condor.log").write_text(
        "000 (777777.000.000) 2026-11-12 12:00:00 Job submitted from host ...\n"
        "...\n"
        "001 (777777.000.000) 2026-11-12 12:00:01 Job executing on host ...\n"
        "...\n"
        "005 (777777.000.000) 2026-11-12 12:00:05 Job terminated.\n"
        "    (1) Normal termination (return value 0)\n"
        "...\n",
        encoding="utf-8",
    )
    (repro_dir / "node_0.json").write_text(
        json.dumps({"process_id": 0, "exit_code": 0, "walltime_sec": 4.5}),
        encoding="utf-8",
    )

    data = {
        "run_id": run_dir.name,
        "status": "WAITING",
        "submission": {"cluster_id": 700000},  # DAGMan CID
        "resources_requested": {"sweep_count": 1},
        "dependency": {
            "dagman_cluster_id": 700000,
            "job_cluster_id": None,
            "parent_cluster_id": 699999,
        },
        "executions": [],
    }
    atomic_write_json(run_dir / "run.json", data, repro_dir)

    assert extract_child_cluster_id_from_logs(repro_dir) == 777777

    # reconcile_run with live_clusters={} (job already left condor_q)
    rec = reconcile_run(run_dir, data, live_clusters={})
    assert rec["status"] == "COMPLETED"
    assert rec["dependency"]["job_cluster_id"] == 777777
    assert rec["submission"]["cluster_id"] == 777777
    assert len(rec["executions"]) == 1
    assert rec["executions"][0]["exit_code"] == 0


def test_provenance_equivalent_cli_resolves_run_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that equivalent_cli stores the resolved folder name, not 'latest' or '1'."""
    import hermes_tools.send_cli as send_cli
    from hermes_tools.core import update_latest_symlink

    monkeypatch.chdir(tmp_path)
    runs_root = tmp_path / "runs"
    parent_dir = runs_root / "2026-11-12_09-00-00_train_GPU3090"
    p_repro = parent_dir / "repro"
    p_repro.mkdir(parents=True)
    atomic_write_json(
        parent_dir / "run.json",
        {
            "run_id": parent_dir.name,
            "status": "RUNNING",
            "submission": {"cluster_id": 8888},
            "resources_requested": {"sweep_count": 1},
            "executions": [],
        },
        p_repro,
    )
    update_latest_symlink(runs_root, parent_dir)

    target_py = tmp_path / "eval.py"
    target_py.write_text("print('eval')\n", encoding="utf-8")

    monkeypatch.setattr(
        send_cli,
        "query_live_condor_clusters",
        lambda: {8888: {"total": 1, "running": 1, "idle": 0, "held": 0}},
    )

    # Run with --after latest
    monkeypatch.setattr(
        sys,
        "argv",
        ["send", "eval.py", "-a", "AMD", "--after", "latest", "--dry-run"],
    )
    with pytest.raises(SystemExit) as exc:
        send_cli.main()
    assert exc.value.code == 0

    from hermes_tools.runs_cli import list_run_dirs

    child_dirs = [d for d in list_run_dirs(runs_root) if "eval" in d.name]
    assert len(child_dirs) == 1
    child_rj = load_run_json(child_dirs[0])

    eq_cli = child_rj["submission"]["equivalent_cli"]
    assert f"--after {parent_dir.name}" in eq_cli
    assert "--after latest" not in eq_cli


def test_subdirectory_send_runs_root_discovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that running send from a project subdirectory locates the parent runs/ directory."""
    import hermes_tools.send_cli as send_cli

    project_root = tmp_path / "my_project"
    project_root.mkdir()
    runs_root = project_root / "runs"
    runs_root.mkdir()

    parent_dir = runs_root / "2026-11-12_08-00-00_base_AMD"
    p_repro = parent_dir / "repro"
    p_repro.mkdir(parents=True)
    atomic_write_json(
        parent_dir / "run.json",
        {
            "run_id": parent_dir.name,
            "status": "RUNNING",
            "submission": {"cluster_id": 9901},
            "resources_requested": {"sweep_count": 1},
            "executions": [],
        },
        p_repro,
    )

    # Subdirectory inside project
    sub_dir = project_root / "experiments" / "deep_sub"
    sub_dir.mkdir(parents=True)
    monkeypatch.chdir(sub_dir)

    target_py = sub_dir / "task.py"
    target_py.write_text("print('task')\n", encoding="utf-8")

    monkeypatch.setattr(
        send_cli,
        "query_live_condor_clusters",
        lambda: {9901: {"total": 1, "running": 1, "idle": 0, "held": 0}},
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["send", "task.py", "-a", "AMD", "--after", "1", "--dry-run"],
    )
    with pytest.raises(SystemExit) as exc:
        send_cli.main()
    assert exc.value.code == 0

    # Ensure run capsule was created in project_root/runs, NOT in sub_dir/runs
    assert not (sub_dir / "runs").exists()
    created = [d for d in runs_root.iterdir() if "task" in d.name]
    assert len(created) == 1


def test_raw_condor_cluster_chaining(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies chaining after a raw HTCondor cluster ID not present in runs/."""
    import hermes_tools.send_cli as send_cli

    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").mkdir()

    target_py = tmp_path / "raw_chain.py"
    target_py.write_text("print('raw')\n", encoding="utf-8")

    # Cluster 999999 is running in condor_q
    monkeypatch.setattr(
        send_cli,
        "query_live_condor_clusters",
        lambda: {999999: {"total": 1, "running": 1, "idle": 0, "held": 0}},
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["send", "raw_chain.py", "-a", "AMD", "--after", "999999", "--after-ok", "--dry-run"],
    )
    with pytest.raises(SystemExit) as exc:
        send_cli.main()
    assert exc.value.code == 0

    from hermes_tools.runs_cli import list_run_dirs

    child_dirs = [d for d in list_run_dirs(tmp_path / "runs") if "raw_chain" in d.name]
    assert len(child_dirs) == 1
    child_rj = load_run_json(child_dirs[0])

    assert child_rj["dependency"]["parent_cluster_id"] == 999999
    assert child_rj["dependency"]["parent_run_dir"] is None
    assert child_rj["dependency"]["require_ok"] is True
    assert "--after 999999" in child_rj["submission"]["equivalent_cli"]


def test_guard_waiting_and_cancelled_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """Verifies that runs sync, runs retry, runs stop, and follow_run_log guard WAITING and CANCELLED runs."""
    import argparse
    import hermes_tools.runs_cli as runs_cli

    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True)

    # 1. Waiting run
    wait_run = runs_root / "2026-11-12_10-00-00_wait_AMD"
    w_repro = wait_run / "repro"
    w_repro.mkdir(parents=True)
    atomic_write_json(
        wait_run / "run.json",
        {
            "run_id": wait_run.name,
            "status": "WAITING",
            "submission": {"cluster_id": 5555},
            "resources_requested": {"sweep_count": 1},
            "dependency": {"dagman_cluster_id": 5555, "parent_cluster_id": 4444},
            "executions": [],
        },
        w_repro,
    )

    monkeypatch.setattr(
        runs_cli,
        "query_live_condor_clusters",
        lambda: {5555: {"total": 1, "running": 1, "idle": 0, "held": 0}},
    )

    # Retrying a WAITING run should be disallowed
    retry_args = argparse.Namespace(run=wait_run.name, all=False, in_place=False, new_run=False, update_code=False)
    runs_cli.cmd_retry(retry_args, runs_root)
    out = capsys.readouterr().out
    assert "retry a waiting run" in out

    # Stopping a WAITING run should remove DAGMan cluster and set status ABORTED
    rm_calls = []

    def mock_condor_rm(cmd, **kwargs):
        rm_calls.append(cmd)
        import subprocess

        return subprocess.CompletedProcess(cmd, 0, stdout="Cluster 5555 was removed.\n", stderr="")

    monkeypatch.setattr(runs_cli.subprocess, "run", mock_condor_rm)
    stop_args = argparse.Namespace(run=wait_run.name, yes=True, all=False)
    runs_cli.cmd_stop(stop_args, runs_root)
    assert any("5555" in cmd for cmd in rm_calls)
    stopped_rj = load_run_json(wait_run)
    assert stopped_rj["status"] == "ABORTED"

    # Stopping an already ABORTED run informs user cleanly
    monkeypatch.setattr(runs_cli, "query_live_condor_clusters", lambda: {})
    runs_cli.cmd_stop(stop_args, runs_root)
    out_stop = capsys.readouterr().out
    assert "has already finished" in out_stop

    # Syncing code / following logs on a CANCELLED run should be guarded cleanly
    atomic_write_json(
        wait_run / "run.json",
        {**stopped_rj, "status": "CANCELLED (parent failed)"},
        w_repro,
    )
    runs_cli.cmd_stop(stop_args, runs_root)
    assert "is already" in capsys.readouterr().out

    sync_args = argparse.Namespace(run=wait_run.name)
    runs_cli.cmd_sync(sync_args, runs_root)
    out_sync = capsys.readouterr().out
    assert "cancelled runs" in out_sync

    # follow_run_log on a CANCELLED run should return without crashing
    runs_cli.follow_run_log(wait_run, wait_run / "wait.out")
    out_follow = capsys.readouterr().out
    assert "Run was cancelled" in out_follow


def test_reconcile_waiting_race_and_terminal_skip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse
    import hermes_tools.condor as condor
    import hermes_tools.runs_cli as runs_cli

    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    parent_dir = runs_root / "2026-11-12_10-00-00_parent_GPU3090"
    parent_dir.mkdir()
    (parent_dir / "repro").mkdir()
    parent_rj = {
        "status": "COMPLETED",
        "submission": {"cluster_id": 1074962},
        "resources_requested": {"sweep_count": 1},
        "executions": [{"process_id": 0, "status": "COMPLETED"}],
    }
    (parent_dir / "run.json").write_text(json.dumps(parent_rj), encoding="utf-8")

    child_dir = runs_root / "2026-11-12_11-00-00_child_GPU3090"
    child_dir.mkdir()
    (child_dir / "repro").mkdir()
    (child_dir / "repro" / "job.dag.dagman.out").write_text(
        "11/12/26 11:00:01 DAGMan running...\n", encoding="utf-8"
    )
    child_rj = {
        "status": "WAITING",
        "submission": {"cluster_id": 1075087},
        "resources_requested": {"sweep_count": 1},
        "dependency": {
            "dagman_cluster_id": 1075087,
            "parent_cluster_id": 1074962,
            "parent_run_dir": str(parent_dir),
            "parent_run_id": parent_dir.name,
        },
    }
    (child_dir / "run.json").write_text(json.dumps(child_rj), encoding="utf-8")

    # Even if condor_q temporarily misses dagman_cid while dagman.out has not exited, keep WAITING
    rec = condor.reconcile_run(child_dir, dict(child_rj), {})
    assert rec["status"] == "WAITING"
    assert rec["_live_badge"] == "WAITING (after #1074962)"
    assert json.loads((child_dir / "run.json").read_text(encoding="utf-8"))["status"] == "WAITING"

    # If wait_parent.py already wrote CANCELLED (parent failed) on disk, sync from disk
    (child_dir / "run.json").write_text(
        json.dumps({**child_rj, "status": "CANCELLED (parent failed)"}),
        encoding="utf-8",
    )
    rec_cancelled = condor.reconcile_run(child_dir, dict(child_rj), {})
    assert rec_cancelled["status"] == "CANCELLED (parent failed)"

    # Terminal runs (including CANCELLED) must NOT trigger query_live_condor_clusters in cmd_ls
    called = []
    monkeypatch.setattr(runs_cli, "query_live_condor_clusters", lambda: called.append(1) or {})
    ls_args = argparse.Namespace(
        quiet=False,
        full=False,
        all=False,
        tag=None,
        arch=None,
        status=None,
        target=None,
        since=None,
        until=None,
        sort=None,
        n=10,
    )
    runs_cli.cmd_ls(ls_args, runs_root)
    assert called == []


def test_wait_parent_query_condor_jobs_uses_constraint(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess
    import hermes_tools.condor as condor

    ns: dict = {}
    exec(condor._WAIT_PARENT_SCRIPT.split("def main()")[0], ns)
    q_fn = ns["query_condor_jobs_for_cids"]

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps([{"ClusterId": 101, "JobStatus": 2, "DAGManJobId": None}]),
            stderr="",
        )

    monkeypatch.setattr(ns["shutil"], "which", lambda exe: "/usr/bin/condor_q")
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert q_fn(set(), set()) == []
    assert calls == []

    res = q_fn({102, 101}, {201})
    assert len(res) == 1
    assert "-constraint" in calls[0]
    c_idx = calls[0].index("-constraint")
    constraint_expr = calls[0][c_idx + 1]
    assert "member(ClusterId, {101,102})" in constraint_expr
    assert "member(DAGManJobId, {201})" in constraint_expr


def test_retry_in_place_updates_dependency_job_cluster_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse
    import subprocess
    import hermes_tools.runs_cli as runs_cli

    runs_root = tmp_path / "runs"
    run_dir = runs_root / "2026-11-12_12-00-00_dep_retry_GPU3090"
    repro = run_dir / "repro"
    repro.mkdir(parents=True)
    (repro / "job.sub").write_text("executable = /bin/true\nqueue\n", encoding="utf-8")
    (repro / "condor.log").write_text(
        "005 (111.000.000) 2026-11-12 12:05:00 Job terminated.\n"
        "    (1) Normal termination (return value 1)\n...\n",
        encoding="utf-8",
    )
    rj = {
        "status": "FAILED",
        "submission": {"cluster_id": 111},
        "resources_requested": {"sweep_count": 1},
        "executions": [{"process_id": 0, "status": "FAILED"}],
        "dependency": {"dagman_cluster_id": 110, "job_cluster_id": 111, "parent_cluster_id": 100},
    }
    atomic_write_json(run_dir / "run.json", rj, repro)

    monkeypatch.setattr(runs_cli, "query_live_condor_clusters", lambda: {})
    monkeypatch.setattr(
        runs_cli.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout="1 job(s) submitted to cluster 222.\n", stderr=""),
    )

    args = argparse.Namespace(run=run_dir.name, all=False, in_place=True, new_run=False, update_code=False, force=False, yes=True)
    runs_cli.cmd_retry(args, runs_root)

    updated = load_run_json(run_dir)
    assert updated["submission"]["cluster_id"] == 222
    assert updated["dependency"]["job_cluster_id"] == 222


def test_why_and_top_raw_cluster_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse
    import hermes_tools.runs_cli as runs_cli

    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    monkeypatch.setattr(runs_cli, "query_live_condor_clusters", lambda: {})

    rendered = []
    monkeypatch.setattr(
        runs_cli,
        "_render_queue_analysis_panel",
        lambda job_target, raw_mode=False: rendered.append((job_target, raw_mode)),
    )
    runs_cli.cmd_why(argparse.Namespace(run="999999", proc=0, raw=False), runs_root)
    assert rendered == [("999999.0", False)]

    job_target, run_name, arch = runs_cli._resolve_running_job_context(
        argparse.Namespace(run="999999", proc=2), runs_root
    )
    assert (job_target, run_name, arch) == ("999999.2", None, None)










def test_dagman_survives_site_condor_submit_wrapper(tmp_path: Path) -> None:
    """Hermes 2's /usr/local/bin/condor_submit runs `sed -i "/notification/d"` on the submit file."""
    from hermes_tools.condor import (
        generate_dag_file,
        generate_dagman_config,
        sanitize_dagman_condor_sub,
    )

    raw = (
        "universe        = scheduler\n"
        "notification    = never\n"
        "arguments       = \"-p 0 -f -l . -Lockfile r/job.dag.lock -Dag r/job.dag "
        "-Suppress_notification -CsdVersion $CondorVersion:' '9.0.9' '$ -Force\"\n"
        "queue\n"
    )
    cleaned = sanitize_dagman_condor_sub(raw)
    after_wrapper = [ln for ln in cleaned.splitlines() if "notification" not in ln]
    assert after_wrapper == cleaned.strip().splitlines()
    assert any(ln.startswith("arguments") and "-Dag r/job.dag" in ln for ln in after_wrapper)

    assert generate_dagman_config(None) is None
    cfg = generate_dagman_config("/usr/local/bin/condor_submit")
    assert "DAGMAN_CONDOR_SUBMIT_EXE = /usr/local/bin/condor_submit" in cfg

    cfg_path = tmp_path / "dagman.config"
    cfg_path.write_text(cfg, encoding="utf-8")
    dag = generate_dag_file(
        sub_path=tmp_path / "job.sub",
        wait_script_path=tmp_path / "wait_parent.py",
        python_bin="/usr/bin/python3",
        parent_cluster_id=123,
        config_path=cfg_path,
    )
    dag_lines = dag.splitlines()
    assert dag_lines[1].startswith("CONFIG ") and dag_lines[1].endswith("dagman.config")
    assert dag_lines[2].startswith("JOB main ")


def test_job_sh_contains_telemetry_daemon_and_teardown(tmp_path: Path) -> None:
    from hermes_tools.condor import generate_bash_wrapper

    run_dir = tmp_path / "runs" / "2026-11-13_12-00-00_telem_GPU3090"
    repro_dir = run_dir / "repro"
    repro_dir.mkdir(parents=True)
    sh = generate_bash_wrapper(
        exec_type="python",
        target_exec_path=repro_dir / "code" / "train.py",
        venv_path=None,
        pythonpath_dir=repro_dir / "code",
        project_cwd=tmp_path,
        run_dir=run_dir,
        repro_dir=repro_dir,
        submit_epoch=1700000000,
        sweep_count=1,
        requested_cpus=4,
    )
    assert "TELEM_TMP=" in sh
    assert "TELEM_PID=$!" in sh
    assert 'kill -TERM "${TELEM_PID}"' in sh
    assert 'wait "${TELEM_PID}"' in sh
    assert "timestamp_iso,epoch,elapsed_s,num_procs,num_threads,io_wait_procs" in sh
    assert "gpu_sm_clock_mhz" in sh
    assert sh.index('wait "${TELEM_PID}"') < sh.index("UPD_RUN_JSON")


def test_runs_plot_and_path_telemetry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    import argparse
    import hermes_tools.condor as condor
    import hermes_tools.runs_cli as runs_cli

    runs_root = tmp_path / "runs"
    run_dir = runs_root / "2026-11-13_12-10-00_train_GPU3090_exp1"
    repro = run_dir / "repro"
    repro.mkdir(parents=True)

    header = (
        "timestamp_iso,epoch,elapsed_s,num_procs,num_threads,io_wait_procs,"
        "cpu_total_pct,cpu_user_pct,cpu_sys_pct,rss_mb,pss_mb,node_ram_avail_mb,"
        "node_load_1m,io_read_mb_s,io_write_mb_s,disk_read_mb_s,disk_write_mb_s,"
        "maj_faults_delta,gpu_util_pct,gpu_mem_bw_pct,vram_used_mb,vram_total_mb,"
        "gpu_temp_c,gpu_power_w,gpu_sm_clock_mhz\n"
    )
    rows_txt = []
    for i in range(15):
        rows_txt.append(
            f"2026-11-13T12:10:{i * 5:02d}+01:00,{1700000000 + i * 5:.1f},{i * 5:.1f},5,16,1,"
            f"145.0,110.0,35.0,4096.0,2048.0,64000.0,4.20,120.000,5.000,85.000,2.000,0,"
            f"28.0,22.0,6144.0,24576.0,61.0,145.0,1695.0\n"
        )
    telem_csv = repro / "telemetry_0.csv"
    telem_csv.write_text(header + "".join(rows_txt), encoding="utf-8")

    summary = condor.compute_telemetry_summary(telem_csv)
    assert summary is not None
    assert summary["samples"] == 15
    assert summary["gpu_util_avg"] == 28.0
    assert summary["io_wait_pct"] == 100.0

    rj = {
        "run_id": run_dir.name,
        "status": "COMPLETED",
        "submission": {"target": "train.py", "tag": "exp1", "cluster_id": 9001},
        "resources_requested": {"architecture": "GPU3090", "sweep_count": 1},
        "executions": [
            {
                "process_id": 0,
                "exit_code": 0,
                "runtime_seconds": 75,
                "peak_ram_mb": 2048.0,
                "telemetry_file": "repro/telemetry_0.csv",
                "telemetry_summary": summary,
            }
        ],
    }
    atomic_write_json(run_dir / "run.json", rj, repro)

    # Test `runs path -t`
    runs_cli.cmd_path(
        argparse.Namespace(run="1", code=False, repro=False, json=False, out=False, err=False, condor=False, telemetry=True, proc=0),
        runs_root,
    )
    path_out = capsys.readouterr().out.strip()
    assert Path(path_out).resolve() == telem_csv.resolve()

    # Test `runs plot 1`
    monkeypatch.setattr(runs_cli, "query_live_condor_clusters", lambda: {})
    runs_cli.cmd_plot(
        argparse.Namespace(run="1", proc=0, watch=False, interval=5, width=40, save=None),
        runs_root,
    )
    plot_out = capsys.readouterr().out
    assert "GPU Compute (%)" in plot_out
    assert "Host RAM PSS (GB)" in plot_out
    assert "Disk I/O (MB/s)" in plot_out
    assert "GPU Starvation Detected" in plot_out


def test_send_from_run_and_retry_new_run_preserves_frozen_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import argparse
    import hermes_tools.runs_cli as runs_cli

    script = tmp_path / "train.py"
    helper = tmp_path / "helper.py"
    script.write_text('print("v1_historical")\n', encoding="utf-8")
    helper.write_text("VAL = 1\n", encoding="utf-8")

    # 1. Submit initial run (dry-run)
    res1 = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_tools.send_cli",
            "train.py",
            "-a",
            "GPU3090",
            "--dry-run",
            "--no-venv",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert res1.returncode == 0, res1.stderr
    runs_root = tmp_path / "runs"
    orig_dirs = sorted(runs_root.iterdir())
    assert len(orig_dirs) == 1
    orig_dir = orig_dirs[0]

    # Mark original run COMPLETED so `runs retry --new-run` also exercises COMPLETED support
    orig_rj = load_run_json(orig_dir)
    orig_rj["status"] = "COMPLETED"
    orig_rj["executions"] = [{"process_id": 0, "exit_code": 0}]
    atomic_write_json(orig_dir / "run.json", orig_rj, orig_dir / "repro")

    # 2. Mutate working tree and delete helper.py
    script.write_text('print("v2_modified")\n', encoding="utf-8")
    helper.unlink()

    # 3. Clone via `send --from-run 1 --dry-run --no-venv -t cloned`
    res2 = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_tools.send_cli",
            "--from-run",
            "1",
            "--dry-run",
            "--no-venv",
            "-t",
            "cloned",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert res2.returncode == 0, res2.stderr
    cloned_dirs = [d for d in runs_root.iterdir() if d.is_dir() and d != orig_dir and "cloned" in d.name]
    assert len(cloned_dirs) == 1
    cloned_dir = cloned_dirs[0]
    assert 'print("v1_historical")' in (cloned_dir / "repro" / "code" / "train.py").read_text(encoding="utf-8")
    assert (cloned_dir / "repro" / "code" / "helper.py").is_file()
    cloned_rj = load_run_json(cloned_dir)
    assert cloned_rj["submission"]["cloned_from_run"] == orig_dir.name
    assert cloned_rj["git"]["cloned_from_run"] == orig_dir.name

    # 4. Verify `runs retry <completed_run> --new-run` passes `--from-run` when `-u` is omitted
    execv_calls = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runs_cli, "query_live_condor_clusters", lambda: {})
    monkeypatch.setattr(runs_cli.os, "execv", lambda exe, cmd: execv_calls.append(cmd))
    runs_cli.cmd_retry(
        argparse.Namespace(
            run=orig_dir.name,
            all=False,
            in_place=False,
            new_run=True,
            update_code=False,
            force=False,
            yes=True,
        ),
        runs_root,
    )
    assert len(execv_calls) == 1
    assert "--from-run" in execv_calls[0]
    assert orig_dir.name in execv_calls[0]
