"""Offline execution tests for the GPU pair's setup script (spec 04)."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "config/deployment_configurations/qwen_qwen3_8b_rtx3090_oct_22.sh"
)


def _prepare(tmp_path):
    commands = tmp_path / "bin"
    commands.mkdir()
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    pip = commands / "pip"
    pip.write_text("#!/bin/sh\nexit 0\n")
    pip.chmod(0o755)
    python = commands / "python3"
    python.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, sys\n"
        "code = sys.argv[2]\n"
        "if 'site.getsitepackages' in code:\n"
        "    print(os.environ['STUB_SITE'])\n"
        "elif 'snapshot_download' in code:\n"
        "    pathlib.Path(os.environ['STUB_CALL']).write_text(code)\n"
        "    if os.environ.get('STUB_FAIL') == '1':\n"
        "        sys.exit(1)\n"
        "    print(os.environ['STUB_SNAPSHOT'])\n"
        "else:\n"
        "    sys.exit('Unexpected Python invocation: ' + code)\n"
    )
    python.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{commands}:{os.environ['PATH']}",
        "STUB_SITE": str(site_packages),
        "STUB_SNAPSHOT": str(snapshot),
        "STUB_CALL": str(tmp_path / "download-call.txt"),
    }
    return env, snapshot, site_packages


def _run(tmp_path, env):
    return subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_setup_uses_pinned_snapshot_and_replaces_identifier(tmp_path):
    env, snapshot, site_packages = _prepare(tmp_path)
    result = _run(tmp_path, env)
    assert result.returncode == 0, result.stderr
    call = (tmp_path / "download-call.txt").read_text()
    assert re.search(r"revision=['\"][0-9a-f]{40}['\"]", call), call
    identifier = tmp_path / "Qwen" / "Qwen3-8B"
    assert identifier.is_symlink()
    assert identifier.resolve() == snapshot
    hook = (site_packages / "sitecustomize.py").read_text()
    assert 'os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")' in hook
    assert 'os.environ.setdefault("VLLM_MARLIN_USE_ATOMIC_ADD", "1")' in hook

    old_snapshot = tmp_path / "old-snapshot"
    old_snapshot.mkdir()
    identifier.unlink()
    identifier.symlink_to(old_snapshot, target_is_directory=True)
    result = _run(tmp_path, env)
    assert result.returncode == 0, result.stderr
    assert identifier.resolve() == snapshot
    assert not list(old_snapshot.iterdir()), "setup must not write inside the old cache"
    result = _run(tmp_path, env)
    assert result.returncode == 0, result.stderr
    assert identifier.resolve() == snapshot
    assert not list(snapshot.glob("snapshot")), (
        "repeated setup must not mutate its target"
    )


@pytest.mark.parametrize("failure", ["download", "missing_config"])
def test_failed_snapshot_preparation_preserves_identifier(tmp_path, failure):
    env, snapshot, _ = _prepare(tmp_path)
    old_snapshot = tmp_path / "old-snapshot"
    old_snapshot.mkdir()
    identifier = tmp_path / "Qwen" / "Qwen3-8B"
    identifier.parent.mkdir()
    identifier.symlink_to(old_snapshot, target_is_directory=True)
    if failure == "download":
        env["STUB_FAIL"] = "1"
    else:
        (snapshot / "config.json").unlink()
    result = _run(tmp_path, env)
    assert result.returncode != 0
    assert identifier.resolve() == old_snapshot
    assert not list(old_snapshot.iterdir())
