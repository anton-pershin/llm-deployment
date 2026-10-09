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
        "if sys.argv[1].endswith('.py'):\n"
        "    pathlib.Path(os.environ['STUB_HELPER_CALL']).write_text(' '.join(sys.argv[1:]))\n"
        "    if os.environ.get('STUB_HELPER_FAIL') == '1': sys.exit(1)\n"
        "    path = pathlib.Path(os.environ['STUB_PREPARED'])\n"
        "    path.mkdir(exist_ok=True)\n"
        "    (path / 'config.json').write_text('{}')\n"
        "    print(path)\n"
        "    sys.exit(0)\n"
        "code = sys.argv[2]\n"
        "if 'site.getsitepackages' in code:\n"
        "    print(os.environ['STUB_SITE'])\n"
        "elif 'snapshot_download' in code:\n"
        "    with pathlib.Path(os.environ['STUB_CALL']).open('a') as log: log.write(code)\n"
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
        "STUB_PREPARED": str(tmp_path / "prepared"),
        "STUB_HELPER_CALL": str(tmp_path / "helper-call.txt"),
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
    assert len(re.findall(r"revision=['\"][0-9a-f]{40}['\"]", call)) == 2, call
    assert "kaitchup/Qwen3-8B-autoround-4bit-gptq" in call
    assert "b7e026b92d0019c20745eae7f843fac019c9c6e0" in call
    assert "JunHowie/Qwen3-8B-GPTQ-Int8" in call
    assert "e131f54dea2ba1f99bbee218f75548ed00646cb9" in call
    helper_call = (tmp_path / "helper-call.txt").read_text()
    assert "qwen_qwen3_8b_rtx3090_oct_22.py" in helper_call
    assert "--format gptq" in helper_call
    identifier = tmp_path / "Qwen" / "Qwen3-8B"
    assert identifier.is_symlink()
    assert identifier.resolve() == Path(env["STUB_PREPARED"])
    hook = (site_packages / "sitecustomize.py").read_text()
    assert 'os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")' in hook

    old_snapshot = tmp_path / "old-snapshot"
    old_snapshot.mkdir()
    identifier.unlink()
    identifier.symlink_to(old_snapshot, target_is_directory=True)
    result = _run(tmp_path, env)
    assert result.returncode == 0, result.stderr
    assert identifier.resolve() == Path(env["STUB_PREPARED"])
    assert not list(old_snapshot.iterdir()), "setup must not write inside the old cache"
    result = _run(tmp_path, env)
    assert result.returncode == 0, result.stderr
    assert identifier.resolve() == Path(env["STUB_PREPARED"])
    assert not list(snapshot.glob("snapshot")), (
        "repeated setup must not mutate its target"
    )


@pytest.mark.parametrize("failure", ["download", "missing_config", "helper"])
def test_failed_snapshot_preparation_preserves_identifier(tmp_path, failure):
    env, snapshot, _ = _prepare(tmp_path)
    old_snapshot = tmp_path / "old-snapshot"
    old_snapshot.mkdir()
    identifier = tmp_path / "Qwen" / "Qwen3-8B"
    identifier.parent.mkdir()
    identifier.symlink_to(old_snapshot, target_is_directory=True)
    if failure == "download":
        env["STUB_FAIL"] = "1"
    elif failure == "helper":
        env["STUB_HELPER_FAIL"] = "1"
    else:
        (snapshot / "config.json").unlink()
    result = _run(tmp_path, env)
    assert result.returncode != 0
    assert identifier.resolve() == old_snapshot
    assert not list(old_snapshot.iterdir())
