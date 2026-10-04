"""Tests for the baseline skeleton KISS spec (01-baseline-skeleton-kiss-spec)."""

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEPLOY_CONFIG_DIR = REPO_ROOT / "config" / "deployment_configurations"

SUPPORTED_MODEL = "Qwen/Qwen3-0.6B"
SUPPORTED_HARDWARE = "huawei-cpu"

UNSUPPORTED_MODEL = "org/does-not-exist"
UNSUPPORTED_HARDWARE = "not-a-hardware"


def _run_deploy(*args: str) -> subprocess.CompletedProcess:
    """Run deploy.py with the given Hydra CLI args."""
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "deploy.py"), *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


# --- T2: unsupported pair must fail clearly and not start a server (R4, B1, B2) ---


def test_t2_unsupported_model(capsys):
    result = _run_deploy(f"model={UNSUPPORTED_MODEL}", f"hardware={SUPPORTED_HARDWARE}")
    assert result.returncode != 0
    assert UNSUPPORTED_MODEL in (result.stderr + result.stdout)
    assert "unsupported" in (result.stderr + result.stdout).lower()


def test_t2_unsupported_hardware(capsys):
    result = _run_deploy(f"model={SUPPORTED_MODEL}", f"hardware={UNSUPPORTED_HARDWARE}")
    assert result.returncode != 0
    assert UNSUPPORTED_HARDWARE in (result.stderr + result.stdout)
    assert "unsupported" in (result.stderr + result.stdout).lower()


# --- T3: deployment-configuration library shape (R2) ---


def test_t3_config_directory_non_empty():
    files = list(DEPLOY_CONFIG_DIR.glob("*.yaml"))
    assert files, f"no deployment configurations in {DEPLOY_CONFIG_DIR}"


def test_t3_every_entry_defines_identifiers_and_options():
    for path in DEPLOY_CONFIG_DIR.glob("*.yaml"):
        entry = yaml.safe_load(path.read_text())
        assert isinstance(entry.get("model"), str) and entry["model"], path
        assert isinstance(entry.get("hardware"), str) and entry["hardware"], path
        assert "vllm_options" in entry, path


def test_t3_supported_pair_entry_exists():
    entries = [yaml.safe_load(p.read_text()) for p in DEPLOY_CONFIG_DIR.glob("*.yaml")]
    matching = [
        e
        for e in entries
        if e.get("model") == SUPPORTED_MODEL and e.get("hardware") == SUPPORTED_HARDWARE
    ]
    assert matching, (
        f"no deployment configuration for ({SUPPORTED_MODEL}, {SUPPORTED_HARDWARE})"
    )
